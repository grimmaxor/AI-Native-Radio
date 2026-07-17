#!/usr/bin/env python3
"""
waveform.py — bridge between the autoencoder symbols and a real IQ waveform.

  encode_to_waveform(bits, encoder)         bits -> learned TX symbols -> RRC-shaped,
                                            preamble-prefixed complex64 IQ for sdr.tx()
  decode_from_waveform(rx, decoder, encoder) IQ -> CFO/matched-filter/preamble sync ->
                                            LS equalize -> RXDecoder -> bits

Framing: a fixed number of messages per frame (n_msg).  The learned codeword is
N_CHANNEL complex symbols, so the payload is n_msg*N_CHANNEL symbols, preceded by the
real-BPSK Barker preamble from sync.py.  There is no byte header (the autoencoder
carries raw messages, not framed bytes), so v1 assumes the receiver knows n_msg — a
known PRBS/test pattern for BER scoring (Phase 4).  A learned/explicit length field
is a later refinement.

Sync is modulation-agnostic (sync.py): one complex correlation against the BPSK
preamble gives timing + residual phase, so nothing here assumes QPSK/16QAM/etc.
The LS step and fast-decode framing are ported in spirit from
dronemac/phy.py:299-411 / latest_working/pluto_video_stream_16qam_stable.py:208-334,
adapted to the learned constellation (the "ideal" reference is the encoder's own
re-encoding of the argmax-decoded message instead of a fixed QAM grid).
"""

import numpy as np
import torch
from scipy.signal import lfilter, find_peaks

import sync
from model import (bits_to_messages, messages_to_bits, messages_to_onehot,
                   N_CHANNEL, M)

SPS = 4                          # samples per symbol (smaller than the 16 the file
                                 # scripts use; the autoencoder is sps-independent) —
                                 # matches DroneMac-V2's locked-in on-air sps=4 baseline
FILT = sync.make_filter(SPS)
DAC_SCALE = 0.8 * 2 ** 15        # leave headroom below the 12-bit DAC full scale
MIN_LOCK_NORM = 0.6              # a "lock" needs a preamble peak >= 0.6*PREAMBLE_LEN
                                 # (matches the find_peaks detection bar); below this
                                 # the correlation max is just noise, so report NO
                                 # LOCK instead of decoding it (BER 0.5 garbage)


# ═══════════════════════════════════════════════════════════════════════════
#  TX:  bits -> shaped IQ
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_to_waveform(bits, encoder, scale_to_dac=True):
    """bits -> complex64 IQ (preamble + RRC-shaped learned payload).

    Returns (iq, n_msg) so the receiver/test knows how many codewords to expect."""
    encoder.eval()
    msgs = bits_to_messages(bits)                       # (n_msg,) message indices
    n_msg = len(msgs)
    one_hot = torch.from_numpy(messages_to_onehot(msgs))
    payload = encoder(one_hot).cpu().numpy().reshape(-1).astype(np.complex64)  # (n_msg*n,)

    preamble = sync.PREAMBLE_SIGNS.astype(np.complex64)
    syms = np.concatenate([preamble, payload]).astype(np.complex64)

    # +len(FILT) zeros of runway so the causal FIR fully shapes the last symbols.
    up = np.zeros(len(syms) * SPS + len(FILT), dtype=np.complex64)
    up[::SPS][:len(syms)] = syms
    si = lfilter(FILT, 1.0, up.real).astype(np.float32)
    sq = lfilter(FILT, 1.0, up.imag).astype(np.float32)
    shaped = (si + 1j * sq).astype(np.complex64)

    if scale_to_dac:
        mx = np.max(np.abs(shaped))
        if mx > 0:
            shaped = (shaped / mx * DAC_SCALE).astype(np.complex64)
    return shaped, n_msg


# ═══════════════════════════════════════════════════════════════════════════
#  RX:  shaped IQ -> bits
# ═══════════════════════════════════════════════════════════════════════════
def _ls_equalize(ds, encoder, decoder, chan_gain):
    """Decision-directed LS equalizer over the (already preamble-derotated) payload
    symbols `ds`.  Seeds amplitude from the preamble (`chan_gain`), hard-decodes each
    codeword with the decoder, re-encodes the decided message with the encoder to get
    the ideal symbols, then forms Sum(rx . conj(ideal)) for a joint residual
    phase+gain estimate and re-decodes once.  Adapted from phy.py:299-347.

    Returns (messages, confidence) for the equalized decode.
    """
    n = N_CHANNEL
    groups = ds.reshape(-1, n)                          # (n_msg, n) complex
    # initial decode with preamble-seeded gain only
    norm = groups / max(chan_gain, 1e-9)
    msgs, conf = _decode_groups(norm, decoder)
    # decision-directed residual phase+gain from the re-encoded ideal symbols
    const = encoder.constellation()                     # (M, n) complex
    ideal = const[msgs]                                 # (n_msg, n)
    rx_flat = groups.reshape(-1)
    id_flat = ideal.reshape(-1)
    corr_ls = np.sum(rx_flat * np.conj(id_flat))
    p = np.sum(np.abs(id_flat) ** 2)
    if p > 1e-9:
        derot = np.exp(-1j * np.angle(corr_ls))
        gain = np.abs(corr_ls) / p
        norm2 = groups * derot / max(gain, 1e-9)
        msgs2, conf2 = _decode_groups(norm2, decoder)
        if conf2 >= conf:                               # keep the eq pass only if it helped
            return msgs2, conf2
    return msgs, conf


@torch.no_grad()
def _decode_groups(groups, decoder):
    """groups: (n_msg, n) complex -> (messages, mean_confidence)."""
    decoder.eval()
    c = torch.from_numpy(np.ascontiguousarray(groups.astype(np.complex64)))
    logits = decoder(c)
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    return pred.cpu().numpy().astype(np.int64), float(conf.mean().item())


def decode_from_waveform(rx, decoder, encoder, n_msg, n_bits=None):
    """complex64 IQ -> recovered bits.

    Returns (bits, info) where info carries diagnostics (corr_peak, cfo_hz, slip,
    confidence).  Picks the timing slip with the highest decoder confidence — a
    modulation-agnostic selector that needs no CRC.
    """
    info = {'corr_peak': 0.0, 'cfo_hz': 0.0, 'slip': None, 'confidence': 0.0,
            'found': False}
    rx = np.asarray(rx, dtype=np.complex64)
    peak = np.max(np.abs(rx))
    if peak < 1e-6:
        return np.zeros(n_bits or 0, dtype=np.uint8), info
    rx = (rx / peak).astype(np.complex64)
    sps = SPS
    L = sync.PREAMBLE_LEN
    need = n_msg * N_CHANNEL
    # CFO from the preamble's own repeat structure (Schmidl-Cox over the Barker x3
    # preamble) — modulation-agnostic, so it works on the learned payload where the
    # old Mth-power FFT estimator did not.  Try both the uncorrected signal and the
    # CFO-corrected copy; the decoder-confidence selector below keeps whichever
    # decodes better (no CRC needed).
    cfo_hz, rx_corr = sync.cfo_correct(rx, sps, FILT)
    info['cfo_hz'] = cfo_hz
    # Try the CFO-corrected signal FIRST: the preamble estimate is reliable, so its
    # decode wins the confidence selector and the early-out below triggers on the
    # right hypothesis.  (Leading with raw `rx` let an over-confident wrong decode of
    # the still-rotated signal break the loop before the corrected variant was tried.)
    variants = [rx] if rx_corr is rx else [rx_corr, rx]

    best = None     # (confidence, msgs, slip, corr_peak)
    for corrected in variants:
        fi = lfilter(FILT, 1.0, corrected.real).astype(np.float32)
        fq = lfilter(FILT, 1.0, corrected.imag).astype(np.float32)
        filt = (fi + 1j * fq).astype(np.complex64)

        corr = sync.strided_preamble_corr(filt, sps)
        if len(corr) == 0:
            continue
        mag = np.abs(corr)
        peaks, _ = find_peaks(mag, height=L * 0.6, distance=L * sps)
        if len(peaks) == 0:
            # fall back to the single global max if no peak clears the bar
            cand = [int(np.argmax(mag))]
        else:
            cand = [int(p) for p in peaks]

        for m in cand:
            if mag[m] > info['corr_peak']:
                info['corr_peak'] = float(mag[m])
            phi = np.angle(corr[m])
            derot = np.exp(-1j * phi)
            chan_gain = mag[m] / L
            data_start = m + L * sps
            for slip in (0, 1, -1, 2, -2):
                ss = data_start + slip
                ds = filt[ss::sps][:need] * derot
                if len(ds) < need:
                    continue
                msgs, conf = _ls_equalize(ds, encoder, decoder, chan_gain)
                if best is None or conf > best[0]:
                    best = (conf, msgs, slip, float(mag[m]))
        if best is not None and best[0] > 0.5:
            break

    if best is None:
        return np.zeros(n_bits or 0, dtype=np.uint8), info

    conf, msgs, slip, cpk = best
    if cpk / sync.PREAMBLE_LEN < MIN_LOCK_NORM:
        # the best "peak" is below the detection bar -> noise, not a real frame.
        # Report NO LOCK rather than decode it (avoids 100%-lock / BER-0.5 noise).
        info['corr_peak'] = cpk
        return np.zeros(n_bits or 0, dtype=np.uint8), info
    info.update(found=True, slip=slip, confidence=conf, corr_peak=cpk)
    bits = messages_to_bits(msgs)
    if n_bits is not None:
        bits = bits[:n_bits]
    return bits, info
