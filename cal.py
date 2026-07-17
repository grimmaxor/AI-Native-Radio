#!/usr/bin/env python3
"""
cal.py — over-the-air auto-calibration for the Phase-4 two-Pluto link.

Two FREE-RUNNING Plutos: there is no shared clock, so each side must set its own
RX gain and the two must agree on TX power before data flows.  The protocol mirrors
the proven one in claude/solo_pyth/pluto_video_stream_16qam_stable_auto.py, but every
beacon is a normal autoencoder frame (encode_to_waveform / decode_from_waveform), so
a decoded beacon proves the *data* link will decode at these settings.

Two carriers are used DURING calibration only — the TX->RX beacon on `freq` and the
RX->TX ack on `freq_ctrl` — to dodge the half-duplex lock-step hazard (both radios
TX/listen in unison and never hear each other).  After calibration both sides retune
to a single `freq` for the simplex data link.

Sequence:
  * RX fixes its RX gain (from CAL_GAIN_START) and dwells a FULL TX power-sweep at
    each gain, staying radio-silent until it decodes a beacon (so TX never locks on
    a stray ack).  Only a fully silent dwell bumps the gain one step.
  * TX free-runs an attenuation sweep weak->strong, beaconing with its current atten
    embedded in the frame, and listens for RX's ack.
  * When RX decodes the beacon it locks its gain, learns the TX atten that got
    through (from the payload) and acks that exact value back; TX locks it
    (+CAL_POWER_MARGIN).  A DONE handshake confirms both ways.

The CFO that two independent LOs introduce (tens–hundreds of Hz) is removed by the
preamble-repeat estimator in sync.py — the whole point of Phase 4 over the
single-board loopback, where one LO fed both sides and CFO was ~0.
"""

import time
import numpy as np
import torch

import sync
from model import TXEncoder, RXDecoder
from waveform import encode_to_waveform, decode_from_waveform
from pluto_io import TX_BUFFER_SIZE

# ─── calibration constants (mirror the proven _auto script) ─────────────────
CAL_RX_BUFFER    = TX_BUFFER_SIZE * 2          # bigger capture during cal
CAL_GAIN_START   = 30                          # RX gain sweep start (dB)
CAL_GAIN_STEP    = 5                           # gain bump on a silent dwell (dB)
CAL_ATTEN_SWEEP  = list(range(-40, 1, 5))      # TX atten weak -> strong (dB)
CAL_POWER_MARGIN = 5                           # extra power (less atten) once locked
CAL_ACK_CAPS     = 3                           # TX captures listened per atten step
CAL_DONE_ROUNDS  = 30                          # DONE-handshake captures
CAL_TIMEOUT_SECS = 150                         # per-role time budget
CAL_FALLBACK_ATTEN = -10                       # safe TX default on total cal failure —
                                                # the confirmed-good on-air baseline
                                                # (NOT 0 dB / max power; see NOTES.md Phase 3/4)
# ADC-clip guard (same convention as DroneMac-V2's cal_sweep.py ADC_CLIP): this project's
# own Phase-3 findings already pin raw_amplitude≈2896 as the saturation point at
# rx-gain 30/tx-atten -10 (NOTES.md), so that pair is the exact known-bad combo the
# headroom check below exists to avoid re-locking.
ADC_FULLSCALE    = 2896.0
ADC_CLIP_PCT     = 92                          # >= this projected % is treated as clipping

_CAP_SECS         = CAL_RX_BUFFER / sync.SAMPLE_RATE
CAL_TX_SWEEP_SECS = len(CAL_ATTEN_SWEEP) * CAL_ACK_CAPS * _CAP_SECS
CAL_RX_DWELL_SECS = CAL_TX_SWEEP_SECS * 1.3 + 1.0    # one full TX sweep + margin

# ─── tiny control-frame codec (rides inside a normal autoencoder frame) ──────
# 24 payload bits = 16-bit marker + 8-bit (atten+64).  16/4 + 8/4 = 6 messages.
CAL_NMSG  = 6
CAL_NBITS = 24
_MARK = {'TONE': 0xA53C, 'ACK': 0x5AC3, 'DONE': 0x3C5A}   # mutually well-separated
_ATTEN_BIAS = 64                                # store atten unsigned (atten in [-64, 63])


def _bits16(v):
    return np.array([(v >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)


def _bits8(v):
    return np.array([(v >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)


def _frame_bits(kind, atten):
    return np.concatenate([_bits16(_MARK[kind]),
                           _bits8((int(atten) + _ATTEN_BIAS) & 0xFF)]).astype(np.uint8)


def make_cal_frame(kind, atten, encoder):
    """kind in {'TONE','ACK','DONE'} + signed atten -> shaped IQ (one frame)."""
    iq, _ = encode_to_waveform(_frame_bits(kind, atten), encoder)
    return iq


def parse_cal_frame(rx, decoder, encoder, max_err=2, min_conf=0.5):
    """Decode a capture as a cal frame.  Returns (kind, atten, confidence) if the
    marker matches one of the known patterns within `max_err` bit errors and the
    decoder is confident, else None."""
    bits, info = decode_from_waveform(rx, decoder, encoder, n_msg=CAL_NMSG,
                                      n_bits=CAL_NBITS)
    if not info['found'] or len(bits) < CAL_NBITS or info['confidence'] < min_conf:
        return None
    mbits, abits = bits[:16], bits[16:24]
    atten = int(np.packbits(abits)[0]) - _ATTEN_BIAS
    for kind, val in _MARK.items():
        if int(np.sum(mbits != _bits16(val))) <= max_err:
            return kind, atten, info['confidence']
    return None


def tile_to_buffer(iq, size=TX_BUFFER_SIZE):
    """Tile WHOLE frame copies (+ silence pad) into the cyclic TX buffer so every
    repeat the partner captures is intact."""
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) >= size:
        return iq[:size].astype(np.complex64)
    n = size // len(iq)
    body = np.tile(iq, n)
    pad = np.zeros(size - len(body), dtype=np.complex64)
    return np.concatenate([body, pad]).astype(np.complex64)


def prbs_bits(seed, n_bits):
    """Deterministic test bit pattern shared by TX and RX (same seed+length -> same
    bits, so the receiver can score BER without a side channel)."""
    return np.random.default_rng(seed).integers(0, 2, n_bits).astype(np.uint8)


def load_model(prefix=""):
    enc, dec = TXEncoder(), RXDecoder()
    enc.load_state_dict(torch.load(prefix + "encoder.pt", map_location="cpu"))
    dec.load_state_dict(torch.load(prefix + "decoder.pt", map_location="cpu"))
    enc.eval(); dec.eval()
    return enc, dec


# ─── role calibrations ───────────────────────────────────────────────────────
def _recv_kind(radio, enc, dec, want, n_caps):
    """Listen up to n_caps captures; return atten of the first frame of kind `want`."""
    for _ in range(n_caps):
        raw = radio.rx()
        if raw is None:
            continue
        got = parse_cal_frame(raw, dec, enc)
        if got and got[0] == want:
            return got[1]
    return None


def calibrate_tx(radio, enc, dec, freq, freq_ctrl):
    """TX role: sweep power weak->strong, beaconing on `freq`, until RX acks on
    `freq_ctrl`; lock the exact atten RX decoded at (+margin)."""
    print("\n" + "=" * 58 + "\n  ROLE TX — calibration (sweep power until RX hears us)\n" + "=" * 58)
    radio.set_freqs(tx_freq=freq, rx_freq=freq_ctrl)
    radio.set_rx_buffer(CAL_RX_BUFFER)
    lo, hi = radio.rx_gain_limits()
    radio.set_rx_gain(int(np.clip(40, lo, hi)))      # generous: just hearing acks

    locked = None
    t0 = time.time()
    rnd = 0
    while locked is None and (time.time() - t0) < CAL_TIMEOUT_SECS:
        rnd += 1
        for atten in CAL_ATTEN_SWEEP:
            radio.set_tx_atten(atten)
            radio.tx_cyclic_load(tile_to_buffer(make_cal_frame('TONE', atten, enc)))
            time.sleep(0.05)
            radio.rx()                                # flush one stale buffer
            acked = _recv_kind(radio, enc, dec, 'ACK', CAL_ACK_CAPS)
            if acked is not None:
                locked = acked
                print(f"  [sweep {rnd}] TX atten {atten:>4} dB -> RX ACK "
                      f"(decoded our beacon at {locked} dB)")
                break
            print(f"  [sweep {rnd}] TX atten {atten:>4} dB -> no ack")
        if locked is None:
            print("  full sweep, no ack — RX still raising gain; sweeping again.")

    if locked is None:
        chosen = CAL_FALLBACK_ATTEN
        print(f"  [!] no ack within {CAL_TIMEOUT_SECS:.0f}s budget — check RX is also "
              f"calibrating, on the matching --freq/--freq-ctrl pair, and in range.  "
              f"Falling back to TX atten {chosen} dB (known-good on-air baseline, not "
              f"max power) instead of continuing to sweep.")
    else:
        chosen = int(min(0, locked + CAL_POWER_MARGIN))
    radio.set_tx_atten(chosen)
    print(f"[TX] locked TX atten = {chosen} dB; confirming link ...")

    radio.tx_cyclic_load(tile_to_buffer(make_cal_frame('DONE', chosen, enc)))
    for _ in range(CAL_DONE_ROUNDS):
        if _recv_kind(radio, enc, dec, 'DONE', 1) is not None:
            print("[TX] ✓ link confirmed both ways.")
            break
    else:
        print("[TX] RX DONE not seen — continuing anyway.")
    return int(np.clip(40, lo, hi)), chosen


def calibrate_rx(radio, enc, dec, freq, freq_ctrl, fallback_gain=40, fallback_atten=-10):
    """RX role: hold gain FIXED for a whole TX sweep, only bumping it on a silent
    dwell.  On decode, lock gain, learn the TX atten, and ack it back on `freq_ctrl`."""
    print("\n" + "=" * 58 + "\n  ROLE RX — calibration (fix gain, dwell a full TX sweep)\n" + "=" * 58)
    radio.set_freqs(tx_freq=freq_ctrl, rx_freq=freq)
    radio.set_rx_buffer(CAL_RX_BUFFER)
    radio.set_tx_atten(0)                              # ack at max power (ctrl uncalibrated)
    lo, hi = radio.rx_gain_limits()
    print(f"  valid RX gain {lo:.0f}..{hi:.0f} dB; dwelling {CAL_RX_DWELL_SECS:.1f} s/gain.")

    radio.tx_cyclic_load(np.zeros(TX_BUFFER_SIZE, dtype=np.complex64))   # stay silent

    heard_atten = None
    chosen_gain = None
    locked_pk = 0.0
    gain = int(np.clip(CAL_GAIN_START, lo, hi))
    t0 = time.time()
    while gain <= hi and (time.time() - t0) < CAL_TIMEOUT_SECS:
        radio.set_rx_gain(gain)
        time.sleep(0.05)
        radio.rx()                                    # flush after gain change
        pk = 0.0
        dwell_end = time.time() + CAL_RX_DWELL_SECS
        while time.time() < dwell_end:
            raw = radio.rx()
            if raw is None:
                continue
            pk = max(pk, float(np.max(np.abs(raw))))
            got = parse_cal_frame(raw, dec, enc)
            if got and got[0] == 'TONE':
                heard_atten = got[1]
                break
        if heard_atten is not None:
            chosen_gain = gain
            locked_pk = pk
            print(f"  RX gain {gain:>3} dB (peak {pk:>6.0f}) -> beacon decoded "
                  f"(TX atten {heard_atten} dB).")
            break
        print(f"  RX gain {gain:>3} dB (peak {pk:>6.0f}) -> silent; raising gain.")
        gain += CAL_GAIN_STEP

    if chosen_gain is None:
        chosen_gain = int(np.clip(fallback_gain, lo, hi))
        heard_atten = 0
        print(f"  [!] beacon never heard within {CAL_TIMEOUT_SECS:.0f}s budget — check TX "
              f"is also calibrating, on the matching --freq/--freq-ctrl pair, and in "
              f"range.  Falling back to RX gain {chosen_gain} dB.")
    else:
        # TX is about to boost by +CAL_POWER_MARGIN dB once it locks (see calibrate_tx) —
        # the beacon we just decoded was measured BEFORE that boost, so locking this gain
        # as-is risks re-landing on the exact rx-gain-30/tx-atten--10 saturation case
        # already documented in NOTES.md Phase 3 (amp≈2896).  Project the post-margin peak
        # and back the gain off if it would clip.
        # ponytail: linear dB->amplitude approximation (10**(dB/20)), not a live re-capture
        # at the boosted power — upgrade to an actual post-margin measurement if this proves
        # inaccurate on real hardware.
        boosted_pk = locked_pk * (10 ** (CAL_POWER_MARGIN / 20.0))
        while boosted_pk / ADC_FULLSCALE * 100 >= ADC_CLIP_PCT and chosen_gain > lo:
            chosen_gain -= CAL_GAIN_STEP
            boosted_pk /= 10 ** (CAL_GAIN_STEP / 20.0)
        boosted_adc = boosted_pk / ADC_FULLSCALE * 100
        print(f"  headroom check: TX's post-lock +{CAL_POWER_MARGIN}dB margin projects to "
              f"{boosted_adc:.0f}% ADC at RX gain {chosen_gain} dB "
              f"({'OK' if boosted_adc < ADC_CLIP_PCT else 'still tight'}).")
    radio.set_rx_gain(chosen_gain)
    print(f"[RX] locked RX gain = {chosen_gain} dB; acking (TX decoded at "
          f"{heard_atten} dB) so TX can lock power ...")

    radio.tx_cyclic_load(tile_to_buffer(make_cal_frame('ACK', heard_atten, enc)))
    for _ in range(CAL_DONE_ROUNDS):
        if _recv_kind(radio, enc, dec, 'DONE', 1) is not None:
            print("[RX] TX locked power — sending DONE.")
            break
    radio.tx_cyclic_load(tile_to_buffer(make_cal_frame('DONE', 0, enc)))
    for _ in range(5):
        radio.rx()
    print("[RX] ✓ calibration complete.")
    return int(chosen_gain), int(fallback_atten)


def calibrate(role, radio, enc, dec, freq, freq_ctrl,
              fallback_gain=40, fallback_atten=-10):
    """Run the role-appropriate calibration; returns (rx_gain, tx_atten)."""
    try:
        if role == 'tx':
            return calibrate_tx(radio, enc, dec, freq, freq_ctrl)
        return calibrate_rx(radio, enc, dec, freq, freq_ctrl,
                            fallback_gain, fallback_atten)
    finally:
        try:
            radio.sdr.tx_destroy_buffer()
        except Exception:
            pass


# ─── offline self-test of the cal-frame codec + data path (no hardware) ──────
def selftest():
    """Round-trip the cal frames and a data frame through a simulated noisy/CFO
    channel.  No radio.  Returns True on success."""
    from model import K_BITS
    enc, dec = load_model()
    rng = np.random.default_rng(0)
    Fs = sync.SAMPLE_RATE

    def channel(iq, cfo_hz=80.0, snr_db=15.0, delay=11):
        x = np.concatenate([np.zeros(delay, dtype=np.complex64), iq.astype(np.complex64)])
        t = np.arange(len(x)) / Fs
        x = (x * np.exp(1j * 2 * np.pi * cfo_hz * t)).astype(np.complex64)
        x = np.concatenate([x, np.zeros(64, dtype=np.complex64)])
        sig = np.mean(np.abs(x) ** 2) + 1e-12
        npow = sig / (10 ** (snr_db / 10.0))
        n = np.sqrt(npow / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
        return (x + n).astype(np.complex64)

    ok = True
    print("[cal selftest] control frames through CFO+AWGN channel:")
    for kind in ('TONE', 'ACK', 'DONE'):
        for atten in (-40, -15, 0):
            iq = make_cal_frame(kind, atten, enc)
            got = parse_cal_frame(channel(iq), dec, enc)
            good = got is not None and got[0] == kind and got[1] == atten
            ok &= good
            print(f"  {kind:4s} atten={atten:>4} -> {got!s:28s} {'OK' if good else 'FAIL'}")

    print("[cal selftest] data frame (64 msg) through CFO+AWGN channel:")
    n_bits = 64 * K_BITS
    bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    iq, n_msg = encode_to_waveform(bits, enc)
    out, info = decode_from_waveform(channel(iq), dec, enc, n_msg=n_msg, n_bits=n_bits)
    ber = float(np.mean(bits != out[:n_bits])) if info['found'] else 1.0
    good = info['found'] and ber < 1e-3
    ok &= good
    print(f"  lock={info['found']} cfo={info['cfo_hz']:+.1f}Hz BER={ber:.2e} "
          f"{'OK' if good else 'FAIL'}")

    print("\n" + ("CAL SELFTEST PASS" if ok else "CAL SELFTEST FAILED"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
