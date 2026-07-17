#!/usr/bin/env python3
"""
sync.py — modulation-agnostic synchronization layer for the AI-native radio.

PORTED (copied + adapted, never imported) from:
  * claude/dronemac/phy.py  (BARKER_13/PREAMBLE_SIGNS, rrcosfilter/make_filter,
    _strided_preamble_corr, _cfo_variants)  — themselves ports of
    latest_working/pluto_image_fdd_raw2_qpsk.py and
    latest_working/pluto_video_stream_16qam_stable.py.

The whole point of this layer: it only needs to find a real BPSK Barker marker and
report symbol TIMING (correlation-peak position) plus residual carrier PHASE
(peak angle).  It is completely indifferent to what symbols follow the preamble —
hand-designed QPSK or a neural-net-learned constellation, it makes no difference.
A single full-rate complex correlation recovers timing AND phase in one pass, so
there is no rotation search and no payload-modulation assumption anywhere here.
"""

import numpy as np
from scipy.signal import lfilter

# ─── fixed link constants ───────────────────────────────────────────────────
SAMPLE_RATE = int(1e6)          # 1 MSPS (matches the rest of the repo)

# ─── self-referencing real-BPSK preamble (Barker-13 x3) ─────────────────────
# Always BPSK regardless of the payload's (learned) constellation.
BARKER_13      = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
PREAMBLE_SIGNS = np.tile(BARKER_13, 3).astype(np.float32)   # 39 real symbols
PREAMBLE_LEN   = len(PREAMBLE_SIGNS)


def rrcosfilter(N, alpha, Ts, Fs):
    """Root-raised-cosine FIR (verbatim from dronemac/phy.py:64-87, canonical
    latest_working/pluto_video_stream_16qam_stable.py:70-82).  Applied at BOTH TX
    shaping and RX matched filtering; the two halves convolve to a raised cosine ->
    zero ISI at the symbol sampling instants.  The learned constellation can carry
    amplitude information, so an ISI-free matched filter (not a plain Hamming FIR)
    matters here the same way it does for 16QAM."""
    T_delta = 1 / Fs
    time_idx = np.arange(-(N - 1) // 2, (N - 1) // 2 + 1) * T_delta
    h_rrc = np.zeros(len(time_idx), dtype=np.float64)
    for x in range(len(time_idx)):
        t = time_idx[x]
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and np.isclose(np.abs(t), Ts / (4 * alpha)):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha)))) + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) + 4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / (np.pi * t / Ts * (1 - (4 * alpha * t / Ts) ** 2))
    return (h_rrc / np.sqrt(np.sum(h_rrc ** 2))).astype(np.float32)


def make_filter(sps):
    """RRC (alpha=0.35, span 12 symbols)."""
    return rrcosfilter(sps * 12 + 1, 0.35, 1, sps)


def strided_preamble_corr(filt, sps):
    """Full-rate complex correlation of `filt` against the sps-spaced preamble.

        corr[m] = sum_k filt[m + k*sps] * preamble_sign[k]

    Equals the per-phase decimated preamble correlation at EVERY sample offset at
    once: one pass locates each packet's preamble (peak position => timing) AND its
    residual carrier phase (peak angle).  Ported from dronemac/phy.py:273-296.
    """
    L = PREAMBLE_LEN
    N = len(filt) - (L - 1) * sps
    if N <= 0:
        return np.zeros(0, dtype=np.complex64)
    corr = np.zeros(N, dtype=np.complex64)
    for k in range(L):
        seg = filt[k * sps: k * sps + N]
        if PREAMBLE_SIGNS[k] > 0:
            corr += seg
        else:
            corr -= seg
    return corr


def _mth_power_cfo(iq, power, analysis_samples=None):
    """Mth-power FFT CFO estimate.  Squaring/4th-powering only concentrates a clean
    spectral line at the harmonic frequency for a constant-(or grid-)modulus
    alphabet (BPSK/QPSK/16QAM).  The autoencoder's learned payload symbols have
    arbitrary amplitude/phase, so they do NOT concentrate that way; if the whole
    buffer is handed in, the (much longer) payload dilutes/corrupts the estimate.
    `analysis_samples` restricts the FFT to a leading window — pass roughly the
    preamble's sample length so the estimate is dominated by its known-BPSK energy.
    """
    seg = iq if analysis_samples is None else iq[:min(analysis_samples, len(iq))]
    nrm = seg / (np.max(np.abs(seg)) + 1e-9)
    sq = nrm ** power
    fv = np.fft.fft(sq)
    fv[0] = 0
    freqs = np.fft.fftfreq(len(sq), d=1.0 / SAMPLE_RATE)
    return freqs[int(np.argmax(np.abs(fv)))] / power


def cfo_variants(iq, power=2, analysis_samples=None):
    """Coarse CFO pre-pass (Mth-power FFT).  Yields the raw IQ first, then a
    CFO-corrected copy if a significant offset is detected.  Ported from
    dronemac/phy.py:259-270, adapted with `analysis_samples` (see _mth_power_cfo)
    so a learned-constellation payload doesn't dilute the estimate.

    `power` is the Mth-power exponent: 2 for the BPSK preamble.  We key off the
    BPSK *preamble* (not the payload), so power=2 is correct regardless of the
    learned payload constellation.
    """
    yield iq
    cfo = _mth_power_cfo(iq, power, analysis_samples)
    if abs(cfo) > 50:
        t = np.arange(len(iq)) / SAMPLE_RATE
        yield (iq * np.exp(-1j * 2 * np.pi * cfo * t)).astype(np.complex64)


def estimate_cfo_hz(iq, power=2, analysis_samples=None):
    """Scalar coarse CFO estimate in Hz (for diagnostics/logging).

    DEPRECATED for the learned constellation — superseded by estimate_cfo_preamble
    below.  Kept only for reference / the BPSK-payload variants it was ported from.
    """
    return float(_mth_power_cfo(iq, power, analysis_samples))


# ─── Schmidl-Cox CFO from the preamble's repeat structure (the right way here) ─
# The preamble is BARKER_13 tiled x3, so symbol k repeats at k+13 and k+26.  A
# constant CFO f turns copy 2 into a phase-ramped copy of copy 1:
#     r[i + Lc] ~= r[i] * exp(j*2*pi*f*Lc/Fs),   Lc = 13*sps  (one Barker, in samples)
# Because the two copies carry the SAME real-BPSK signs, the conjugate product
# r[i] * conj(r[i+Lc]) cancels the (BPSK) modulation and leaves a pure phase ramp —
# no constant-modulus assumption, and no FFT bin-resolution limit (the failure mode
# of the Mth-power estimator above on the learned payload).  Unambiguous to
# +-Fs/(2*Lc) ~= +-4.8 kHz at sps=8, far beyond the sub-kHz offsets two Plutos show.
BARKER_LEN = len(BARKER_13)             # 13 symbols per repeat


def estimate_cfo_preamble(filt, sps, peak_idx):
    """CFO (Hz) from the preamble repeat, given the matched-filtered signal `filt`
    and `peak_idx` = the preamble-start sample index (the strided_preamble_corr peak).

    Decimates `filt` at the symbol centres from `peak_idx` (where the matched RRC is
    ISI-free, so each sample is a clean preamble symbol with no pulse-shaping bleed
    from the neighbouring payload) and self-correlates symbol k against symbol k+13:
        P = sum_{k=0}^{25} conj(s[k]) * s[k+13]
    covering copies 1<->2 and 2<->3.  The aggregate phase over the known 13-symbol
    (= 13*sps-sample) lag converts directly to Hz, with no FFT bin-resolution limit
    and no constant-modulus assumption.  Returns 0.0 if the indices don't fit.
    """
    a = int(peak_idx)
    need = PREAMBLE_LEN * sps                    # 39 symbols worth of samples
    if a < 0 or a + need > len(filt):
        return 0.0
    s = filt[a: a + need: sps][:PREAMBLE_LEN]    # 39 ISI-free preamble symbols
    if len(s) < PREAMBLE_LEN:
        return 0.0
    seg1 = s[:2 * BARKER_LEN]                     # symbols 0..25
    seg2 = s[BARKER_LEN:3 * BARKER_LEN]           # symbols 13..38
    p = np.sum(np.conj(seg1) * seg2)
    if abs(p) < 1e-12:
        return 0.0
    lag_sec = (BARKER_LEN * sps) / SAMPLE_RATE    # 13-symbol lag, in seconds
    return float(np.angle(p) / (2 * np.pi * lag_sec))


def _matched_filter(iq, filt_taps):
    fi = lfilter(filt_taps, 1.0, iq.real).astype(np.float32)
    fq = lfilter(filt_taps, 1.0, iq.imag).astype(np.float32)
    return (fi + 1j * fq).astype(np.complex64)


def _apply_cfo(iq, cfo_hz):
    if abs(cfo_hz) < 1e-9:
        return iq
    t = np.arange(len(iq)) / SAMPLE_RATE
    return (iq * np.exp(-1j * 2 * np.pi * cfo_hz * t)).astype(np.complex64)


def _fine_cfo(iq, sps, filt_taps):
    """Preamble-peak + Schmidl-Cox fine CFO.  Returns (cfo_hz, corr_peak_norm), the
    latter in [0, ~1+] so callers can tell whether the preamble actually correlated."""
    filt = _matched_filter(iq, filt_taps)
    corr = strided_preamble_corr(filt, sps)
    if len(corr) == 0:
        return 0.0, 0.0
    peak = int(np.argmax(np.abs(corr)))
    norm = float(np.abs(corr[peak])) / PREAMBLE_LEN
    return estimate_cfo_preamble(filt, sps, peak), norm


def coarse_cfo_search(iq, sps, filt_taps, max_hz=20000.0, step_hz=500.0):
    """Grid-search CFO by maximizing the BPSK-preamble correlation peak.  The
    preamble is always BPSK, so it correlates regardless of the learned payload —
    this stays modulation-agnostic.  Two free-running Plutos can sit several kHz
    apart, beyond the fine estimator's +-Fs/(2*13*sps) ~= +-4.8 kHz unambiguous
    range AND beyond where the preamble correlates coherently at all; this brings
    the offset back inside that window.  `step_hz` < the preamble's frequency
    half-width (~1/(2*39*sps/Fs) ~= 1.6 kHz) so no true offset falls between grid
    points.  Returns (cfo_hz, corr_peak_norm)."""
    iq = np.asarray(iq, dtype=np.complex64)
    t = np.arange(len(iq)) / SAMPLE_RATE
    best_cfo, best_norm = 0.0, -1.0
    for f in np.arange(-max_hz, max_hz + 1, step_hz):
        d = (iq * np.exp(-1j * 2 * np.pi * f * t)).astype(np.complex64)
        filt = _matched_filter(d, filt_taps)
        corr = strided_preamble_corr(filt, sps)
        if len(corr) == 0:
            continue
        norm = float(np.max(np.abs(corr))) / PREAMBLE_LEN
        if norm > best_norm:
            best_norm, best_cfo = norm, float(f)
    return best_cfo, best_norm


def cfo_correct(iq, sps, filt_taps, min_hz=5.0, acq_norm=0.6, coarse_win=8192,
                max_hz=20000.0):
    """Estimate CFO and return (cfo_hz, derotated_iq).

    Fast path: the preamble-repeat fine estimate.  If the preamble barely correlates
    (corr_norm < acq_norm) — the signature of an offset beyond what the fine path can
    grab, as with two independent LOs — fall back to a coarse grid search over a
    leading window (cheaper than the whole buffer; it still contains a full preamble),
    then refine.  Below `min_hz` the IQ is returned unchanged.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    cfo, norm = _fine_cfo(iq, sps, filt_taps)
    if norm < acq_norm:
        win = iq[:coarse_win] if len(iq) > coarse_win else iq
        c0, cnorm = coarse_cfo_search(win, sps, filt_taps, max_hz=max_hz)
        if cnorm > norm:                          # coarse found a better preamble
            iqd = _apply_cfo(iq, c0)
            cfine, _ = _fine_cfo(iqd, sps, filt_taps)
            cfo = c0 + cfine
    if abs(cfo) < min_hz:
        return cfo, iq
    return cfo, _apply_cfo(iq, cfo)
