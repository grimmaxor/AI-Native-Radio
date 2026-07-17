#!/usr/bin/env python3
"""
diag.py — modulation-agnostic link-health probe.

Ported from latest_working/pluto_link_debug.py:174-221 (iq_to_packets_diag),
stripped to the part that does NOT need a payload decode: it reports amplitude,
coarse CFO, and the BPSK-preamble correlation peak, so it works regardless of the
(learned) payload constellation.

Used in Phase 3 (loopback) and Phase 4/5 (over-the-air) to answer "is there signal,
and is it just a level/CFO problem?" BEFORE trusting BER — distinguishes
  no RF        (raw_amplitude tiny)
  RF, no lock  (amplitude OK, corr_peak << PREAMBLE_LEN)
  good link    (corr_peak near PREAMBLE_LEN)
"""

import numpy as np
from scipy.signal import lfilter, find_peaks

import sync
from waveform import FILT, SPS


def link_diag(iq):
    """raw IQ -> dict {raw_amplitude, cfo_hz, corr_peak, corr_peak_norm,
    n_candidates}.  corr_peak_norm is corr_peak / PREAMBLE_LEN in [0,1]:
    ~1.0 = clean preamble lock, the L*0.6 detection bar is 0.6."""
    L = sync.PREAMBLE_LEN
    raw_amp = float(np.max(np.abs(iq))) if len(iq) else 0.0
    d = {'raw_amplitude': raw_amp, 'cfo_hz': 0.0, 'corr_peak': 0.0,
         'corr_peak_norm': 0.0, 'n_candidates': 0}
    if raw_amp < 1e-9:
        return d

    iqn = (iq / raw_amp).astype(np.complex64)
    # CFO from the BPSK preamble's repeat structure (modulation-agnostic), then
    # report the correlation peak on the CFO-corrected signal.
    cfo_hz, corrected = sync.cfo_correct(iqn, SPS, FILT)
    d['cfo_hz'] = cfo_hz

    best_peak = 0.0
    n_cand = 0
    for sig in ([iqn] if corrected is iqn else [iqn, corrected]):
        fi = lfilter(FILT, 1.0, sig.real).astype(np.float32)
        fq = lfilter(FILT, 1.0, sig.imag).astype(np.float32)
        filt = (fi + 1j * fq).astype(np.complex64)
        corr = sync.strided_preamble_corr(filt, SPS)
        if len(corr) == 0:
            continue
        mag = np.abs(corr)
        peaks, _ = find_peaks(mag, height=L * 0.6, distance=L * SPS)
        n_cand += len(peaks)
        pk = float(np.max(mag)) if len(mag) else 0.0
        if pk > best_peak:
            best_peak = pk
    d['corr_peak'] = best_peak
    d['corr_peak_norm'] = best_peak / L
    d['n_candidates'] = int(n_cand)
    return d


def format_diag(d):
    return (f"amp={d['raw_amplitude']:8.1f}  cfo={d['cfo_hz']:+8.1f}Hz  "
            f"corr={d['corr_peak']:5.1f}/{sync.PREAMBLE_LEN} "
            f"({d['corr_peak_norm']:.2f})  cand={d['n_candidates']}")
