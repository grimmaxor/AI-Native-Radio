#!/usr/bin/env python3
"""
test_offline.py — Phase 0-2 verification, no hardware needed.

Confirms the framing/sync/equalize layer (waveform.py + sync.py) does not break what
the autoencoder learned (model.py + train_sim.py), under simulated RF impairments
(AWGN, random CFO, random delay) injected between encode and decode.

Run:  python3 test_offline.py
If encoder.pt/decoder.pt are missing it trains a quick model first (use
train_sim.py for the full run + BLER curve).
"""

import os
import numpy as np
import torch

import sync
from model import TXEncoder, RXDecoder, K_BITS
from waveform import encode_to_waveform, decode_from_waveform, SPS
import train_sim

SEED = 1234


def get_model(quick_steps=8000):
    enc, dec = TXEncoder(), RXDecoder()
    if os.path.exists("encoder.pt") and os.path.exists("decoder.pt"):
        enc.load_state_dict(torch.load("encoder.pt", map_location="cpu"))
        dec.load_state_dict(torch.load("decoder.pt", map_location="cpu"))
        print("[test] loaded encoder.pt / decoder.pt")
    else:
        print(f"[test] no checkpoint -> quick-training {quick_steps} steps "
              "(run train_sim.py for the full model + BLER curve)")
        enc, dec = train_sim.train(steps=quick_steps, batch=512, device="cpu",
                                   seed=0)
    enc.eval(); dec.eval()
    return enc, dec


def apply_impairments(iq, snr_db=None, cfo_hz=0.0, delay=0, amp=1.0, rng=None):
    """Simulate an RF channel between encode and decode."""
    rng = rng or np.random.default_rng()
    x = iq.astype(np.complex64) * amp
    if cfo_hz:
        t = np.arange(len(x)) / sync.SAMPLE_RATE
        x = (x * np.exp(1j * 2 * np.pi * cfo_hz * t)).astype(np.complex64)
    if delay:
        x = np.concatenate([np.zeros(delay, dtype=np.complex64), x])
    # trailing silence so the matched filter has runway and the frame isn't at the edge
    x = np.concatenate([x, np.zeros(4 * SPS, dtype=np.complex64)]).astype(np.complex64)
    if snr_db is not None:
        sig_p = np.mean(np.abs(x) ** 2) + 1e-12
        npow = sig_p / (10 ** (snr_db / 10.0))
        n = np.sqrt(npow / 2) * (rng.standard_normal(len(x)) +
                                 1j * rng.standard_normal(len(x)))
        x = (x + n).astype(np.complex64)
    return x


def ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 1.0
    return float(np.mean(tx_bits[:n] != rx_bits[:n]))


def run_case(name, enc, dec, n_msg=64, trials=20, **imp):
    rng = np.random.default_rng(SEED)
    n_bits = n_msg * K_BITS
    tot = 0.0
    found = 0
    for _ in range(trials):
        bits = rng.integers(0, 2, n_bits).astype(np.uint8)
        iq, nm = encode_to_waveform(bits, enc)
        rx = apply_impairments(iq, rng=rng, **imp)
        out, info = decode_from_waveform(rx, dec, enc, n_msg=nm, n_bits=n_bits)
        tot += ber(bits, out)
        found += int(info['found'])
    avg = tot / trials
    print(f"  {name:32s} BER={avg:.4e}  lock={found}/{trials}")
    return avg


def main():
    torch.manual_seed(SEED)
    enc, dec = get_model()

    print("\n[test] waveform round-trip (Phase 2):")
    results = {}
    results['noiseless'] = run_case("noiseless", enc, dec)
    results['delay+amp'] = run_case("delay=37, amp=0.5 (no noise)", enc, dec,
                                    delay=37, amp=0.5)
    results['cfo'] = run_case("cfo=+120 Hz (no noise)", enc, dec, cfo_hz=120.0)
    results['cfo_big'] = run_case("cfo=+6500 Hz (coarse acq)", enc, dec, cfo_hz=6500.0)
    results['snr15'] = run_case("AWGN 15 dB + cfo + delay", enc, dec,
                                snr_db=15, cfo_hz=80.0, delay=11)
    results['snr8'] = run_case("AWGN 8 dB + cfo + delay", enc, dec,
                               snr_db=8, cfo_hz=80.0, delay=11)
    # pure noise must NOT lock (no preamble present) — guards the lock gate
    rng = np.random.default_rng(SEED)
    noise_locks = 0
    for _ in range(20):
        noise = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype(np.complex64)
        _, ninfo = decode_from_waveform(noise, dec, enc, n_msg=64, n_bits=256)
        noise_locks += int(ninfo['found'])
    print(f"  {'noise (should NOT lock)':32s} false-locks={noise_locks}/20")

    print("\n[test] assertions:")
    ok = True
    # noiseless / clean-impairment cases must be essentially perfect
    for k in ('noiseless', 'delay+amp', 'cfo', 'cfo_big'):
        passed = results[k] < 1e-3
        ok &= passed
        print(f"  {k:12s} BER<1e-3 : {'PASS' if passed else 'FAIL'} "
              f"({results[k]:.2e})")
    passed = noise_locks == 0
    ok &= passed
    print(f"  noise        no-lock  : {'PASS' if passed else 'FAIL'} "
          f"({noise_locks}/20 false locks)")
    # moderate SNR should still be solid
    passed = results['snr15'] < 5e-2
    ok &= passed
    print(f"  snr15        BER<5e-2 : {'PASS' if passed else 'FAIL'} "
          f"({results['snr15']:.2e})")

    print("\n" + ("ALL OFFLINE TESTS PASS" if ok else "OFFLINE TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
