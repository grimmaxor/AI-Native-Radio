#!/usr/bin/env python3
"""
tx_node.py — Phase 4 TRANSMIT node (one of two free-running Plutos).

Runs on the Pluto that sends.  Encodes a deterministic test bit pattern with the
learned autoencoder, RRC-shapes + preamble-prefixes it (waveform.py), and transmits
it cyclically and continuously on a single carrier.  The matching rx_node.py decodes
it on the other Pluto and scores BER against the same pattern.

Unlike the single-board loopback (loopback_test.py), the two radios here have
INDEPENDENT local oscillators, so there is a real carrier frequency offset (tens to
hundreds of Hz).  That offset is what the preamble-repeat CFO estimator in sync.py
exists to remove — this node just transmits; all CFO work happens at the receiver.

Run (TX machine):
    python3 tx_node.py --ip ip:192.168.2.1 --freq 915e6
        # auto-calibrates power over the air first (rx_node.py must be calibrating too)
    python3 tx_node.py --ip ip:192.168.2.1 --skip-cal --tx-atten -10
        # skip cal, use a known-good TX power (e.g. the on-air Phase-3 value)

The --seed and --n-msg MUST match the receiver.  Offline self-test (no radio):
    python3 tx_node.py --selftest
"""

import argparse
import time
import numpy as np

import cal
from waveform import encode_to_waveform
from pluto_io import TX_BUFFER_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="ip:192.168.2.1")
    ap.add_argument("--freq", type=float, default=915_000_000, help="data carrier (Hz)")
    ap.add_argument("--freq-ctrl", type=float, default=2_437_000_000,
                    help="control carrier for cal acks (Hz)")
    ap.add_argument("--n-msg", type=int, default=64, help="codewords per frame (match RX)")
    ap.add_argument("--seed", type=int, default=0, help="PRBS seed (match RX)")
    ap.add_argument("--skip-cal", action="store_true", help="skip auto-cal; use --tx-atten")
    ap.add_argument("--tx-atten", type=int, default=-10, help="TX atten dB (if --skip-cal)")
    ap.add_argument("--rx-gain", type=int, default=40, help="RX gain dB (cal ack listen)")
    ap.add_argument("--selftest", action="store_true", help="offline round-trip, no radio")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if cal.selftest() else 1)

    enc, _dec = cal.load_model()
    dec = _dec

    from pluto_io import PlutoRadio
    radio = PlutoRadio(ip=args.ip, tx_freq=args.freq, rx_freq=args.freq,
                       rx_gain=args.rx_gain, tx_atten=args.tx_atten,
                       tx_cyclic=True, rx_buffer_size=cal.CAL_RX_BUFFER)

    if args.skip_cal:
        tx_atten = args.tx_atten
        print(f"[TX] skipping calibration; TX atten = {tx_atten} dB")
    else:
        _rxg, tx_atten = cal.calibrate('tx', radio, enc, dec, args.freq, args.freq_ctrl)

    # ── data phase: simplex on --freq, transmit the PRBS frame cyclically ──
    n_bits = args.n_msg * 4                      # K_BITS = 4
    bits = cal.prbs_bits(args.seed, n_bits)
    iq, n_msg = encode_to_waveform(bits, enc)
    buf = cal.tile_to_buffer(iq, TX_BUFFER_SIZE)

    radio.set_freqs(tx_freq=args.freq, rx_freq=args.freq)
    radio.set_tx_atten(tx_atten)
    radio.tx_cyclic_load(buf)
    print(f"\n[TX] transmitting: {n_msg} msgs ({n_bits} bits, seed {args.seed}) "
          f"on {args.freq/1e6:.3f} MHz at TX atten {tx_atten} dB, cyclic.")
    print("[TX] Ctrl+C to stop.")
    try:
        t0 = time.time()
        while True:
            time.sleep(5)
            print(f"  [TX] still transmitting ({time.time()-t0:.0f}s) — "
                  f"frame repeats every {len(iq)} samples in the {TX_BUFFER_SIZE}-sample buffer.")
    except KeyboardInterrupt:
        pass
    finally:
        radio.close()
        print("\n[TX] stopped.")


if __name__ == "__main__":
    main()
