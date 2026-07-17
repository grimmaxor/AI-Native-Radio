#!/usr/bin/env python3
"""
rx_node.py — Phase 4 RECEIVE node (one of two free-running Plutos).

Runs on the Pluto that receives.  Captures the air, runs the modulation-agnostic
sync + preamble-repeat CFO correction + LS equalizer + learned decoder
(waveform.decode_from_waveform), and scores BER against the SAME deterministic test
pattern the transmitter sends.  Prints a live per-capture diagnostic and a running
summary.

This is the real test of the autoencoder PHY across two INDEPENDENT oscillators: the
`cfo` column will now show the genuine LO offset (tens–hundreds of Hz), unlike the
single-board loopback where it sat at ~0.  A healthy link reads consistent `lock` and
low BER with `corr` near/above PREAMBLE_LEN.

Run (RX machine):
    python3 rx_node.py --ip ip:192.168.2.1 --freq 915e6
        # auto-calibrates its RX gain over the air (tx_node.py must be calibrating too)
    python3 rx_node.py --ip ip:192.168.2.1 --skip-cal --rx-gain 40
        # skip cal, use a known-good RX gain (e.g. the on-air Phase-3 value)

The --seed and --n-msg MUST match the transmitter.  Offline self-test (no radio):
    python3 rx_node.py --selftest
"""

import argparse
import numpy as np

import cal
from waveform import decode_from_waveform
from diag import link_diag, format_diag
from pluto_io import TX_BUFFER_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="ip:192.168.2.1")
    ap.add_argument("--freq", type=float, default=915_000_000, help="data carrier (Hz)")
    ap.add_argument("--freq-ctrl", type=float, default=2_437_000_000,
                    help="control carrier for cal acks (Hz)")
    ap.add_argument("--n-msg", type=int, default=64, help="codewords per frame (match TX)")
    ap.add_argument("--seed", type=int, default=0, help="PRBS seed (match TX)")
    ap.add_argument("--skip-cal", action="store_true", help="skip auto-cal; use --rx-gain")
    ap.add_argument("--rx-gain", type=int, default=40, help="RX gain dB (if --skip-cal)")
    ap.add_argument("--tx-atten", type=int, default=-10, help="TX atten dB (cal ack power)")
    ap.add_argument("--captures", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--selftest", action="store_true", help="offline round-trip, no radio")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if cal.selftest() else 1)

    enc, dec = cal.load_model()

    from pluto_io import PlutoRadio
    radio = PlutoRadio(ip=args.ip, tx_freq=args.freq, rx_freq=args.freq,
                       rx_gain=args.rx_gain, tx_atten=args.tx_atten,
                       tx_cyclic=True, rx_buffer_size=cal.CAL_RX_BUFFER)

    if args.skip_cal:
        rx_gain = args.rx_gain
        print(f"[RX] skipping calibration; RX gain = {rx_gain} dB")
    else:
        rx_gain, _txa = cal.calibrate('rx', radio, enc, dec, args.freq, args.freq_ctrl,
                                      fallback_gain=args.rx_gain, fallback_atten=args.tx_atten)

    # ── data phase: simplex on --freq, decode + score BER ──
    n_bits = args.n_msg * 4                      # K_BITS = 4
    ref = cal.prbs_bits(args.seed, n_bits)

    radio.set_freqs(tx_freq=args.freq, rx_freq=args.freq)
    radio.set_rx_gain(rx_gain)
    radio.set_rx_buffer(TX_BUFFER_SIZE)
    for _ in range(3):
        radio.rx()                               # flush stale buffers

    print(f"\n[RX] decoding on {args.freq/1e6:.3f} MHz, RX gain {rx_gain} dB, "
          f"expecting {args.n_msg} msgs ({n_bits} bits, seed {args.seed}).")
    print("[RX] Ctrl+C to stop.\n")

    bers, locks, total, cfos = [], 0, 0, []
    try:
        i = 0
        while args.captures == 0 or i < args.captures:
            i += 1
            raw = radio.rx()
            if raw is None:
                continue
            total += 1
            d = link_diag(raw)
            bits, info = decode_from_waveform(raw, dec, enc, n_msg=args.n_msg,
                                              n_bits=n_bits)
            if info['found']:
                locks += 1
                b = float(np.mean(ref != bits[:n_bits]))
                bers.append(b)
                cfos.append(info['cfo_hz'])
                print(f"  cap {i:4d}  {format_diag(d)}  slip={info['slip']:+d} "
                      f"conf={info['confidence']:.2f}  BER={b:.3e}")
            else:
                print(f"  cap {i:4d}  {format_diag(d)}  NO LOCK")
    except KeyboardInterrupt:
        pass
    finally:
        radio.close()

    print(f"\n[RX] locked {locks}/{total} captures")
    if bers:
        print(f"[RX] mean BER = {np.mean(bers):.3e}  min = {np.min(bers):.3e}  "
              f"zero-error frames = {sum(b == 0 for b in bers)}/{len(bers)}")
        print(f"[RX] mean |CFO| = {np.mean(np.abs(cfos)):.1f} Hz  "
              f"(range {np.min(cfos):+.0f}..{np.max(cfos):+.0f} Hz)")


if __name__ == "__main__":
    main()
