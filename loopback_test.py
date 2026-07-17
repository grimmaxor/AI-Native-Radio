#!/usr/bin/env python3
"""
loopback_test.py — Phase 3: single-board hardware loopback (REQUIRES ONE PLUTO).

One Pluto, TX port -> SMA cable + attenuator -> RX port.  Transmits the
autoencoder waveform cyclically and decodes captured RX, measuring BER against the
known TX bits.  Isolates real ADC/DAC/quantization noise + timing from the harder
two-free-running-oscillator problem (that's Phase 4).

Run (with a Pluto connected and TX->RX cabled through an attenuator):
    python3 loopback_test.py --ip ip:192.168.2.1 --rx-gain 30 --tx-atten -10

Start with heavy attenuation (e.g. 30-40 dB) and back off until corr_peak (from the
diag line) is healthy without ADC saturation (raw_amplitude well under ~2000).
"""

import argparse
import time
import numpy as np
import torch

from model import TXEncoder, RXDecoder, K_BITS
from waveform import encode_to_waveform, decode_from_waveform
from pluto_io import PlutoRadio, TX_BUFFER_SIZE
from diag import link_diag, format_diag


def load_model():
    enc, dec = TXEncoder(), RXDecoder()
    enc.load_state_dict(torch.load("encoder.pt", map_location="cpu"))
    dec.load_state_dict(torch.load("decoder.pt", map_location="cpu"))
    enc.eval(); dec.eval()
    return enc, dec


def tile_to_buffer(iq, size):
    if len(iq) >= size:
        return iq[:size].astype(np.complex64)
    n = size // len(iq)
    body = np.tile(iq, n)
    pad = np.zeros(size - len(body), dtype=np.complex64)
    return np.concatenate([body, pad]).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="ip:192.168.2.1")
    ap.add_argument("--freq", type=float, default=2_412_000_000)
    ap.add_argument("--rx-gain", type=int, default=30)
    ap.add_argument("--tx-atten", type=int, default=-10)
    ap.add_argument("--n-msg", type=int, default=64)
    ap.add_argument("--captures", type=int, default=30)
    args = ap.parse_args()

    enc, dec = load_model()
    rng = np.random.default_rng(0)
    n_bits = args.n_msg * K_BITS
    tx_bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    iq, n_msg = encode_to_waveform(tx_bits, enc)
    print(f"[loopback] frame: {n_msg} msgs, {len(iq)} samples; "
          f"tiling into {TX_BUFFER_SIZE}-sample cyclic buffer")

    radio = PlutoRadio(ip=args.ip, tx_freq=args.freq, rx_freq=args.freq,
                       rx_gain=args.rx_gain, tx_atten=args.tx_atten,
                       tx_cyclic=True, rx_buffer_size=TX_BUFFER_SIZE)
    radio.tx_cyclic_load(tile_to_buffer(iq, TX_BUFFER_SIZE))
    time.sleep(0.2)
    for _ in range(3):
        radio.rx()                      # flush stale buffers

    bers, locks = [], 0
    for i in range(args.captures):
        raw = radio.rx()
        if raw is None:
            continue
        d = link_diag(raw)
        out, info = decode_from_waveform(raw, dec, enc, n_msg=n_msg, n_bits=n_bits)
        if info['found']:
            locks += 1
            b = float(np.mean(tx_bits != out[:n_bits]))
            bers.append(b)
            print(f"  cap {i:3d}  {format_diag(d)}  slip={info['slip']:+d} "
                  f"conf={info['confidence']:.2f}  BER={b:.3e}")
        else:
            print(f"  cap {i:3d}  {format_diag(d)}  NO LOCK")

    radio.close()
    print(f"\n[loopback] locked {locks}/{args.captures} captures")
    if bers:
        print(f"[loopback] mean BER = {np.mean(bers):.3e}  "
              f"min = {np.min(bers):.3e}")


if __name__ == "__main__":
    main()
