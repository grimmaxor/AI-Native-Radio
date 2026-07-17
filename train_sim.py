#!/usr/bin/env python3
"""
train_sim.py — joint end-to-end training of TXEncoder + RXDecoder (simulation only).

Standard supervised deep learning, NOT online/self-modifying learning: each step
samples random messages, runs them through encoder -> differentiable channel ->
decoder, computes cross-entropy, and backprops through the WHOLE chain (the channel
is differentiable), so gradients flow from the decoder's error back into the
encoder.  That is what "jointly trains TX and RX" means.

Differentiable channel = AWGN (calibrated to a per-batch random Eb/N0) PLUS a random
phase rotation and a small random CFO ramp.  The phase/CFO augmentation is the whole
reason the learned decoder has a chance over real RF later (Phase 4): the textbook
AWGN-only autoencoder never sees the offsets a real Pluto pair has.

Outputs: encoder.pt, decoder.pt, and a printed BLER-vs-Eb/N0 table (sanity check
against the published O'Shea (7,4) curve shape).
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from model import TXEncoder, RXDecoder, M, N_CHANNEL, K_BITS

RATE = K_BITS / N_CHANNEL          # bits per channel use


def channel(c, ebno_db, phase=True, cfo=True, cfo_max=0.01, phase_max=0.35,
           device="cpu"):
    """Differentiable channel.  c: (B, n) complex symbols, mean energy ~1.

    AWGN calibrated to Eb/N0, plus a small *residual* phase offset and a small CFO
    ramp across the n channel uses per codeword.

    Phase is bounded to +-phase_max radians (~20 deg default), NOT a full 0-2pi
    rotation: the real waveform pipeline (waveform.py) derotates every payload
    symbol by the carrier phase recovered from the BPSK preamble's correlation
    BEFORE the decoder ever sees it (sync.py). Training against a full random
    rotation would force the decoder to learn a phase-invariant (capacity-limited,
    amplitude-only) code to handle a case the real pipeline already removes. This
    augmentation instead models what's actually left after that correction: preamble
    phase-estimation noise plus any residual CFO drift across the n symbols.
    """
    B, n = c.shape
    if phase:
        theta = (torch.rand(B, 1, device=device) * 2 - 1) * phase_max
        c = c * torch.exp(1j * theta)
    if cfo:
        df = (torch.rand(B, 1, device=device) * 2 - 1) * cfo_max   # cycles/symbol
        idx = torch.arange(n, device=device).float().unsqueeze(0)  # (1, n)
        c = c * torch.exp(1j * 2 * np.pi * df * idx)
    # AWGN: Es=1 (mean), Es/N0 = (Eb/N0)*R  ->  N0 = 1/(Es/N0)
    ebno = 10 ** (ebno_db / 10.0)
    esno = ebno * RATE
    n0 = 1.0 / esno
    std = np.sqrt(n0 / 2.0)
    noise = std * (torch.randn(B, n, device=device) + 1j * torch.randn(B, n, device=device))
    return c + noise


def train(steps=20000, batch=512, lr=1e-3, ebno_lo=0.0, ebno_hi=14.0,
          device="cpu", seed=0, out_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = TXEncoder().to(device)
    dec = RXDecoder().to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    enc.train(); dec.train()
    for step in range(1, steps + 1):
        msgs = torch.randint(0, M, (batch,), device=device)
        one_hot = torch.zeros(batch, M, device=device)
        one_hot[torch.arange(batch), msgs] = 1.0
        # randomize Eb/N0 per batch so the model generalizes across SNR
        ebno_db = float(np.random.uniform(ebno_lo, ebno_hi))

        tx = enc(one_hot)
        rx = channel(tx, ebno_db, device=device)
        logits = dec(rx)
        loss = ce(logits, msgs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, steps // 20) == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == msgs).float().mean().item()
            print(f"  step {step:6d}/{steps}  ebno={ebno_db:5.1f}dB  "
                  f"loss={loss.item():.4f}  acc={acc:.4f}")

    torch.save(enc.state_dict(), out_prefix + "encoder.pt")
    torch.save(dec.state_dict(), out_prefix + "decoder.pt")
    print(f"[train] saved {out_prefix}encoder.pt / {out_prefix}decoder.pt")
    return enc, dec


@torch.no_grad()
def bler_curve(enc, dec, ebno_list=range(0, 13), n_blocks=200000, batch=20000,
               device="cpu"):
    """Block-error-rate vs Eb/N0.  Shape (not exact values) should track the
    published O'Shea (7,4) autoencoder curve: monotone, ~1 decade per few dB."""
    enc.eval(); dec.eval()
    print("\n  Eb/N0(dB)    BLER")
    rows = []
    for ebno_db in ebno_list:
        errs = 0
        seen = 0
        while seen < n_blocks:
            b = min(batch, n_blocks - seen)
            msgs = torch.randint(0, M, (b,), device=device)
            one_hot = torch.zeros(b, M, device=device)
            one_hot[torch.arange(b), msgs] = 1.0
            tx = enc(one_hot)
            # evaluate against the SAME impairments the model is meant to survive
            rx = channel(tx, float(ebno_db), device=device)
            pred = dec(rx).argmax(1)
            errs += (pred != msgs).sum().item()
            seen += b
        bler = errs / seen
        rows.append((float(ebno_db), bler))
        print(f"   {ebno_db:6.1f}    {bler:.3e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ebno-lo", type=float, default=0.0)
    ap.add_argument("--ebno-hi", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curve", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  (M={M}, n={N_CHANNEL}, k={K_BITS}, "
          f"rate={RATE:.3f} bits/use)")
    enc, dec = train(steps=args.steps, batch=args.batch, lr=args.lr,
                     ebno_lo=args.ebno_lo, ebno_hi=args.ebno_hi,
                     device=device, seed=args.seed)
    if not args.no_curve:
        bler_curve(enc, dec, device=device)


if __name__ == "__main__":
    main()
