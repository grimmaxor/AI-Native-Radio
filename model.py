#!/usr/bin/env python3
"""
model.py — communications autoencoder (O'Shea/Hoydis style).

A small (k=4 bits -> M=16 messages, n=7 complex channel uses) end-to-end
autoencoder PHY.  The TX encoder invents its own constellation/coding (replacing
hand-designed QPSK/16QAM); the RX decoder learns to separate it.  Both are trained
jointly through a differentiable channel (see train_sim.py).

Deliberately tiny so it can be validated against the published O'Shea BLER curve
before scaling up.  Pure PyTorch; no hardware dependency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── default system parameters ──────────────────────────────────────────────
K_BITS = 4                 # bits per message
M = 2 ** K_BITS            # 16 messages
N_CHANNEL = 7              # complex channel uses per message ("(7,4)" autoencoder)


class TXEncoder(nn.Module):
    """one-hot message (M) -> n complex symbols, average-power constrained.

    Linear(M,M) -> ReLU -> Linear(M, 2n), reshaped to n complex symbols, then
    normalized so the mean symbol energy across the batch is 1 (average-power
    constraint).  This normalization layer is what stands in for "modulation":
    the network is free to place its 2n real coordinates anywhere, subject only to
    the power budget.
    """

    def __init__(self, m=M, n=N_CHANNEL):
        super().__init__()
        self.m, self.n = m, n
        self.net = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, 2 * n),
        )

    def forward(self, one_hot):
        x = self.net(one_hot)                       # (B, 2n) real
        x = x.view(-1, self.n, 2)                   # (B, n, 2) = I/Q
        c = torch.complex(x[..., 0], x[..., 1])     # (B, n) complex
        # average-power constraint: mean |c|^2 == 1 over the whole batch
        power = torch.mean(torch.abs(c) ** 2)
        c = c / torch.sqrt(power + 1e-12)
        return c

    @torch.no_grad()
    def constellation(self):
        """Return the n complex symbols the encoder emits for each of the M
        messages -> (M, n) complex numpy array.  Used by the waveform layer to
        re-encode for decision-directed equalization, and for plotting."""
        self.eval()
        eye = torch.eye(self.m)
        c = self.forward(eye)
        return c.detach().cpu().numpy()


class RXDecoder(nn.Module):
    """n complex symbols -> probability over M messages.

    Takes the 2n real coordinates (I/Q de-interleaved): Linear(2n, M) -> ReLU ->
    Linear(M, M) -> (logits).  Softmax/CE is applied in the loss; forward returns
    logits so it composes cleanly with cross-entropy and with argmax at inference.
    """

    def __init__(self, m=M, n=N_CHANNEL):
        super().__init__()
        self.m, self.n = m, n
        self.net = nn.Sequential(
            nn.Linear(2 * n, m),
            nn.ReLU(),
            nn.Linear(m, m),
        )

    def forward(self, c):
        # c: (B, n) complex -> (B, 2n) real
        x = torch.stack([c.real, c.imag], dim=-1).view(-1, 2 * self.n)
        return self.net(x)                          # (B, M) logits


# ─── bit <-> message helpers (fixed, not learned) ───────────────────────────
def bits_to_messages(bits, k=K_BITS):
    """Pack a flat uint8 bit array into integer message indices (0..2^k-1).
    Pads the tail with zeros to a multiple of k."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    b = bits.reshape(-1, k)
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)   # MSB first
    return (b @ weights).astype(np.int64)


def messages_to_bits(msgs, k=K_BITS):
    """Inverse of bits_to_messages (no de-padding — caller trims to known length)."""
    msgs = np.asarray(msgs, dtype=np.int64).ravel()
    shifts = np.arange(k - 1, -1, -1)
    return ((msgs[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def messages_to_onehot(msgs, m=M):
    msgs = np.asarray(msgs, dtype=np.int64).ravel()
    oh = np.zeros((len(msgs), m), dtype=np.float32)
    oh[np.arange(len(msgs)), msgs] = 1.0
    return oh
