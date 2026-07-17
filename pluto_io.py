#!/usr/bin/env python3
"""
pluto_io.py — minimal ADALM-Pluto setup for the AI-native radio.

Ported (simplified) from claude/dronemac/phy.py:417-603 (PlutoPhy.__init__ /
_disable_dds / send_cyclic / recv).  Kept deliberately thin: 1 MSPS, manual gain,
FDD-capable independent TX/RX LOs, and the critical DDS-disable block (the Pluto's
onboard DDS tone leaks into TX and corrupts it if left on).

No DSP here — this only moves complex64 IQ to/from the radio.  Shaping/sync live in
waveform.py / sync.py, so this stays modulation-agnostic.
"""

import os
import ctypes.util
import threading
import numpy as np

# Patch ctypes BEFORE `import adi` anywhere: pylibiio 0.25 (needs libiio 0.x symbols)
# otherwise resolves libiio.so.1 on this box and dies with
# "undefined symbol: iio_get_backends_count". Same fix as the sibling Pluto projects
# (Pluto-Video-Link-Pi5/params.py etc.) — no-op if the 0.23 lib isn't present.
_LIBIIO0 = "/usr/lib/x86_64-linux-gnu/libiio.so.0.23"
if os.path.exists(_LIBIIO0):
    _orig_find_library = ctypes.util.find_library
    ctypes.util.find_library = (
        lambda name: _LIBIIO0 if name == "iio" else _orig_find_library(name)
    )

SAMPLE_RATE = int(1e6)
TX_BUFFER_SIZE = 65536


class PlutoRadio:
    def __init__(self, ip="ip:192.168.2.1", tx_freq=2_412_000_000,
                 rx_freq=2_412_000_000, rx_gain=40, tx_atten=-20,
                 tx_cyclic=False, rx_buffer_size=65536):
        import adi
        self.lock = threading.Lock()
        self.tx_cyclic = bool(tx_cyclic)
        print(f"[pluto] connecting to {ip} ...")
        self.sdr = sdr = adi.Pluto(ip)
        sdr.sample_rate             = SAMPLE_RATE
        sdr.tx_lo                   = int(tx_freq)
        sdr.rx_lo                   = int(rx_freq)
        sdr.tx_rf_bandwidth         = SAMPLE_RATE
        sdr.rx_rf_bandwidth         = SAMPLE_RATE
        sdr.gain_control_mode_chan0 = 'manual'
        sdr.rx_hardwaregain_chan0   = int(rx_gain)
        sdr.tx_hardwaregain_chan0   = int(tx_atten)
        sdr.rx_buffer_size          = int(rx_buffer_size)
        sdr.tx_cyclic_buffer        = bool(tx_cyclic)
        self._disable_dds(ip)
        print(f"[pluto] TX {tx_freq/1e6:.3f} MHz  RX {rx_freq/1e6:.3f} MHz  "
              f"RXg={rx_gain} TXa={tx_atten} cyclic={tx_cyclic}")

    @staticmethod
    def _disable_dds(ip):
        """Kill the onboard DDS tone, otherwise it leaks into TX and corrupts it."""
        try:
            import iio
            dds = iio.Context(ip).find_device("cf-ad9361-dds-core-lpc")
            if dds:
                for ch in dds.channels:
                    if ch.output:
                        for attr in ["raw", "scale"]:
                            try:
                                ch.attrs[attr].value = "0" if attr == "raw" else "0.0"
                            except Exception:
                                pass
        except Exception:
            pass

    def set_rx_gain(self, g):  self.sdr.rx_hardwaregain_chan0 = int(g)
    def set_tx_atten(self, a): self.sdr.tx_hardwaregain_chan0 = int(a)

    def set_freqs(self, tx_freq=None, rx_freq=None):
        """Retune the TX and/or RX LO (used to swap between the beacon/control
        carriers during calibration and the single data carrier afterwards)."""
        if tx_freq is not None:
            self.sdr.tx_lo = int(tx_freq)
        if rx_freq is not None:
            self.sdr.rx_lo = int(rx_freq)

    def set_rx_buffer(self, n):
        self.sdr.rx_buffer_size = int(n)

    def rx_gain_limits(self):
        """Valid (lo, hi) RX hardware-gain range at the current LO; (0, 71) fallback."""
        try:
            ch = self.sdr._ctrl.find_channel("voltage0", False)
            nums = [float(x) for x in
                    ch.attrs["hardwaregain_available"].value.strip("[] ").split()]
            if len(nums) == 3 and nums[2] > nums[0]:
                return nums[0], nums[2]
        except Exception:
            pass
        return 0.0, 71.0

    def tx_once(self, iq):
        """One-shot (non-cyclic) transmit of a complex64 IQ buffer."""
        with self.lock:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
            self.sdr.tx_cyclic_buffer = False
            self.tx_cyclic = False
            self.sdr.tx(iq.astype(np.complex64))

    def tx_cyclic_load(self, iq):
        """Load a buffer into the cyclic TX DMA; it repeats until replaced."""
        with self.lock:
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
            self.sdr.tx_cyclic_buffer = True
            self.tx_cyclic = True
            self.sdr.tx(iq.astype(np.complex64))

    def rx(self):
        with self.lock:
            try:
                return self.sdr.rx()
            except Exception:
                return None

    def close(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass
