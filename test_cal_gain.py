#!/usr/bin/env python3
"""Offline mock-radio test for cal.py's TX/RX gain-adaptation fixes (no hardware).

cal.py's own selftest() (`python3 cal.py`) verifies the frame CODEC through a noisy
channel. It never drives calibrate_tx/calibrate_rx themselves, so it can't catch the
bug this covers: TX's ack-listening gain was hardcoded to 40 dB and never adapted,
so a weak freq_ctrl return path made TX sweep forever and fall back, even though RX
had already locked. This drives the real state machines against a scripted MockRadio.
"""
import contextlib
import io

import cal


class MockRadio:
    """Fakes just enough of pluto_io.Radio for calibrate_tx/calibrate_rx.
    `incoming_fn(radio) -> iq array or None` scripts what radio.rx() returns."""

    def __init__(self, incoming_fn):
        self.rx_gain = None
        self.tx_atten = None
        self._incoming_fn = incoming_fn

    def set_freqs(self, tx_freq, rx_freq): pass
    def set_rx_buffer(self, n): pass
    def rx_gain_limits(self): return (-1.0, 73.0)
    def set_rx_gain(self, g): self.rx_gain = g
    def set_tx_atten(self, a): self.tx_atten = a
    def tx_cyclic_load(self, iq): pass
    def rx(self): return self._incoming_fn(self)


def test_tx_raises_ack_gain_on_weak_return_path():
    """TX must raise its ack-listen gain when repeated sweeps get no ack, instead
    of staying pinned at a fixed 40 dB and eventually giving up."""
    enc, dec = cal.load_model()
    GAIN_THRESHOLD = 55   # ACK only "arrives" once TX's listening gain clears this

    def incoming(radio):
        if radio.rx_gain is None or radio.rx_gain < GAIN_THRESHOLD:
            return None
        return cal.tile_to_buffer(cal.make_cal_frame('ACK', -20, enc))

    radio = MockRadio(incoming)
    orig_timeout = cal.CAL_TIMEOUT_SECS
    cal.CAL_TIMEOUT_SECS = 5           # shrink the 150s real budget for a fast test
    try:
        _, chosen = cal.calibrate_tx(radio, enc, dec, freq=915e6, freq_ctrl=2437e6)
    finally:
        cal.CAL_TIMEOUT_SECS = orig_timeout

    assert chosen != cal.CAL_FALLBACK_ATTEN, (
        f"TX fell back to {chosen} dB instead of locking — gain never reached the "
        f"simulated {GAIN_THRESHOLD} dB threshold (stuck at {radio.rx_gain})")
    assert radio.rx_gain >= GAIN_THRESHOLD, (
        f"TX locked without raising gain to {GAIN_THRESHOLD} dB (stuck at "
        f"{radio.rx_gain}) — adaptive gain step isn't working")
    print(f"OK: TX raised ack-listen gain to {radio.rx_gain} dB and locked atten="
          f"{chosen} dB")


def test_rx_warns_when_tx_done_never_arrives():
    """RX must print an explicit warning, not a silent '✓ calibration complete',
    when it never hears TX's DONE confirmation — the exact asymmetry seen on-air."""
    enc, dec = cal.load_model()

    def incoming(radio):
        # RX always hears a TONE (locks fast); TX's DONE never arrives.
        return cal.tile_to_buffer(cal.make_cal_frame('TONE', -20, enc))

    radio = MockRadio(incoming)
    orig_timeout, orig_dwell = cal.CAL_TIMEOUT_SECS, cal.CAL_RX_DWELL_SECS
    cal.CAL_TIMEOUT_SECS = 5
    cal.CAL_RX_DWELL_SECS = 0.2
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cal.calibrate_rx(radio, enc, dec, freq=915e6, freq_ctrl=2437e6)
    finally:
        cal.CAL_TIMEOUT_SECS, cal.CAL_RX_DWELL_SECS = orig_timeout, orig_dwell

    out = buf.getvalue()
    assert "TX DONE not seen" in out, (
        "RX did not warn about a missing DONE — it silently claims "
        "'calibration complete' even when TX never confirmed:\n" + out)
    print("OK: RX warned about missing TX DONE instead of a silent false success")


if __name__ == "__main__":
    test_tx_raises_ack_gain_on_weak_return_path()
    test_rx_warns_when_tx_done_never_arrives()
    print("\nALL CAL-GAIN TESTS PASS")
