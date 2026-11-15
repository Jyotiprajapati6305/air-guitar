import numpy as np
import config
from audio import karplus_strong, midi_to_freq
from guitar import Plucker, chord_for_count, string_note, CHORD_ORDER


def test_chord_mapping_clamps():
    assert chord_for_count(0) == CHORD_ORDER[0]
    assert chord_for_count(5) == CHORD_ORDER[5]
    assert chord_for_count(99) == CHORD_ORDER[5]


def test_muted_and_open_strings():
    assert string_note("C", 0) is None      # C chord mutes low E
    assert string_note("Em", 0) == 40       # open low E
    assert string_note("G", 5) == 67        # G on high e (fret 3)


def test_synth_pitch():
    sr = config.SAMPLE_RATE
    for midi in (40, 57, 67):
        f = midi_to_freq(midi)
        w = karplus_strong(f, seconds=1.0)[: sr // 2].astype(np.float64)
        ac = np.correlate(w, w, "full")[len(w) - 1:]
        lo, hi = int(sr / (f * 1.3)), int(sr / (f * 0.77))
        lag = lo + np.argmax(ac[lo:hi])
        assert abs(sr / lag - f) / f < 0.02          # within ~2% (a third of a semitone)
        assert np.abs(w).max() <= 1.0


def test_downstroke_plays_strings_in_order():
    p = Plucker()
    x = 0.7
    p.update({"Index": (x, 0.10)}, 0.0)
    got = []
    for k in range(1, 12):                  # sweep 0.10 -> 0.90 quickly
        got += [pl.string for pl in p.update({"Index": (x, 0.10 + 0.08 * k)}, 0.03 * k)]
    assert got == sorted(got) and set(got) == set(range(6))


def test_slow_motion_and_outside_zone_are_silent():
    p = Plucker()
    p.update({"Index": (0.7, 0.2)}, 0.0)
    assert p.update({"Index": (0.7, 0.9)}, 10.0) == []        # far too slow
    p.reset()
    p.update({"Index": (0.1, 0.1)}, 0.0)
    assert p.update({"Index": (0.1, 0.9)}, 0.1) == []         # outside strum zone


def test_cooldown_blocks_double_trigger():
    p = Plucker()
    p.update({"Index": (0.7, 0.1), "Middle": (0.7, 0.1)}, 0.0)
    plucks = p.update({"Index": (0.7, 0.9), "Middle": (0.7, 0.9)}, 0.1)
    assert len(plucks) == 6                                    # each string once, not twice
