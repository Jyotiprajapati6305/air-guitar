"""Guitar-like sound: Karplus-Strong string synthesis played through pygame."""
import threading
import time
import numpy as np
import config


def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def karplus_strong(freq, sr=None, seconds=None, decay=0.996, seed=0):
    """Plucked-string tone as float32 in [-1, 1]."""
    sr = sr or config.SAMPLE_RATE
    seconds = seconds or config.NOTE_SECONDS
    rng = np.random.default_rng(seed)
    n = max(2, int(round(sr / freq - 0.5)))   # averaging filter adds half a sample of delay
    total = int(sr * seconds)
    buf = rng.uniform(-1, 1, n)
    buf = 0.5 * (buf + np.roll(buf, 1))       # soften the pick attack a little
    buf -= buf.mean()                         # no DC offset (avoids clicks/thumps)
    out = np.empty(-(-total // n) * n)
    for b in range(0, len(out), n):
        out[b:b + n] = buf
        buf = decay * 0.5 * (buf + np.roll(buf, -1))
    out = out[:total]
    fade = int(0.05 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)
    out /= max(1e-9, np.abs(out).max())
    return (out * 0.9).astype(np.float32)


class AudioEngine:
    def __init__(self):
        import pygame
        self._pg = pygame
        pygame.mixer.pre_init(config.SAMPLE_RATE, -16, 1, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(24)
        self._cache = {}

    def _sound(self, midi):
        if midi not in self._cache:
            wave = karplus_strong(midi_to_freq(midi), seed=midi)
            pcm = (wave * 32767).astype(np.int16)
            self._cache[midi] = self._pg.mixer.Sound(buffer=pcm.tobytes())
        return self._cache[midi]

    def warm_up(self, midi_notes):
        for m in midi_notes:
            self._sound(m)

    def play(self, midi, velocity=1.0):
        snd = self._sound(midi)
        snd.set_volume(max(0.0, min(1.0, velocity * config.MASTER_VOLUME)))
        snd.play()

    def strum(self, midi_notes, velocity=1.0):
        """Play notes one after another a few ms apart, like a pick sweeping the strings."""
        def run():
            for i, m in enumerate(midi_notes):
                try:
                    self.play(m, velocity * (1 - 0.04 * i))
                except Exception:
                    return
                time.sleep(config.STRUM_STAGGER)
        threading.Thread(target=run, daemon=True).start()

    def close(self):
        self._pg.mixer.quit()
