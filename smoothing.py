"""One Euro filter: removes jitter when still, stays responsive when moving fast."""
import math
import config


class OneEuro:
    def __init__(self, min_cutoff=None, beta=None, d_cutoff=1.0):
        self.min_cutoff = config.SMOOTH_MIN_CUTOFF if min_cutoff is None else min_cutoff
        self.beta = config.SMOOTH_BETA if beta is None else beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = max(t - self._t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x
