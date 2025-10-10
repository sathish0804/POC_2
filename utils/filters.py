from typing import Optional


class Ewma:
    """Simple exponential weighted moving average for scalar streams."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = float(alpha)
        self._y: Optional[float] = None

    def reset(self) -> None:
        self._y = None

    def update(self, x: Optional[float]) -> Optional[float]:
        if x is None:
            return self._y
        if self._y is None:
            self._y = float(x)
        else:
            a = self.alpha
            self._y = a * float(x) + (1.0 - a) * self._y
        return self._y


