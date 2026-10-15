"""Pure routing contracts; no default audio endpoints are allowed."""
from dataclasses import dataclass
from collections import deque
import numpy as np


class Segmenter:
    """20 ms energy gate with bounded turns; ASR applies neural VAD later."""
    def __init__(self, threshold=0.01, quiet_ms=600, max_ms=8000):
        self.threshold = threshold
        self.quiet_samples = max(320, int(quiet_ms * 16))
        self.max_samples = max(4000, int(max_ms * 16))
        self.reset()

    def reset(self):
        self.pending = np.empty(0, dtype=np.float32)
        self.preroll = deque(maxlen=10)
        self.active = []
        self.length = self.voiced = self.quiet = 0

    def push(self, samples):
        self.pending = np.concatenate((self.pending, np.asarray(samples, dtype=np.float32)))
        turns = []
        while len(self.pending) >= 320:
            frame = self.pending[:320].copy()
            self.pending = self.pending[320:]
            speaking = float(np.sqrt(np.mean(frame * frame))) >= self.threshold
            if not self.active and not speaking:
                self.preroll.append(frame)
                continue
            if not self.active:
                self.active = list(self.preroll)
                self.length = sum(len(part) for part in self.active)
                self.preroll.clear()
            self.active.append(frame)
            self.length += len(frame)
            self.voiced += len(frame) if speaking else 0
            self.quiet = 0 if speaking else self.quiet + len(frame)
            if self.quiet >= self.quiet_samples or self.length >= self.max_samples:
                if self.voiced >= 4000:
                    turns.append(np.concatenate(self.active)[:self.max_samples])
                self.active = []
                self.length = self.voiced = self.quiet = 0
        return turns


@dataclass(frozen=True)
class Headset:
    key: str
    microphone_id: str
    output_id: str
    label: str


@dataclass(frozen=True)
class Route:
    speaker: int
    listener: int
    source_language: str
    target_language: str
    microphone_id: str
    output_id: str


def make_routes(first: Headset, second: Headset) -> tuple[Route, Route]:
    for headset in (first, second):
        if not all((headset.key, headset.microphone_id, headset.output_id)):
            raise ValueError("Select two complete microphone/headphone pairs")
    if (first.key == second.key or first.microphone_id == second.microphone_id
            or first.output_id == second.output_id):
        raise ValueError("Each person needs a different USB headset")
    return (
        Route(0, 1, "fi", "en", first.microphone_id, second.output_id),
        Route(1, 0, "en", "fi", second.microphone_id, first.output_id),
    )
