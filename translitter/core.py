"""Pure routing contracts; no default audio endpoints are allowed."""
from dataclasses import dataclass
from collections import deque
import numpy as np

from .languages import translation_steps


MAX_TURN_SECONDS = 30
MAX_TURN_SAMPLES = 16000 * MAX_TURN_SECONDS


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
            # Widen before squaring: every finite float32 frame has a safe
            # float64 energy, including subnormal and maximum amplitudes.
            energy_frame = frame.astype(np.float64)
            speaking = float(np.sqrt(np.mean(energy_frame * energy_frame))) >= self.threshold
            while len(frame):
                if not self.active and not speaking:
                    self.preroll.append(frame)
                    break
                if not self.active:
                    self.active = list(self.preroll)
                    self.length = sum(len(part) for part in self.active)
                    # Idle context must leave room for an eligible voiced turn.
                    # Keep the newest context, trimming only silent preroll.
                    budget = self.max_samples - 4000
                    while self.length > budget:
                        part = self.active.pop(0)
                        trim = min(len(part), self.length - budget)
                        self.length -= trim
                        if trim < len(part):
                            self.active.insert(0, part[trim:])
                    self.preroll.clear()
                # Split at the limit instead of clipping a complete frame:
                # the remainder still belongs to the next automatic turn.
                size = min(len(frame), self.max_samples - self.length)
                self.active.append(frame[:size])
                frame = frame[size:]
                self.length += size
                self.voiced += size if speaking else 0
                self.quiet = 0 if speaking else self.quiet + size
                if self.quiet >= self.quiet_samples or self.length >= self.max_samples:
                    if self.voiced >= 4000:
                        turns.append(np.concatenate(self.active))
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


def make_routes(first: Headset, second: Headset, first_language="fi",
                second_language="en") -> tuple[Route, Route]:
    translation_steps(first_language, second_language)
    for headset in (first, second):
        if not all((headset.key, headset.microphone_id, headset.output_id)):
            raise ValueError("Select two complete microphone/headphone pairs")
    if (first.key == second.key or first.microphone_id == second.microphone_id
            or first.output_id == second.output_id):
        raise ValueError("Each person needs a different USB headset")
    return (
        Route(0, 1, first_language, second_language, first.microphone_id, second.output_id),
        Route(1, 0, second_language, first_language, second.microphone_id, first.output_id),
    )
