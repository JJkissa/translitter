"""One offline model thread; fair, bounded turns and explicit session cancellation."""
from collections import deque
from pathlib import Path
from threading import Condition

import numpy as np
from PySide6.QtCore import QThread, Signal

from .core import Headset, Route, make_routes
from .engine import ModelEngine


class ModelWorker(QThread):
    ready = Signal()
    progress = Signal(int, str)
    result = Signal(object)
    settled = Signal(int, int, bool)
    error = Signal(int, str)
    loading = Signal(str)

    def __init__(self, root: Path, model_name="whisper-turbo", device="cuda", parent=None):
        super().__init__(parent)
        self._root = Path(root)
        self._model_name = model_name
        self._device = device
        self._condition = Condition()
        self._generation = None
        self._routes = None
        self._closed = False
        self._pending = (deque(), deque())
        self._next_speaker = 0

    def _new_generation(self, generation):
        if type(generation) is not int or generation < 0:
            raise ValueError("Session generation must be a nonnegative integer")
        if self._closed or (self._generation is not None and generation <= self._generation):
            raise ValueError("Worker closed or session generation is not newer")

    def _clear(self):
        for queue in self._pending:
            queue.clear()
        self._next_speaker = 0

    def set_session(self, generation: int, routes: tuple[Route, Route]):
        if len(routes) != 2 or not all(isinstance(route, Route) for route in routes):
            raise ValueError("Exactly two validated routes are required")
        expected = make_routes(
            Headset("first", routes[0].microphone_id, routes[1].output_id, "first"),
            Headset("second", routes[1].microphone_id, routes[0].output_id, "second"))
        if tuple(routes) != expected:
            raise ValueError("Invalid cross-headset routes")
        with self._condition:
            self._new_generation(generation)
            self._generation = generation
            self._routes = tuple(routes)
            self._clear()
            self._condition.notify_all()

    def stop_session(self, generation: int):
        with self._condition:
            if self._closed:
                return
            self._new_generation(generation)
            self._generation = generation
            self._routes = None
            self._clear()
            self._condition.notify_all()

    def _valid(self, generation):
        return not self._closed and self._routes is not None and generation == self._generation

    def submit(self, generation, speaker, samples):
        if type(speaker) is not int or speaker not in (0, 1) or type(generation) is not int:
            return False
        try:
            samples = np.asarray(samples)
            if samples.ndim != 1 or not 0 < len(samples) <= 128000 or not np.isfinite(samples).all():
                return False
            audio = np.array(samples, dtype=np.float32, copy=True)
            if not np.isfinite(audio).all():
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        with self._condition:
            if not self._valid(generation) or len(self._pending[speaker]) >= 2:
                return False
            self._pending[speaker].append((generation, self._routes[speaker], audio))
            self._condition.notify()
            return True

    def shutdown(self):
        with self._condition:
            self._closed = True
            self._routes = None
            self._clear()
            self._condition.notify_all()

    def _cancelled(self, generation):
        with self._condition:
            return not self._valid(generation)

    def _progress(self, generation, text):
        with self._condition:
            if self._valid(generation):
                self.progress.emit(generation, text)

    def run(self):
        with self._condition:
            if self._closed:
                return
        try:
            engine = ModelEngine(self._root, self._model_name, self._device)
            engine.load(self.loading.emit)
        except Exception as exc:
            with self._condition:
                if not self._closed:
                    self.error.emit(-1, f"Model loading failed ({type(exc).__name__})")
                self._closed = True
                self._routes = None
                self._clear()
            return
        with self._condition:
            if self._closed:
                return
            self.ready.emit()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or any(self._pending))
                if self._closed:
                    return
                speaker = self._next_speaker if self._pending[self._next_speaker] else 1 - self._next_speaker
                generation, route, samples = self._pending[speaker].popleft()
                self._next_speaker = 1 - speaker
            try:
                turn = engine.interpret(samples, route, generation,
                                        lambda: self._cancelled(generation),
                                        lambda text: self._progress(generation, text))
            except Exception as exc:
                with self._condition:
                    if self._valid(generation):
                        self.error.emit(generation, f"Turn processing failed ({type(exc).__name__})")
                        self.settled.emit(generation, route.speaker, False)
                continue
            with self._condition:
                if self._valid(generation):
                    if turn is not None:
                        self.result.emit(turn)
                    self.settled.emit(generation, route.speaker, turn is not None)
