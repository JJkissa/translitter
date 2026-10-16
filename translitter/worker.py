"""One offline model thread; fair, bounded turns and explicit session cancellation."""
from collections import deque
from pathlib import Path
from threading import Condition

import numpy as np
from PySide6.QtCore import QThread, Signal

from .core import MAX_TURN_SAMPLES, Headset, Route, make_routes
from .engine import ModelEngine


class ModelWorker(QThread):
    ready = Signal()
    progress = Signal(int, str)
    result = Signal(object)
    settled = Signal(int, int, bool)
    error = Signal(int, str)
    loading = Signal(str)
    translation_ready = Signal(int, str, bool, str)

    def __init__(self, root: Path, model_name="whisper-turbo", device="cuda", parent=None):
        super().__init__(parent)
        self._root = Path(root)
        self._model_name = model_name
        self._device = device
        self._condition = Condition()
        self._generation = None
        self._routes = None
        self._closed = False
        self._config_revision = 0
        self._config_pending = None
        self._translation_ready = True  # Unchanged Marian supports legacy callers.
        self._pending = (deque(), deque())
        self._clear()

    def _new_generation(self, generation):
        if type(generation) is not int or generation < 0:
            raise ValueError("Session generation must be a nonnegative integer")
        if self._closed or (self._generation is not None and generation <= self._generation):
            raise ValueError("Worker closed or session generation is not newer")

    def _clear(self):
        for queue in self._pending:
            queue.clear()
        self._next_speaker = 0

    def set_translation_backend(self, backend):
        """Queue stopped-only preflight; no engine calls on the caller's thread."""
        if backend not in ("marian", "translategemma"):
            raise ValueError("Select Marian or local TranslateGemma")
        with self._condition:
            if self._closed or self._routes is not None:
                raise ValueError("Translation backend can only change while stopped")
            self._config_revision += 1
            self._config_pending = (self._config_revision, backend)
            self._translation_ready = False
            self._condition.notify_all()
            return self._config_revision

    def set_session(self, generation: int, routes: tuple[Route, Route]):
        if len(routes) != 2 or not all(isinstance(route, Route) for route in routes):
            raise ValueError("Exactly two validated routes are required")
        expected = make_routes(
            Headset("first", routes[0].microphone_id, routes[1].output_id, "first"),
            Headset("second", routes[1].microphone_id, routes[0].output_id, "second"),
            routes[0].source_language, routes[1].source_language)
        if tuple(routes) != expected:
            raise ValueError("Invalid cross-headset routes")
        with self._condition:
            self._new_generation(generation)
            if not self._translation_ready:
                raise ValueError("Translation backend is not ready; select a ready local backend")
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
            if samples.ndim != 1 or not 0 < len(samples) <= MAX_TURN_SAMPLES or not np.isfinite(samples).all():
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
        engine = None
        try:
            engine = ModelEngine(self._root, self._model_name, self._device)
            engine.load(self.loading.emit)
            with self._condition:
                if self._closed:
                    return
                self.ready.emit()
            self._run_jobs(engine)
        except Exception as exc:
            with self._condition:
                if not self._closed:
                    self.error.emit(-1, f"Model loading failed ({type(exc).__name__})")
        finally:
            with self._condition:
                self._closed = True
                self._routes = None
                self._config_pending = None
                self._translation_ready = False
                self._clear()
                self._condition.notify_all()
            if engine is not None and hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception:
                    pass  # Never expose runtime output during terminal cleanup.

    def _run_jobs(self, engine):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._config_pending is not None or any(self._pending))
                if self._closed:
                    return
                configuration = self._config_pending
                self._config_pending = None
                if configuration is None:
                    available = [bool(queue) for queue in self._pending]
                    speaker = self._next_speaker if available[self._next_speaker] else 1 - self._next_speaker
                    generation, route, samples = self._pending[speaker].popleft()
                    self._next_speaker = 1 - speaker
            if configuration is not None:
                self._configure(engine, *configuration)
                continue

            try:
                turn = engine.interpret(samples, route, generation,
                                        lambda: self._cancelled(generation),
                                        lambda text: self._progress(generation, text))
            except Exception as exc:
                with self._condition:
                    if self._valid(generation):
                        message = (str(exc) if isinstance(exc, FileNotFoundError)
                                   else f"Turn processing failed ({type(exc).__name__})")
                        self.error.emit(generation, message)
                        self.settled.emit(generation, route.speaker, False)
                continue
            with self._condition:
                if self._valid(generation):
                    if turn is not None:
                        self.result.emit(turn)
                    self.settled.emit(generation, route.speaker, turn is not None)

    def _configure(self, engine, revision, backend):
        def cancelled():
            with self._condition:
                return self._closed or revision != self._config_revision
        ready, error = False, ""
        try:
            ready = engine.set_translation_backend(backend, cancelled=cancelled,
                                                   progress=lambda text: None)
        except Exception as exc:
            error = f"Local translation backend unavailable ({type(exc).__name__}); select Marian or install the optional runtime"
        with self._condition:
            if not cancelled():
                self._translation_ready = bool(ready)
                self.translation_ready.emit(revision, backend, bool(ready), error)
