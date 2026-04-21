import time
import multiprocessing as mp
from contextvars import ContextVar
from typing import Optional
from queue import Empty


_training_context: ContextVar[Optional["TrainingContext"]] = ContextVar(
    "training_context", default=None
)


def log_iteration(iteration: int):
    ctx = _training_context.get()
    if ctx is not None:
        ctx._log(iteration)


class TrainingContext:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._token = None

    def __enter__(self):
        self._queue = mp.Queue()
        self._process = mp.Process(
            target=_writer_loop, args=(self._queue, self.output_path)
        )
        self._process.start()
        self._token = _training_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._queue is not None:
            self._queue.put(None)
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
        if self._token is not None:
            _training_context.reset(self._token)
        return False

    def _log(self, iteration: int):
        if self._queue is not None:
            timestamp = time.perf_counter()
            self._queue.put((timestamp, iteration))


def _writer_loop(queue: mp.Queue, output_path: str):
    with open(output_path, "w") as f:
        f.write("timestamp_s,iteration\n")
        while True:
            try:
                item = queue.get(timeout=1)
                if item is None:
                    break
                timestamp, iteration = item
                f.write(f"{timestamp:.6f},{iteration}\n")
                f.flush()
            except Empty:
                continue
