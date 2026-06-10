import json
import multiprocessing as mp
import time
from contextvars import ContextVar
from queue import Empty, Full
from typing import Any, Optional

import psutil


_logger_context: ContextVar[Optional["Logger"]] = ContextVar(
    "logger_context", default=None
)


def log_iteration(iteration: int, gaussians_loaded=None, gaussians_total=None, loss=None, cell=None):
    ctx = _logger_context.get()
    if ctx is not None:
        ctx._log_iteration(iteration, gaussians_loaded, gaussians_total, loss, cell)


def log_event(message: str, **kwargs):
    ctx = _logger_context.get()
    if ctx is not None:
        ctx._log_event(message, **kwargs)


class Logger:
    def __init__(self, pid: int, output_path: str, interval_ms: int = 100):
        self.pid = pid
        self.output_path = output_path
        self.interval_ms = interval_ms
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._token = None

    def __enter__(self):
        self._queue = mp.Queue(maxsize=10000)
        self._process = mp.Process(
            target=_writer_loop,
            args=(self.pid, self.output_path, self.interval_ms, self._queue),
        )
        self._process.start()
        self._token = _logger_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
        if self._queue is not None:
            self._queue.close()
            self._queue.cancel_join_thread()
        if self._token is not None:
            _logger_context.reset(self._token)
        return False

    def _log_iteration(self, iteration: int, gaussians_loaded=None, gaussians_total=None, loss=None, cell=None):
        if self._queue is not None:
            try:
                self._queue.put_nowait(("iteration", time.time(), iteration, gaussians_loaded, gaussians_total, loss, cell))
            except Full:
                pass

    def _log_event(self, message: str, **kwargs):
        if self._queue is not None:
            try:
                self._queue.put_nowait(("event", time.time(), message, kwargs))
            except Full:
                pass


def _writer_loop(pid: int, output_path: str, interval_ms: int, queue: mp.Queue):
    import pynvml

    pynvml.nvmlInit()
    start_time = time.time()

    try:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        gpu_handle = None

    target_process = psutil.Process(pid)

    def _write_snapshot(f, start_time, target_process, gpu_handle, pid):
        try:
            ram_mb = target_process.memory_info().rss / 1024 / 1024
        except psutil.NoSuchProcess:
            return

        vram_mb = 0.0
        if gpu_handle is not None:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)
                for proc in procs:
                    if proc.pid == pid:
                        vram_mb = proc.usedGpuMemory / 1024 / 1024
                        break
            except pynvml.NVMLError:
                pass

        elapsed = time.time() - start_time
        f.write(
            json.dumps(
                {
                    "type": "memory_snapshot",
                    "timestamp_s": round(elapsed, 6),
                    "ram_mb": round(ram_mb, 2),
                    "vram_mb": round(vram_mb, 2),
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()

    def _handle_queue_item(f, start_time, item):
        entry_type = item[0]
        if entry_type == "iteration":
            _, timestamp, iteration, gaussians_loaded, gaussians_total, loss, cell = item
            elapsed = timestamp - start_time
            record = {
                "type": "iteration",
                "timestamp_s": round(elapsed, 6),
                "iteration": iteration,
            }
            if cell is not None:
                record["cell"] = cell
            if gaussians_loaded is not None:
                record["gaussians_loaded"] = gaussians_loaded
            if gaussians_total is not None:
                record["gaussians_total"] = gaussians_total
            if loss is not None:
                record["loss"] = round(loss, 8)
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
        elif entry_type == "event":
            _, timestamp, message, kwargs = item
            elapsed = timestamp - start_time
            record = {
                "type": "event",
                "timestamp_s": round(elapsed, 6),
                "message": message,
            }
            record.update(kwargs)
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()

    with open(output_path, "w") as f:
        _write_snapshot(f, start_time, target_process, gpu_handle, pid)

        while True:
            while True:
                try:
                    item = queue.get_nowait()
                except Empty:
                    break
                if item is None:
                    pynvml.nvmlShutdown()
                    return
                _handle_queue_item(f, start_time, item)

            _write_snapshot(f, start_time, target_process, gpu_handle, pid)

            time.sleep(interval_ms / 1000)


def log_tensor_set(key: str, tensor, role: str = "resident"):
    import torch

    if not isinstance(tensor, torch.Tensor):
        return
    log_event(
        "tensor",
        event="set",
        key=key,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        bytes=tensor.numel() * tensor.element_size(),
        role=role,
    )


def log_tensor_delete(key: str, reason: str = "unknown"):
    log_event("tensor", event="delete", key=key, reason=reason)