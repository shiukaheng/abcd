import time
import multiprocessing as mp
from typing import Optional

import psutil


class MemoryMonitor:
    """
    Monitors VRAM usage
    """
    def __init__(self, pid: int, output_path: str, interval_ms: int = 100):
        self.pid = pid
        self.output_path = output_path
        self.interval_ms = interval_ms
        self._stop_event: Optional[mp.Event] = None
        self._process: Optional[mp.Process] = None

    def __enter__(self):
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=_monitor_loop,
            args=(self.pid, self.output_path, self.interval_ms, self._stop_event),
        )
        self._process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
        return False


def _monitor_loop(pid: int, output_path: str, interval_ms: int, stop_event: mp.Event):
    import pynvml

    pynvml.nvmlInit()
    start_time = time.perf_counter()

    try:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        gpu_handle = None

    target_process = psutil.Process(pid)

    with open(output_path, "w") as f:
        f.write("timestamp_s,ram_mb,vram_mb\n")

        while not stop_event.is_set():
            try:
                ram_mb = target_process.memory_info().rss / 1024 / 1024

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

                timestamp = time.perf_counter() - start_time
                f.write(f"{timestamp:.6f},{ram_mb:.2f},{vram_mb:.2f}\n")
                f.flush()

            except psutil.NoSuchProcess:
                break

            time.sleep(interval_ms / 1000)

    pynvml.nvmlShutdown()
