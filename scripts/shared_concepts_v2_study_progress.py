"""Отображение общего и локального прогресса в процессе-координаторе."""

from contextlib import ExitStack
import multiprocessing as mp
import os
from queue import Empty
from time import monotonic

from tqdm import tqdm


# Только эти две версии различаются выводом прогресса и совместимы по чекпойнтам.
COMPATIBLE_RUNNERS = {
    "f1b30bb44ab7730f654e8df981cd6d00dc3644b39eebe24a689fc7d8097e6406": "7649a1ff9dc7a4fcd270e030ae599f4b5a7ae0d1cb40f1895f38b4565ce3ad12",
}


class WorkerProgress:
    def __init__(self, events, point=None):
        self.events = events
        self.point = point
        self.pid = os.getpid()
        self.last_sent = 0.0

    def report(self, status, *, force=True):
        now = monotonic()
        if force or now - self.last_sent >= 0.25:
            self.events.put((self.pid, self.point, status))
            self.last_sent = now

    def track(self, items, phase):
        size = len(items)
        self.report(f"{phase} 0/{size}")
        for index, item in enumerate(items, 1):
            yield item
            self.report(f"{phase} {index}/{size}", force=index == size)


class WorkerRows:
    def __init__(self, events, workers):
        self.events = events
        self.workers = workers
        self.stack = ExitStack()
        self.pids = {}
        self.states = {}
        self.displayed = {}

    def __enter__(self):
        self.bars = [self.stack.enter_context(tqdm(
            desc=f"Worker {index + 1}: starting", position=index + 2, leave=False,
            dynamic_ncols=True, bar_format="{desc}")) for index in range(self.workers)]
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def collect(self):
        now = monotonic()
        while True:
            try:
                pid, point, status = self.events.get_nowait()
            except Empty:
                break
            if pid not in self.pids:
                if len(self.pids) == self.workers:
                    # Pool может заменить завершившийся дочерний процесс.
                    import psutil

                    inactive = next((old for old in self.pids if not psutil.pid_exists(old)), None)
                    if inactive is None:
                        continue
                    slot = self.pids.pop(inactive)
                    self.states.pop(inactive, None)
                    self.pids[pid] = slot
                else:
                    self.pids[pid] = len(self.pids)
            previous = self.states.get(pid)
            started = previous[2] if previous is not None and previous[0] == point else now
            self.states[pid] = (point, status, started)
        for pid, (point, status, started) in self.states.items():
            slot = self.pids[pid]
            details = status
            if point is not None:
                a, b, c, d, delta = (number / 100 for number in point)
                details = f"EC {a:.2f}/{b:.2f} CC {c:.2f}/{d:.2f} δ{delta:.2f} | {status} | {int(now - started)}s"
            line = f"Worker {slot + 1} [{pid}]: {details}"
            if self.displayed.get(slot) != line:
                self.bars[slot].set_description_str(line)
                self.displayed[slot] = line

    def results(self, iterator, count):
        for _ in range(count):
            while True:
                self.collect()
                try:
                    row = iterator.next(timeout=0.25)
                except mp.TimeoutError:
                    continue
                self.collect()
                yield row
                break
