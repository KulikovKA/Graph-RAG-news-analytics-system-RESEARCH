"""Sample the inference client and Ollama process tree together on a shared host."""
import os
import threading
import time
import psutil


class Meter:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = 0
        self.cpu = 0.0
        self.previous = {}
        self.server_pid = None
        for c in psutil.net_connections(kind='tcp'):
            if c.status == psutil.CONN_LISTEN and c.laddr.port == 11434 and c.pid:
                self.server_pid = c.pid
                break
        if not self.server_pid:
            raise RuntimeError('Cannot locate Ollama listener for resource accounting')
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        processes = [psutil.Process(os.getpid())]
        try:
            server = psutil.Process(self.server_pid)
            processes += [server] + server.children(recursive=True)
        except psutil.Error:
            pass
        rss = 0
        for p in processes:
            try:
                key = (p.pid, p.create_time())
                times = p.cpu_times()
                cpu = times.user + times.system
                self.cpu += max(0.0, cpu - self.previous.get(key, cpu))
                self.previous[key] = cpu
                rss += p.memory_info().rss
            except psutil.Error:
                continue
        self.peak = max(self.peak, rss)

    def _run(self):
        while not self.stop_event.wait(0.25):
            self.sample()

    def __enter__(self):
        self.start = time.perf_counter()
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        self.wall = time.perf_counter() - self.start

    def result(self):
        return dict(wall_seconds=self.wall, sampled_cpu_seconds=self.cpu, peak_sampled_rss_bytes=self.peak,
                    method='0.25s sampled sum of client, Ollama server and its descendants; CPU deltas; shared server may include unrelated work',
                    server_pid=self.server_pid)
