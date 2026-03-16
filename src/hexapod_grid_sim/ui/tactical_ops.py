"""Tactical Operations Center - sci-fi system metrics dashboard using Tkinter."""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
import tkinter as tk
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil  # noqa: N811

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#080C14"
CYAN = "#00E6FF"
GREEN = "#00FF88"
AMBER = "#FFA000"
RED = "#FF3232"
DIM_CYAN = "#004455"
TEXT_DIM = "#3A5A6A"


# ---------------------------------------------------------------------------
# System metrics helpers
# ---------------------------------------------------------------------------
class SystemMetrics:
    """Collect CPU / GPU / memory / temperature readings."""

    __slots__ = ("cpu", "gpu", "mem_used", "mem_total", "temp")

    def __init__(self) -> None:
        self.cpu: float = 0.0
        self.gpu: float = 0.0
        self.mem_used: float = 0.0
        self.mem_total: float = 0.0
        self.temp: Optional[float] = None

    def refresh(self) -> None:
        if HAS_PSUTIL:
            self.cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            self.mem_used = mem.used / (1024 ** 3)
            self.mem_total = mem.total / (1024 ** 3)
        else:
            self.cpu = 0.0
            self.mem_used = 0.0
            self.mem_total = 1.0

        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu = gpus[0].load * 100 if gpus else 0.0
            except Exception:
                self.gpu = 0.0
        else:
            self.gpu = 0.0

        self.temp = self._read_temperature()

    @staticmethod
    def _read_temperature() -> Optional[float]:
        if HAS_PSUTIL:
            try:
                temps = psutil.sensors_temperatures()
                for entries in temps.values():
                    for entry in entries:
                        if entry.current > 0:
                            return entry.current
            except (AttributeError, Exception):
                pass
        return None


# ---------------------------------------------------------------------------
# Socket listener for simulation state
# ---------------------------------------------------------------------------
class StateReceiver:
    """Listens for grid-state JSON packets on localhost UDP."""

    DEFAULT_PORT = 9877

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self.latest_state: dict = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_state(self) -> dict:
        with self._lock:
            return dict(self.latest_state)

    def _listen(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
            sock.settimeout(0.5)
        except OSError:
            return
        while self._running:
            try:
                data, _ = sock.recvfrom(65535)
                payload = json.loads(data.decode("utf-8"))
                with self._lock:
                    self.latest_state = payload
            except socket.timeout:
                continue
            except Exception:
                continue
        sock.close()


# ---------------------------------------------------------------------------
# Tactical Ops Center window
# ---------------------------------------------------------------------------
class TacticalOpsCenter:
    """Tkinter-based sci-fi dashboard for hexapod simulation monitoring."""

    WIDTH = 820
    HEIGHT = 520

    def __init__(self, *, connect_sim: bool = True, port: int = StateReceiver.DEFAULT_PORT) -> None:
        self.metrics = SystemMetrics()
        self.receiver = StateReceiver(port) if connect_sim else None
        self._tick = 0
        self._scan_x = 0.0

        self.root = tk.Tk()
        self.root.title("TACTICAL OPS CENTER")
        self.root.configure(bg=BG)
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root, width=self.WIDTH, height=self.HEIGHT,
            bg=BG, highlightthickness=0,
        )
        self.canvas.pack()

    # -- public API ---------------------------------------------------------
    def run(self) -> None:
        if self.receiver:
            self.receiver.start()
        self._draw_frame()
        self.root.mainloop()
        if self.receiver:
            self.receiver.stop()

    # -- rendering ----------------------------------------------------------
    def _draw_frame(self) -> None:
        self._tick += 1
        self.metrics.refresh()
        c = self.canvas
        c.delete("dyn")  # clear dynamic layer

        self._draw_border(c)
        self._draw_header(c)
        self._draw_scan_line(c)
        self._draw_metrics_panel(c)
        self._draw_sim_state(c)
        self._draw_status_indicators(c)
        self._draw_coordinates(c)

        self.root.after(60, self._draw_frame)

    def _draw_border(self, c: tk.Canvas) -> None:
        pad = 6
        c.create_rectangle(
            pad, pad, self.WIDTH - pad, self.HEIGHT - pad,
            outline=CYAN, width=1, tags="dyn",
        )
        c.create_rectangle(
            pad + 2, pad + 2, self.WIDTH - pad - 2, self.HEIGHT - pad - 2,
            outline=DIM_CYAN, width=1, tags="dyn",
        )

    def _draw_header(self, c: tk.Canvas) -> None:
        blink = self._tick % 30 < 20
        colour = CYAN if blink else DIM_CYAN
        c.create_text(
            self.WIDTH // 2, 24, text="\u2550 TACTICAL OPS CENTER \u2550",
            fill=colour, font=("Consolas", 14, "bold"), tags="dyn",
        )
        c.create_line(10, 42, self.WIDTH - 10, 42, fill=DIM_CYAN, tags="dyn")

    def _draw_scan_line(self, c: tk.Canvas) -> None:
        self._scan_x = (self._scan_x + 2.5) % self.WIDTH
        x = self._scan_x
        c.create_line(x, 44, x, self.HEIGHT - 10, fill=DIM_CYAN, width=1, tags="dyn")
        # glow region
        for offset in range(1, 8):
            alpha_hex = format(max(0, 0x44 - offset * 8), "02x")
            glow = f"#00{alpha_hex}{alpha_hex}"
            c.create_line(x - offset, 44, x - offset, self.HEIGHT - 10, fill=glow, tags="dyn")

    def _draw_metrics_panel(self, c: tk.Canvas) -> None:
        x0, y0 = 30, 60
        m = self.metrics

        lines = [
            ("CPU", f"{m.cpu:5.1f}%", self._bar_colour(m.cpu)),
            ("GPU", f"{m.gpu:5.1f}%", self._bar_colour(m.gpu)),
            ("MEM", f"{m.mem_used:.1f}/{m.mem_total:.1f} GB", CYAN),
        ]
        if m.temp is not None:
            lines.append(("TMP", f"{m.temp:.0f}\u00b0C", self._bar_colour(m.temp, hi=90)))

        for i, (label, value, colour) in enumerate(lines):
            y = y0 + i * 36
            c.create_text(x0, y, text=f"\u251c {label}", anchor="w",
                          fill=TEXT_DIM, font=("Consolas", 10), tags="dyn")
            c.create_text(x0 + 50, y, text=value, anchor="w",
                          fill=colour, font=("Consolas", 11, "bold"), tags="dyn")
            # mini bar
            pct = self._extract_pct(label, m)
            bw = 120
            c.create_rectangle(x0 + 180, y - 6, x0 + 180 + bw, y + 6,
                               outline=DIM_CYAN, tags="dyn")
            fill_w = int(bw * pct / 100)
            if fill_w > 0:
                c.create_rectangle(x0 + 180, y - 6, x0 + 180 + fill_w, y + 6,
                                   fill=colour, outline="", tags="dyn")

    def _draw_sim_state(self, c: tk.Canvas) -> None:
        x0, y0 = 430, 60
        c.create_text(x0, y0, text="\u2502 SIMULATION STATE", anchor="w",
                      fill=AMBER, font=("Consolas", 10, "bold"), tags="dyn")
        state = self.receiver.get_state() if self.receiver else {}
        if not state:
            c.create_text(x0 + 10, y0 + 22, text="NO LINK", anchor="w",
                          fill=RED if self._tick % 20 < 10 else BG,
                          font=("Consolas", 10), tags="dyn")
            return
        row = 0
        for key in ("grid_cols", "grid_rows", "active_hexapods", "time"):
            val = state.get(key, "---")
            c.create_text(x0 + 10, y0 + 22 + row * 20, text=f"{key}: {val}",
                          anchor="w", fill=CYAN, font=("Consolas", 9), tags="dyn")
            row += 1

    def _draw_status_indicators(self, c: tk.Canvas) -> None:
        x0 = 30
        y0 = self.HEIGHT - 50
        indicators = [
            ("SYS", True), ("NET", self.receiver is not None),
            ("GPU", HAS_GPUTIL), ("PSU", HAS_PSUTIL),
        ]
        for i, (label, ok) in enumerate(indicators):
            x = x0 + i * 90
            blink_on = self._tick % 40 < 30
            fill = (GREEN if ok else RED) if blink_on else BG
            c.create_oval(x, y0, x + 10, y0 + 10, fill=fill, outline=DIM_CYAN, tags="dyn")
            c.create_text(x + 16, y0 + 5, text=label, anchor="w",
                          fill=TEXT_DIM, font=("Consolas", 9), tags="dyn")

    def _draw_coordinates(self, c: tk.Canvas) -> None:
        t = time.time()
        x_val = math.sin(t * 0.7) * 45.0
        y_val = math.cos(t * 0.5) * 30.0
        z_val = math.sin(t * 0.3) * 10.0
        coord_text = f"X:{x_val:+07.2f}  Y:{y_val:+07.2f}  Z:{z_val:+07.2f}"
        c.create_text(
            self.WIDTH // 2, self.HEIGHT - 24, text=coord_text,
            fill=GREEN, font=("Consolas", 10), tags="dyn",
        )

    # -- util ---------------------------------------------------------------
    @staticmethod
    def _bar_colour(value: float, hi: float = 80.0) -> str:
        if value > hi:
            return RED
        if value > hi * 0.6:
            return AMBER
        return CYAN

    @staticmethod
    def _extract_pct(label: str, m: SystemMetrics) -> float:
        if label == "CPU":
            return m.cpu
        if label == "GPU":
            return m.gpu
        if label == "MEM":
            return (m.mem_used / m.mem_total * 100) if m.mem_total > 0 else 0
        if label == "TMP":
            return min((m.temp or 0) / 100 * 100, 100)
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def start_tactical_ops(*, connect_sim: bool = True, port: int = StateReceiver.DEFAULT_PORT) -> None:
    """Launch the Tactical Ops Center window."""
    toc = TacticalOpsCenter(connect_sim=connect_sim, port=port)
    toc.run()


if __name__ == "__main__":
    start_tactical_ops()
