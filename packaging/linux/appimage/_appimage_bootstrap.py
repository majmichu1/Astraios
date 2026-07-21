"""First-run PyTorch provisioning for the Astraios AppImage.

The AppImage deliberately ships without PyTorch (a CUDA build alone is
~1.8 GB, over GitHub's release-asset limit and wasteful for CPU-only
machines). This script detects the GPU once and installs the matching
PyTorch into a writable per-user runtime directory, which ``AppRun`` then
puts on ``PYTHONPATH``.

Shows a Qt progress window when a display is available (the usual
double-click-from-the-file-manager case) and falls back to plain terminal
output otherwise, so it also works over SSH or in a container.

Exit code 0 means torch is importable afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def runtime_dir() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(data_home) / "Astraios" / "runtime"


def has_nvidia_gpu() -> bool:
    """Same detection the shell installer uses: nvidia-smi, then lspci."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            if subprocess.run([smi], capture_output=True, timeout=15).returncode == 0:
                return True
        except Exception:
            pass
    lspci = shutil.which("lspci")
    if lspci:
        try:
            out = subprocess.run([lspci], capture_output=True, text=True, timeout=15)
            if "nvidia" in out.stdout.lower():
                return True
        except Exception:
            pass
    return False


def pip_command(target: Path, index_url: str) -> list[str]:
    return [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir",
        "--prefix", str(target),
        "--index-url", index_url,
        "torch", "torchvision",
    ]


def run_install(target: Path, index_url: str, on_line=None) -> int:
    target.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        pip_command(target, index_url),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if on_line:
            on_line(line)
        else:
            print(line, flush=True)
    return proc.wait()


def torch_importable(target: Path) -> bool:
    """Verify in a subprocess with the new path, so this process stays clean."""
    env = dict(os.environ)
    sites = list(target.glob("lib/python*/site-packages"))
    if not sites:
        return False
    env["PYTHONPATH"] = os.pathsep.join(
        [str(s) for s in sites] + [env.get("PYTHONPATH", "")]
    )
    check = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, env=env,
    )
    if check.returncode == 0:
        print(f"Astraios: PyTorch {check.stdout.strip()} ready.", flush=True)
        return True
    print(check.stderr.strip(), file=sys.stderr, flush=True)
    return False


def run_headless(target: Path, index_url: str, gpu: bool) -> int:
    kind = "CUDA (NVIDIA GPU)" if gpu else "CPU"
    print(f"Astraios: installing the {kind} compute runtime. "
          "This happens once and needs an internet connection.", flush=True)
    rc = run_install(target, index_url)
    if rc != 0:
        print("Astraios: the download failed. Check your internet connection "
              "and run the AppImage again.", file=sys.stderr)
        return rc
    return 0 if torch_importable(target) else 1


def run_gui(target: Path, index_url: str, gpu: bool) -> int:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QLabel,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
    )

    class Worker(threading.Thread):
        def __init__(self, dialog):
            super().__init__(daemon=True)
            self.dialog = dialog
            self.rc = 1

        def run(self):
            rc = run_install(target, index_url, on_line=self.dialog.line.emit)
            if rc == 0 and torch_importable(target):
                self.rc = 0
            self.dialog.finished_ok.emit(self.rc == 0)

    class Dialog(QDialog):
        line = pyqtSignal(str)
        finished_ok = pyqtSignal(bool)

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Astraios — first-time setup")
            self.setMinimumWidth(520)
            lay = QVBoxLayout(self)
            kind = "CUDA (NVIDIA GPU)" if gpu else "CPU"
            head = QLabel(
                f"Setting up the {kind} compute runtime.\n\n"
                "This runs once, needs an internet connection, and downloads "
                "roughly 1-2 GB. Astraios starts automatically when it is done."
            )
            head.setWordWrap(True)
            lay.addWidget(head)
            self.bar = QProgressBar()
            self.bar.setRange(0, 0)  # indeterminate: pip gives no total up front
            lay.addWidget(self.bar)
            self.status = QLabel("Starting…")
            self.status.setWordWrap(True)
            self.status.setTextFormat(Qt.TextFormat.PlainText)
            lay.addWidget(self.status)
            self.close_btn = QPushButton("Close")
            self.close_btn.setVisible(False)
            self.close_btn.clicked.connect(self.accept)
            lay.addWidget(self.close_btn)
            self.line.connect(self._on_line)
            self.finished_ok.connect(self._on_done)
            self.ok = False

        def _on_line(self, text: str):
            # pip's progress lines are long; show the tail that matters.
            self.status.setText(text[-160:])

        def _on_done(self, ok: bool):
            self.ok = ok
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            if ok:
                self.accept()
            else:
                self.status.setText(
                    "Setup failed. Check your internet connection and start "
                    "Astraios again."
                )
                self.close_btn.setVisible(True)

    # The QApplication must outlive the dialog, so keep the reference bound
    # for the whole function rather than letting it be collected.
    qt_app = QApplication(sys.argv[:1])
    dlg = Dialog()
    worker = Worker(dlg)
    worker.start()
    dlg.exec()
    qt_app.quit()
    return 0 if dlg.ok else 1


def main() -> int:
    target = runtime_dir()
    gpu = has_nvidia_gpu()
    index_url = CUDA_INDEX if gpu else CPU_INDEX

    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if headless:
        return run_headless(target, index_url, gpu)
    try:
        return run_gui(target, index_url, gpu)
    except Exception as exc:  # no usable Qt platform, broken display, etc.
        print(f"Astraios: falling back to terminal setup ({exc})", flush=True)
        return run_headless(target, index_url, gpu)


if __name__ == "__main__":
    sys.exit(main())
