"""Worker-lifecycle helpers for dialogs that run background QThread workers.

A dialog destroyed while its worker still runs receives progress/finished
signals on a deleted C++ object — RuntimeError at best, segfault at worst.
Every worker-owning dialog calls :func:`stop_worker` from ``reject()`` and
``closeEvent()`` so the thread is always stopped (or safely detached) before
the dialog goes away.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QDialog

log = logging.getLogger(__name__)

# Workers that refused to stop within the timeout are detached here so the
# Python wrapper (and the running C++ thread) stay alive until finished —
# destroying a running QThread aborts the process.
_DETACHED: set[QThread] = set()


def stop_worker(
    dialog: QDialog,
    worker: QThread | None = None,
    timeout_ms: int = 5000,
) -> None:
    """Stop a dialog's worker before the dialog is destroyed.

    Defaults to the dialog's ``_worker`` attribute; dialogs with several
    workers pass each one explicitly. Asks the worker to cancel (``cancel()``
    when implemented, else ``requestInterruption()``), waits up to
    ``timeout_ms``, and as a last resort detaches it: signals are
    disconnected so nothing reaches the closed dialog, and the thread is
    kept alive until it finishes on its own.
    """
    if worker is None:
        worker = getattr(dialog, "_worker", None)
        dialog._worker = None
    if worker is None:
        return
    if not worker.isRunning():
        return

    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception:
            log.debug("worker.cancel() raised during dialog close", exc_info=True)
    else:
        worker.requestInterruption()

    if worker.wait(timeout_ms):
        return

    # Still running — never terminate a Python thread; detach instead.
    log.warning(
        "Dialog worker %s did not stop within %d ms; detaching",
        type(worker).__name__, timeout_ms,
    )
    try:
        worker.disconnect()
    except TypeError:
        pass
    worker.finished.connect(lambda _=None, w=worker: _DETACHED.discard(w))
    _DETACHED.add(worker)
