"""Bound waiting for synchronous, read-only network calls.

The caller remains cancellable even if a library blocks in socket reads. Only
pure network work belongs here: never pass database writes or commits. Bounded
daemon workers keep an abandoned network call from stalling process shutdown.
"""
from contextvars import copy_context
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
import time

from src.gaengine.progress import check_cancelled

_slots = BoundedSemaphore(4)


def cancellable_io(operation, *, timeout):
    deadline = time.monotonic() + max(0., timeout)
    while True:
        check_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batas waktu layanan narasi habis.")
        if _slots.acquire(timeout=min(.05, remaining)):
            break
    result = Queue(maxsize=1)
    context = copy_context()

    def run():
        try:
            result.put((True, context.run(operation)))
        except BaseException as exc:
            result.put((False, exc))
        finally:
            _slots.release()

    try:
        Thread(target=run, name="analysis-network", daemon=True).start()
    except BaseException:
        _slots.release()
        raise
    while True:
        check_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batas waktu layanan narasi habis.")
        try:
            success, value = result.get(timeout=min(.05, remaining))
        except Empty:
            continue
        check_cancelled()
        if not success:
            raise value
        return value
