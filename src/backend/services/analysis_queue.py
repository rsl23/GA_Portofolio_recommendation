"""Bounded FIFO admission shared by JSON and SSE analysis routes in one server."""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, Event

from fastapi import HTTPException

from src.backend.core.config import settings
from src.gaengine.progress import AnalysisCancelled, progress_listener


@dataclass(eq=False)
class AnalysisTicket:
    user_id: str
    stop: Event = field(default_factory=Event)
    running: bool = False


class AnalysisQueue:
    def __init__(self, capacity=1, max_queued=20):
        self.capacity = capacity
        self.max_queued = max_queued
        self._condition = Condition()
        self._waiting = deque()
        self._users = {}
        self._running = 0

    def reserve(self, user_id):
        with self._condition:
            if user_id in self._users:
                raise HTTPException(409, "Akun Anda masih memiliki analisis aktif atau dalam antrean. "
                                    "Tunggu hingga selesai atau hentikan analisis sebelumnya.")
            if len(self._users) >= self.capacity + self.max_queued:
                raise HTTPException(429, "Antrean analisis penuh. Silakan coba beberapa saat lagi.",
                                    headers={"Retry-After": "10"})
            ticket = AnalysisTicket(user_id)
            self._users[user_id] = ticket
            self._waiting.append(ticket)
            return ticket

    def try_start(self, ticket):
        with self._condition:
            if ticket.stop.is_set() or self._users.get(ticket.user_id) is not ticket:
                raise AnalysisCancelled("Analisis dalam antrean dibatalkan.")
            if self._waiting[0] is ticket and self._running < self.capacity:
                self._waiting.popleft()
                ticket.running = True
                self._running += 1
                return True
            return False

    def position(self, ticket):
        with self._condition:
            if ticket.stop.is_set():
                raise AnalysisCancelled("Analisis dalam antrean dibatalkan.")
            return self._waiting.index(ticket) + 1

    def finish(self, ticket):
        with self._condition:
            if self._users.get(ticket.user_id) is not ticket:
                return
            del self._users[ticket.user_id]
            if ticket.running:
                self._running -= 1
            else:
                self._waiting.remove(ticket)
            self._condition.notify_all()

    def cancel(self, ticket):
        with self._condition:
            ticket.stop.set()
            # A disconnected running worker still owns its slot and account
            # until its transaction has rolled back and the worker has exited.
            if not ticket.running:
                self.finish(ticket)
            self._condition.notify_all()

    def cancel_all(self):
        with self._condition:
            for ticket in list(self._users.values()):
                self.cancel(ticket)

    def wait(self, ticket):
        with self._condition:
            while not self.try_start(ticket):
                self._condition.wait(timeout=.25)


analysis_queue = AnalysisQueue(settings.ANALYSIS_MAX_CONCURRENT, settings.ANALYSIS_MAX_QUEUED)


@contextmanager
def analysis_slot(user_id):
    """Non-streaming requests share the same queue and per-account reservation."""
    ticket = analysis_queue.reserve(user_id)
    try:
        analysis_queue.wait(ticket)
        with progress_listener(None, cancel_event=ticket.stop):
            yield
    except AnalysisCancelled as exc:
        raise HTTPException(503, "Analisis dihentikan karena server sedang ditutup atau dimulai ulang.") from exc
    finally:
        analysis_queue.finish(ticket)
