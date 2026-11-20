"""Request-local progress reporting; optimization also works without a listener."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import math
import time

_listener = ContextVar("ga_progress_listener", default=None)
_context = ContextVar("ga_progress_context", default=None)
_cancel_event = ContextVar("ga_progress_cancel_event", default=None)
_evolution_clock = ContextVar("ga_evolution_clock", default=None)
logger = logging.getLogger(__name__)


class AnalysisCancelled(Exception):
    """The requesting connection closed; stop before further work/commits."""


def check_cancelled():
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise AnalysisCancelled("Koneksi analisis ditutup.")


def cancellable_sleep(seconds):
    """Allow shutdown/disconnect to interrupt retry backoff immediately."""
    check_cancelled()
    event = _cancel_event.get()
    if event is None:
        time.sleep(seconds)
    else:
        event.wait(max(0., seconds))
    check_cancelled()


@contextmanager
def progress_listener(listener, *, cancel_event=None):
    listener_token = _listener.set(listener)
    context_token = _context.set({})
    cancel_token = _cancel_event.set(cancel_event)
    clock_token = _evolution_clock.set(None)
    try:
        yield
    finally:
        _evolution_clock.reset(clock_token)
        _cancel_event.reset(cancel_token)
        _context.reset(context_token)
        _listener.reset(listener_token)


def set_progress_context(**values):
    _context.set({**(_context.get() or {}), **values})


def report_progress(stage, title, detail="", **values):
    check_cancelled()
    listener = _listener.get()
    if listener is not None:
        try:
            listener({**(_context.get() or {}), "stage": stage,
                      "title": title, "detail": detail, **values})
        except AnalysisCancelled:
            raise
        except Exception:
            # A display failure must never affect optimization or persistence.
            logger.exception("Progress listener failed")


def report_generation(generation, total, fitness):
    check_cancelled()
    score = float(fitness) if fitness is not None else None
    now = time.perf_counter()
    period = (_context.get() or {}).get("series_index")
    clock = _evolution_clock.get()
    if clock is None or clock[0] != period or generation <= clock[3]:
        clock = (period, now, generation, generation)
    elapsed = now - clock[1]
    rate = (generation - clock[2]) / elapsed if elapsed > 0 else None
    _evolution_clock.set((period, clock[1], clock[2], generation))
    if generation == 1 or generation % 25 == 0 or generation == total:
        logger.info("GA generasi %d/%d | fitness=%s | waktu_evolusi=%.3fs | jadwal=%s",
                    generation, total, score, elapsed,
                    (_context.get() or {}).get("series_date", "tunggal"))
    report_progress(
        "optimizing", "Mencari komposisi yang paling sesuai",
        "Berbagai kombinasi saham dibandingkan sesuai modal dan profil risiko Anda.",
        generation=generation, total_generations=total,
        fitness=score if score is not None and math.isfinite(score) else None,
        engine_elapsed_seconds=elapsed, generations_per_second=rate,
    )


def report_candidates(completed, total, *, evaluating=False):
    check_cancelled()
    if completed not in (0, total) and completed % 10:
        return
    title = "Menilai kombinasi awal" if evaluating else "Menyiapkan kombinasi awal"
    if completed in (0, total):
        logger.info("GA %s: %d/%d", title.lower(), completed, total)
    report_progress("optimizing", title,
                    f"{completed} dari {total} kombinasi awal "
                    + ("sudah diperiksa risikonya. Generasi pertama dimulai setelah penilaian ini selesai."
                       if evaluating else "sudah disiapkan untuk pencarian komposisi saham."),
                    population_completed=completed, population_total=total)
