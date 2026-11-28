"""POST/SSE transport for the existing, authenticated portfolio operations."""
import asyncio
import json
import logging
import time
from threading import Event, Thread

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse

from src.gaengine.progress import AnalysisCancelled, progress_listener, report_progress
from src.backend.services.analysis_queue import analysis_queue

logger = logging.getLogger(__name__)


def cancel_active_portfolio_analyses():
    """Called before server shutdown starts waiting for active responses."""
    analysis_queue.cancel_all()


def portfolio_progress_response(operation, *, user_id):
    # Reserve before response headers so duplicate/full requests receive a
    # regular HTTP 409/429, including requests using the explicit SSE URLs.
    ticket = analysis_queue.reserve(user_id)

    async def events():
        messages = asyncio.Queue(maxsize=256)
        disconnected = Event()
        stop_requested = ticket.stop
        loop = asyncio.get_running_loop()
        last_phase = None
        last_progress_at = 0.
        pending_progress = None

        def enqueue(kind, payload):
            if disconnected.is_set():
                return
            if messages.full():
                messages.get_nowait()
            messages.put_nowait((kind, payload))

        def send(kind, payload):
            if disconnected.is_set():
                return
            # Serialize on the producer so terminal serialization failures are
            # handled by run() and reach the page as an error event.
            payload = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            # Wake the asynchronous response without tying up a thread-pool
            # slot. Cancellation can then interrupt an idle stream promptly.
            try:
                loop.call_soon_threadsafe(enqueue, kind, payload)
            except RuntimeError:
                disconnected.set()
                stop_requested.set()

        def publish(kind, payload):
            nonlocal last_phase, last_progress_at, pending_progress
            if kind == "progress":
                phase = (payload.get("stage"), payload.get("title"), payload.get("series_index"))
                now = time.monotonic()
                if phase == last_phase and now - last_progress_at < 1.:
                    pending_progress = payload
                    return
                if phase != last_phase and pending_progress is not None:
                    send("progress", pending_progress)
                pending_progress = None
                last_phase, last_progress_at = phase, now
            elif pending_progress is not None:
                # Always deliver the last generation before a terminal result.
                send("progress", pending_progress)
                pending_progress = None
            send(kind, payload)

        def run():
            with progress_listener(lambda payload: publish("progress", payload),
                                   cancel_event=stop_requested):
                try:
                    report_progress("preparing", "Menyiapkan analisis Anda",
                                    "Mencocokkan pengaturan dan menyiapkan data yang diperlukan.")
                    result = jsonable_encoder(operation())
                    publish("result", result)
                except AnalysisCancelled:
                    logger.info("Analisis portofolio dihentikan setelah permintaan pembatalan.")
                    publish("error", {"status": 503, "detail":
                        "Analisis dihentikan karena server sedang ditutup atau dimulai ulang. Jadwal rebalance yang sebelumnya sudah selesai tetap tersedia di riwayat."})
                except HTTPException as exc:
                    publish("error", {"status": exc.status_code, "detail": exc.detail})
                except Exception:
                    logger.exception("Portfolio streaming operation failed")
                    publish("error", {"status": 500, "detail":
                        "Analisis belum berhasil diselesaikan. Silakan periksa portofolio Anda sebelum mencoba kembali."})
                finally:
                    analysis_queue.finish(ticket)

        try:
            previous_position = None
            last_queue_update = 0.
            while not analysis_queue.try_start(ticket):
                position = analysis_queue.position(ticket)
                now = time.monotonic()
                if position != previous_position or now - last_queue_update >= 10:
                    payload = json.dumps({"stage": "queued", "title": "Menunggu antrean analisis",
                        "detail": f"Posisi antrean Anda: {position}. Analisis dimulai otomatis saat giliran tersedia.",
                        "queue_position": position}, ensure_ascii=False)
                    yield f"event: progress\ndata: {payload}\n\n"
                    previous_position, last_queue_update = position, now
                await asyncio.sleep(.25)
            try:
                Thread(target=run, name="portfolio-analysis", daemon=True).start()
            except BaseException:
                analysis_queue.finish(ticket)
                raise
            while True:
                try:
                    kind, payload = await asyncio.wait_for(messages.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {kind}\ndata: {payload}\n\n"
                if kind in {"result", "error"}:
                    break
        except AnalysisCancelled:
            yield 'event: error\ndata: {"status":503,"detail":"Analisis dalam antrean dihentikan karena server sedang ditutup atau dimulai ulang."}\n\n'
        finally:
            # Stop at the next candidate/checkpoint. The worker's own session
            # rolls back its uncommitted work; earlier series commits remain.
            disconnected.set()
            stop_requested.set()
            analysis_queue.cancel(ticket)

    class AnalysisResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # Also covers a disconnect before the iterator is first read.
                analysis_queue.cancel(ticket)

    return AnalysisResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
    })
