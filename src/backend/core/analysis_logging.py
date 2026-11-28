"""Keep a paused/occluded development terminal off the computation path."""
from contextlib import contextmanager
from logging.handlers import QueueHandler
from queue import Empty, Full, Queue
from threading import Event, Thread
import logging


class NonBlockingLogHandler(QueueHandler):
    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except Full:
            # QueueHandler's default handleError writes to stderr. Never fall
            # back to the same potentially stalled terminal from the GA thread.
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except Full:
                pass


def _destinations(logger):
    handlers = []
    current = logger
    while current is not None:
        handlers.extend(current.handlers)
        if not current.propagate:
            break
        current = current.parent
    return tuple(handlers)


@contextmanager
def nonblocking_analysis_logging():
    """Retain existing console formatting and route only analysis logs."""
    names = ("src.gaengine.progress", "src.gaengine.data_loader",
             "src.backend.controller.portfolio_controller",
             "src.backend.services.portfolio_progress",
             "src.postprocessing.openai_llm_caller", "src.postprocessing.portfolio_news")
    bridges = []
    for name in names:
        logger = logging.getLogger(name)
        destinations = _destinations(logger)
        queue = Queue(maxsize=512)
        stopped = Event()

        def deliver(output_queue=queue, stop=stopped, targets=destinations):
            while not stop.is_set():
                try:
                    record = output_queue.get(timeout=.1)
                except Empty:
                    continue
                for target in targets:
                    if record.levelno >= target.level:
                        try:
                            target.handle(record)
                        except Exception:
                            # Console failures must not end or block analysis.
                            pass

        previous = (list(logger.handlers), logger.propagate)
        handler = NonBlockingLogHandler(queue)
        logger.handlers = [handler]
        logger.propagate = False
        thread = Thread(target=deliver, name="analysis-console", daemon=True)
        thread.start()
        bridges.append((logger, previous, handler, stopped, thread))
    try:
        yield
    finally:
        for logger, previous, handler, stopped, thread in bridges:
            logger.handlers, logger.propagate = previous
            stopped.set()
            handler.close()
        for _, _, _, _, thread in bridges:
            # Closing a terminal cannot stall server shutdown either.
            thread.join(timeout=.1)
