"""Cancel GA work on shutdown while preserving the server's signal handlers."""
from contextlib import contextmanager
import signal
from threading import current_thread, main_thread

from src.backend.services.portfolio_progress import cancel_active_portfolio_analyses


@contextmanager
def cancel_analyses_on_shutdown():
    installed = {}
    if current_thread() is main_thread():
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for number in signals:
            previous = signal.getsignal(number)
            if not callable(previous):
                continue

            def handle_shutdown(signum, frame, forward=previous):
                cancel_active_portfolio_analyses()
                forward(signum, frame)

            signal.signal(number, handle_shutdown)
            installed[number] = (previous, handle_shutdown)
    try:
        yield
    finally:
        cancel_active_portfolio_analyses()
        for number, (previous, wrapper) in installed.items():
            if signal.getsignal(number) is wrapper:
                signal.signal(number, previous)
