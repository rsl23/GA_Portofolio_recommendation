"""Declare latency-sensitive CPU work to Windows without changing priority.

Windows can infer a lower QoS from the visibility of a parent editor window.
Use its documented thread-scoped HighQoS opt-out only during evolution, then
restore the caller's policy. Unsupported platforms keep their usual behavior.
"""
from contextlib import contextmanager
import ctypes
import sys


class _PowerThrottling(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32),
                ("StateMask", ctypes.c_uint32)]


_THREAD_POWER_THROTTLING = 3
_EXECUTION_SPEED = 1


def _load_windows_api():
    if sys.platform != "win32":
        return None
    try:
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.GetCurrentThread.restype = ctypes.c_void_p
        for name in ("GetThreadInformation", "SetThreadInformation"):
            function = getattr(api, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            function.restype = ctypes.c_int
        return api
    except (AttributeError, OSError):
        return None


@contextmanager
def evolution_performance():
    """Temporarily disable execution-speed throttling on this thread only."""
    api = _load_windows_api()
    original = _PowerThrottling(1, 0, 0)
    active = False
    thread = None
    if api is not None:
        try:
            thread = api.GetCurrentThread()
            if api.GetThreadInformation(thread, _THREAD_POWER_THROTTLING,
                                        ctypes.byref(original), ctypes.sizeof(original)):
                desired = _PowerThrottling(original.Version,
                    original.ControlMask | _EXECUTION_SPEED, original.StateMask & ~_EXECUTION_SPEED)
                active = bool(api.SetThreadInformation(thread, _THREAD_POWER_THROTTLING,
                    ctypes.byref(desired), ctypes.sizeof(desired)))
        except OSError:
            pass
    try:
        yield active
    finally:
        if active:
            try:
                api.SetThreadInformation(thread, _THREAD_POWER_THROTTLING,
                    ctypes.byref(original), ctypes.sizeof(original))
            except OSError:
                pass
