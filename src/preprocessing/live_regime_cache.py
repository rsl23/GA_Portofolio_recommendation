"""Short-lived, coherent live assessments shared across reloads and workers."""
from contextlib import contextmanager
from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4

CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'cache' / 'live_regime'
TTL_SECONDS = 15 * 60
UNKNOWN_TTL_SECONDS = 30
_thread_lock = Lock()
_memory = OrderedDict()
logger = logging.getLogger(__name__)


@contextmanager
def _process_lock(path):
    """OS locks are released on crashes; a second worker reads the first result."""
    with path.open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + 600
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Penilaian pasar live masih diperbarui.')
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fresh(entry, as_of, parameters):
    try:
        assessment = entry['assessment']
        ttl = UNKNOWN_TTL_SECONDS if assessment['status'] == 'unknown' else TTL_SECONDS
        return (assessment['requested_date'] == as_of.isoformat()
                and assessment['config_version'] == parameters['version']
                and 0 <= time.time() - entry['created_at'] < ttl)
    except (KeyError, TypeError, ValueError):
        return False


def _remember(key, entry):
    _memory[key] = entry
    _memory.move_to_end(key)
    while len(_memory) > 64:
        _memory.popitem(last=False)
    return deepcopy(entry['assessment'])


def live_assessment(as_of, parameters, factory):
    """Freeze the entire observation set, never just the changing breadth part.

    Date and *all* detector parameters form the key. Expiry recomputes the
    assessment, so existing fingerprint validation still rejects stale choices.
    Unknown results have a short lifetime to permit recovery from outages.
    """
    payload = json.dumps([as_of.isoformat(), parameters], sort_keys=True, allow_nan=False)
    key = hashlib.sha256(payload.encode()).hexdigest()
    path = CACHE_DIR / f'{as_of.isoformat()}_{key}.json'
    with _thread_lock:
        # The persistent file is authoritative across workers. Memory is only
        # a fallback when the directory cannot be accessed or written.
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with _process_lock(path.with_suffix('.lock')):
                try:
                    entry = json.loads(path.read_text(encoding='utf-8'))
                    if _fresh(entry, as_of, parameters):
                        return _remember(key, entry)
                except (OSError, ValueError, TypeError):
                    pass
                if _fresh(_memory.get(key), as_of, parameters):
                    return deepcopy(_memory[key]['assessment'])
                entry = {'created_at': time.time(), 'assessment': factory()}
                # Start the validity window after potentially slow downloads.
                entry['created_at'] = time.time()
                result = _remember(key, entry)
                temporary = path.with_suffix(f'.{uuid4().hex}.tmp')
                try:
                    temporary.write_text(json.dumps(entry, allow_nan=False), encoding='utf-8')
                    os.replace(temporary, path)
                except (OSError, ValueError):
                    logger.warning('Penilaian live memakai cache memori; cache berkas tidak dapat disimpan.')
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                return result
        except OSError:
            logger.warning('Cache penilaian live tidak dapat diakses; memakai cache memori.')
            if _fresh(_memory.get(key), as_of, parameters):
                return deepcopy(_memory[key]['assessment'])
            assessment = factory()
            return _remember(key, {'created_at': time.time(), 'assessment': assessment})
