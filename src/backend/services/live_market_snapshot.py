"""Refresh the live GA snapshot when a newer completed session may exist."""
from datetime import datetime, timedelta, timezone
from threading import RLock
from types import SimpleNamespace
import time

from src.gaengine.data_loader import build_market_data
from src.gaengine.progress import report_progress

_lock = RLock()
RECHECK_SECONDS = 300


def get_live_market_snapshot(state, *, profile_policy="legacy"):
    if profile_policy not in {"legacy", "composition_v1", "composition_v2"}:
        raise ValueError("Kebijakan profil tidak dikenali.")
    now = datetime.now(timezone(timedelta(hours=7)))
    cutoff = now.date() if now.hour >= 17 else now.date() - timedelta(days=1)
    with _lock:
        # Keep profile evidence separate: v2 uses a live benchmark whereas v1
        # uses the historical benchmark. Neither may mutate startup's snapshot.
        slot = state
        if profile_policy != "legacy":
            slots = getattr(state, 'profile_market_snapshots', None)
            if slots is None:
                slots = {}
                state.profile_market_snapshots = slots
            slot = slots.setdefault(profile_policy, SimpleNamespace())
        cached = getattr(slot, 'market_data_today', None)
        cached_day = getattr(cached, 'price_date', None)
        checked = getattr(slot, 'live_snapshot_checked_through', None)
        checked_at = getattr(slot, 'live_snapshot_checked_at', 0.)
        has_features = profile_policy == "legacy" or getattr(cached, 'profile_features', None) is not None
        if has_features and cached is not None and cached.n_stocks > 0 and cached_day is not None and cached_day <= cutoff:
            if cached_day == cutoff or (checked == cutoff and time.monotonic() - checked_at < RECHECK_SECONDS):
                report_progress("market", "Data pasar siap digunakan",
                                "Harga, fundamental, dan suku bunga acuan tersedia dari data yang sudah disiapkan.")
                return cached
        options = {}
        if profile_policy != "legacy":
            options = {"include_profile_features": True,
                       "profile_benchmark_live": profile_policy == "composition_v2"}
        # Assemble the requested profile in the first download. Returning a
        # legacy-only snapshot forces the controller to download all stocks again.
        fresh = build_market_data(min_price=1., live_as_of=cutoff, **options)
        fresh_day = getattr(fresh, 'price_date', None)
        if fresh is None or fresh.n_stocks == 0 or fresh_day is None or fresh_day > cutoff:
            raise ValueError('Harga live untuk sesi selesai terbaru belum tersedia. Perbarui OHLCV lalu coba lagi.')
        if profile_policy != "legacy" and getattr(fresh, 'profile_features', None) is None:
            raise ValueError('Bukti komposisi profil tidak tersedia. Perbarui sumber harga lalu coba lagi.')
        # Never regress to an older provider snapshot after a successful sync.
        if has_features and cached_day is not None and fresh_day < cached_day <= cutoff:
            fresh = cached
        slot.market_data_today = fresh
        slot.live_snapshot_checked_through = cutoff
        slot.live_snapshot_checked_at = time.monotonic()
        return fresh
