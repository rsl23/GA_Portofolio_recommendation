"""Retrospective detector diagnostics, independent from the trading signal."""
from __future__ import annotations
from datetime import date

import numpy as np
import pandas as pd


def reference_labels(price: pd.Series, threshold=.20):
    """20% down / 20% up turning points, confirmed only within supplied history.

    Reference bear spans peak..trough. Unconfirmed terminal swings are censored.
    These labels use hindsight and MUST NOT be supplied to GA or monitoring.
    """
    values = price.to_numpy(dtype=float)
    labels = pd.Series(False, index=price.index, dtype=object)
    episodes = []
    if not len(values):
        return labels, episodes
    peak = trough = 0
    falling = False
    for i in range(1, len(values)):
        if not falling:
            if values[i] >= values[peak]:
                peak = i
            elif values[i] <= values[peak] * (1 - threshold):
                falling, trough = True, i
        elif values[i] < values[trough]:
            trough = i
        elif values[i] >= values[trough] * (1 + threshold):
            labels.iloc[peak:trough + 1] = True
            episodes.append({"start": str(price.index[peak].date()), "end": str(price.index[trough].date()),
                             "confirmed_on": str(price.index[i].date()),
                             "drawdown_pct": float((1 - values[trough] / values[peak]) * 100)})
            falling, peak = False, i
    labels.iloc[peak:] = None
    return labels, episodes


def _ranges(mask):
    ranges, start = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            ranges.append((start, i - 1))
            start = None
    return ranges


def evaluate_bear(price, history, start_date, end_date):
    price = price.sort_index().loc[:str(end_date)]
    labels, episodes = reference_labels(price)
    signals = pd.Series({pd.Timestamp(row["date"]): (row["status"] == "bear"
        if row["status"] in ("bear", "normal") else None) for row in history}, dtype=object)
    selected = price.loc[str(start_date):str(end_date)].index
    ref, signal = labels.reindex(selected), signals.reindex(selected)
    valid = ref.notna() & signal.notna()
    alerts = _ranges((signal.eq(True) & signal.notna()).tolist())
    # Boundary-truncated alerts cannot establish onset/false-alarm rates.
    evaluable = [(a, b) for a, b in alerts if a > 0 and b < len(selected) - 1
                 and valid.iloc[a:b + 1].all() and valid.iloc[a - 1] and valid.iloc[b + 1]]
    hits = sum(bool(ref.iloc[a:b + 1].eq(True).any()) for a, b in evaluable)
    scored = []
    for episode in episodes:
        begin, end = pd.Timestamp(episode["start"]), pd.Timestamp(episode["end"])
        # Episode recall and onset lag require the entire reference episode.
        if not len(selected) or begin < selected[0] or end > selected[-1]:
            continue
        positions = np.flatnonzero((selected >= begin) & (selected <= end))
        if not len(positions) or not valid.iloc[positions].all():
            continue
        detected = [i for i in positions if signal.iloc[i] is True or signal.iloc[i] == True]
        first = detected[0] if detected else None
        after = np.flatnonzero(selected > end)
        recovery = None
        next_start = min((pd.Timestamp(e["start"]) for e in episodes if pd.Timestamp(e["start"]) > end), default=None)
        for i in after:
            if next_start is not None and selected[i] >= next_start:
                break
            if pd.isna(signal.iloc[i]):
                break
            if signal.iloc[i] == False:
                recovery = int(i - positions[-1])
                break
        scored.append({**episode, "detected": first is not None,
                       "onset_lag_sessions": int(first - positions[0]) if first is not None else None,
                       "recovery_lag_sessions": recovery if first is not None else None})
    lags = [e["onset_lag_sessions"] for e in scored if e["onset_lag_sessions"] is not None]
    recovery_lags = [e["recovery_lag_sessions"] for e in scored if e["recovery_lag_sessions"] is not None]
    return {"version": "bear-evaluation-v2", "source": "historical_daily_replay_not_monitor_execution_log",
            "reference": "ex_post_20pct_decline_20pct_rebound_peak_to_trough", "threshold_pct": 20,
            "start_date": str(selected[0].date()) if len(selected) else None,
            "end_date": str(selected[-1].date()) if len(selected) else None,
            "sessions": len(selected), "scored_sessions": int(valid.sum()),
            "censored_sessions": int(ref.isna().sum()), "signal_missing_sessions": int(signal.isna().sum()),
            "episodes": scored, "episode_count": len(scored), "evaluable_alerts": len(evaluable),
            "excluded_alerts": len(alerts) - len(evaluable),
            "metric_status": {
                "precision": "available" if evaluable else "incomplete_alerts" if alerts else "no_alerts",
                "recall": "available" if scored else "no_complete_reference_episodes",
                "onset_lag": "available" if lags else "no_detected_episodes" if scored else "no_complete_reference_episodes",
                "recovery_lag": "available" if recovery_lags else "recovery_not_observed" if lags else "no_detected_episodes",
                "false_alerts": "available" if valid.sum() and (evaluable or not alerts) else "incomplete_alerts",
            },
            "alert_precision_pct": hits / len(evaluable) * 100 if evaluable else None,
            "episode_recall_pct": sum(e["detected"] for e in scored) / len(scored) * 100 if scored else None,
            "false_alerts_per_year": ((len(evaluable) - hits) / (valid.sum() / 252)
                                      if valid.sum() and (evaluable or not alerts) else None),
            "median_onset_lag_sessions": float(np.median(lags)) if lags else None,
            "median_recovery_lag_sessions": float(np.median(recovery_lags)) if recovery_lags else None,
            "counterfactual_effect": "not_evaluated_requires_paired_bear_on_off_backtests",
            "warnings": ["Label 20% adalah definisi eksperimen ex-post; bukan ground truth universal dan tidak dipakai sebagai sinyal.",
                         "Replay menilai status harian detector dari IHSG; tidak menilai severity, breadth, atau jadwal monitoring mingguan/bulanan.",
                         "Episode/alert yang terpotong batas periode dan swing akhir yang belum terkonfirmasi dikeluarkan dari skor terkait.",
                         "Precision menghitung proporsi alert yang beririsan episode acuan; recall menghitung proporsi episode yang tersentuh sinyal. Ini bukan implementasi lengkap range precision/recall Tatbul.",
                         "Manfaat ekonomi detector belum dapat disimpulkan tanpa simulasi berpasangan dengan biaya, seed, dan arus dana yang sama."]}


def historical_bear_evaluation(start_date, end_date, scope="portfolio"):
    from src.preprocessing.bear_market_detection import (
        CONFIG, _historical_context, _series, _clean, detect_market_regime,
    )
    price = _clean(_series(_historical_context(), "^JKSE"), end_date)
    if price.empty:
        return {"status": "unavailable", "reason": "Arsip IHSG tidak tersedia."}
    cutoff = price.index[-1].date()
    if scope not in ("portfolio", "market_history"):
        raise ValueError("Cakupan evaluasi bear tidak dikenal.")
    if cutoff < start_date and scope == "portfolio":
        return {"status": "unavailable", "reason": f"Arsip IHSG hanya sampai {cutoff}, sebelum awal periode evaluasi."}
    result = detect_market_regime(price, cutoff, return_history=True)
    history = result.get("history", [])
    if not history:
        return {"status": "unavailable", "reason": result.get("reason", "Histori awal detector belum cukup.")}
    actual_start = (date.fromisoformat(history[0]["date"]) if scope == "market_history" else start_date)
    evaluation = evaluate_bear(price, history, actual_start, end_date)
    evaluation.update(status="available", detector_version=CONFIG.version,
                      archive_start=str(price.index[0].date()), archive_end=str(cutoff),
                      requested_start=str(start_date), requested_end=str(end_date), scope=scope,
                      calendar_version="ihsg-observed-sessions-v1", parameters=result["parameters"])
    return evaluation
