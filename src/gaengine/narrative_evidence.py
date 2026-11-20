"""Small, deterministic comparisons for explaining a completed GA decision.

These are local diagnostics, not another optimisation or proof of why a search
chose a stock. Each score uses the original decision inputs and scoring policy.
"""
from copy import copy
import math

import numpy as np


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics(candidate, config):
    weights = candidate.lots * config.data.prices_per_lot / config.budget
    daily = weights @ config.data.returns
    daily = daily[np.isfinite(daily)]
    return {
        "fitness": _finite(candidate.fitness),
        "sharpe_portofolio": _finite(candidate.sharpe_ratio),
        "volatilitas_modal_tahunan": _finite(np.std(daily) * np.sqrt(252)) if daily.size else None,
        "return_aritmetika_modal_tahunan": _finite(np.mean(daily) * 252) if daily.size else None,
        "mdd_modal": _finite(candidate.max_drawdown),
        "jumlah_saham": int(candidate.n_active),
        "nilai_saham_harga_keputusan": _finite(candidate.cost),
    }


def build_selection_evidence(solution, config, *, unavailable_reason=None):
    """Compare equal invested allocations and removing each selected stock.

    Equal allocations use the SAME selected names and gross invested amount,
    floored to lots. Removal leaves the remaining lots unchanged and cash idle;
    this changes exposure, so it is not an isolated causal stock attribution.
    Unaffordable or constraint-breaking alternatives never count as evidence.
    Config/data copies keep narrative work from mutating the live GA objects.
    """
    result = {
        "status": "tidak_tersedia",
        "batas_interpretasi": (
            "Pembanding lokal pada data historis yang sama, bukan semua alternatif, "
            "bukan bukti optimum global atau sebab pasti pemilihan. Penghapusan saham "
            "memindahkan nilainya ke kas, sehingga exposure dan biaya juga berubah."
        ),
        "bobot_merata": None,
        "tanpa_satu_saham": [],
    }
    if unavailable_reason:
        result["alasan"] = unavailable_reason
        return result
    if config is None or getattr(config, "data", None) is None or solution.lots is None:
        result["alasan"] = "Data keputusan atau lot tidak tersedia."
        return result
    if (getattr(config, "rebalance_model", "legacy") == "guarded_v1"
            or getattr(config, "fitness_mode", "rebalance_aware") == "decision_aware"):
        result["alasan"] = "Pembanding untuk model utility ini belum dihitung dengan basis yang setara."
        return result

    from .chromosome import Chromosome
    from .guarded_rebalance import transaction_account

    cfg = copy(config)
    cfg.data = copy(config.data)
    cfg.old_lots = dict(config.old_lots or {})
    cfg.old_prices_per_lot = dict(config.old_prices_per_lot or {})
    lots = np.asarray(solution.lots, dtype=int).copy()
    prices = np.asarray(cfg.data.prices_per_lot, dtype=float)
    active = np.flatnonzero(lots > 0)
    if not len(active) or len(active) > 10 or cfg.budget <= 0:
        result["alasan"] = "Pembanding memerlukan 1–10 posisi aktif dan modal positif."
        return result
    original = _metrics(solution, cfg)
    price_map = {code: float(price) for code, price in zip(cfg.data.stock_codes, prices)}
    all_prices = {**cfg.old_prices_per_lot, **price_map}

    def evaluate(target):
        target = np.asarray(target, dtype=int)
        count = int(np.count_nonzero(target))
        if not cfg.min_stocks <= count <= cfg.max_stocks:
            return {"layak": False, "alasan": "Jumlah saham melanggar batas minimum/maksimum."}
        if (getattr(cfg, "allow_cash", False)
                and float(target @ prices) > cfg.budget * cfg.max_exposure + .01):
            return {"layak": False, "alasan": "Melebihi batas exposure."}
        # Exclude buys of ineligible names and trades in frozen positions.
        eligible = getattr(cfg.data, "buy_eligible", None)
        if eligible is not None and np.any((target > lots) & ~np.asarray(eligible, dtype=bool)):
            return {"layak": False, "alasan": "Menambah saham yang tidak boleh dibeli."}
        frozen = set(getattr(cfg.data, "untraded_tickers", None) or [])
        if any(code in frozen and target[i] != lots[i] for i, code in enumerate(cfg.data.stock_codes)):
            return {"layak": False, "alasan": "Mengubah posisi tanpa perdagangan."}
        try:
            target_map = dict(zip(cfg.data.stock_codes, map(int, target)))
            cash, fees = transaction_account(cfg.old_lots, target_map,
                {code: price / 100.0 for code, price in all_prices.items()},
                float(cfg.old_cash if cfg.old_cash is not None else cfg.budget),
                cfg.buy_fee_rate, cfg.sell_fee_rate)
            if cash < -.01:
                return {"layak": False, "alasan": "Kas setelah biaya transaksi negatif."}
            candidate = Chromosome(cfg, solution.genes)
            candidate.exposure = solution.exposure
            candidate.evaluate_fitness(lots_override=target)
        except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
            return {"layak": False, "alasan": str(exc)}
        if (not candidate.budget_ok or not candidate.diversification_ok
                or not candidate.profile_constraints_ok or _finite(candidate.fitness) is None):
            return {"layak": False, "alasan": "Melanggar batas kelayakan penilaian GA."}
        metrics = _metrics(candidate, cfg)
        return {
            "layak": True, **metrics,
            "kas_setelah_biaya": _finite(cash), "biaya_transaksi": _finite(fees),
            "selisih_fitness_hasil_dikurangi_pembanding": _finite(solution.fitness - candidate.fitness),
            "selisih_volatilitas_hasil_dikurangi_pembanding": _finite(
                original["volatilitas_modal_tahunan"] - metrics["volatilitas_modal_tahunan"]),
        }

    try:
        # Reproduce the selected score first. Never compare different objectives.
        selected = evaluate(lots)
        if (not selected["layak"] or not math.isclose(
                selected["fitness"], solution.fitness, rel_tol=1e-9, abs_tol=1e-9)):
            result["alasan"] = "Skor hasil tidak dapat direproduksi pada basis pembanding."
            return result
        equal = np.zeros_like(lots)
        equal[active] = np.floor(float(lots @ prices) / len(active) / prices[active]).astype(int)
        result["bobot_merata"] = {
            "metode": "Nama saham sama; nilai saham dibagi merata sebelum pembulatan lot; sisa menjadi kas.",
            **evaluate(equal),
        }
        for i in active:
            removed = lots.copy()
            removed[i] = 0
            result["tanpa_satu_saham"].append({
                "ticker": cfg.data.stock_codes[i],
                "metode": "Lot saham ini menjadi nol; lot lain tetap; dana menjadi kas.",
                **evaluate(removed),
            })
        result["status"] = "tersedia"
        result["hasil"] = selected
    except (ValueError, TypeError, KeyError, AttributeError, FloatingPointError) as exc:
        result["alasan"] = f"Pembanding tidak dapat dihitung: {exc}"
    return result


def risk_notes(solution, config):
    """Short investor-facing limitations, also usable with the template fallback."""
    notes = []
    diagnostics = getattr(solution, "fitness_diagnostics", None) or {}
    if diagnostics.get("policy") == "legacy_cvar_v1":
        count = diagnostics.get("scenario_count")
        horizon = diagnostics.get("horizon_days", getattr(config, "forecast_horizon_days", None))
        if count is not None:
            notes.append(f"Pengujian kerugian memakai {count} skenario historis masing-masing {horizon} sesi.")
        if diagnostics.get("downside_cvar") == 0:
            notes.append("Hasil rata-rata ekor skenario tidak menimbulkan penalti kerugian; ini tidak berarti peluang rugi nol.")
    evidence = getattr(getattr(config, "data", None), "market_stress", None)
    if evidence is not None:
        if evidence.history_end is not None and evidence.history_end < evidence.as_of:
            notes.append(f"Arsip pengujian tekanan pasar tersedia hingga {evidence.history_end}, "
                         f"lebih awal dari tanggal penilaian {evidence.as_of}.")
        coverage = getattr(solution, "market_stress_coverage", None)
        if coverage is not None and coverage < 1:
            notes.append(f"Cakupan bukti historis saham {coverage:.1%}; bagian yang belum terbukti tetap memiliki ketidakpastian.")
    return notes
