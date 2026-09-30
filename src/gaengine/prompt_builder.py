"""
Prompt Builder - Bahan Penjelasan Portofolio untuk LLM
=====================================================

Modul ini merakit BAHAN (konteks terstruktur) + PROMPT yang dikirim ke LLM
(Gemini, lihat :mod:`src.backend.services.gemini_llm_caller`) supaya hasil
Algoritma Genetika (GA) bisa DIJELASKAN: mengapa portofolio ini yang terpilih,
bagaimana GA menilainya (metrik fitness function), dan apa arti angka-angka
tersebut bagi investor.

Prinsip modul ini:
    - MURNI / tanpa efek samping: TIDAK mengakses jaringan, database, maupun
      SDK LLM. Semua angka diambil dari objek hasil GA yang sudah ada
      (``Chromosome``, ``MarketData``, ``GAConfig``) sehingga mudah diuji.
    - ANTI-HALUSINASI: seluruh fakta dihitung di Python (deterministik) dan
      diserahkan sebagai tabel; tugas LLM hanya MENJELASKAN, bukan menghitung
      atau mengarang angka/berita.

Isi utama:
    build_portfolio_context(...)   -> dict metrik lengkap (JSON-serializable)
    render_context_text(...)       -> tabel teks siap tempel ke prompt
    build_explanation_prompt(...)  -> PortfolioPrompt (system + user prompt)
    build_narrative_prompt(...)    -> gabungan context + prompt (dipakai
                                      oleh gemini_llm_caller)

Satuan yang dipakai di konteks (mengikuti MarketData/loader):
    PER, PBV        : kelipatan (x)
    ROE             : persen (%)
    DER             : rasio (x)  -> mis. 0.53 = 53%
    Dividend Yield  : persen (%) -> hasil konversi dari fraksi loader
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Label & teks baku
# ----------------------------------------------------------------------
# Urutan kolom market_data.fundamental_metrics (lihat _provide_*_data di
# src/gaengine/data_loader.py): [PER, PBV, ROE, DER, DivYld].
FUNDAMENTAL_LABELS: Tuple[str, ...] = (
    "PER (x)",
    "PBV (x)",
    "ROE (%)",
    "DER (rasio)",
    "Dividend Yield (%)",
)

# Penjelasan rumus fitness yang benar-benar dipakai Chromosome.evaluate_fitness.
FITNESS_FORMULA_TEXT = (
    "Fitness = Sharpe - lambda*|MDD| - gamma*RataKorelasi + alpha*SkorFundamental "
    "- DeathPenalty_Budget - DeathPenalty_Diversifikasi"
)

# Peran tiap koefisien, supaya LLM menjelaskan bobot penalti dengan tepat.
COEFFICIENT_NOTES = {
    "lambda_mdd": "lambda = bobot penalti maximum drawdown (makin besar = makin "
                  "menghindari risiko turun dalam; ditentukan profil risiko).",
    "gamma_korelasi": "gamma = bobot penalti korelasi rata-rata antar saham "
                      "(makin besar = makin menekan risiko portofolio bergerak seragam).",
    "alpha_fundamental": "alpha = bobot bonus skor fundamental (PER/PBV/ROE/DER/DivYield "
                         "yang sudah dinormalisasi min-max menjadi skor 0-1).",
    "death_penalty": "DeathPenalty = nilai penalti besar ketika alokasi melebihi budget "
                     "atau jumlah saham aktif di luar batas minimum/maksimum.",
}

DEFAULT_SYSTEM_INSTRUCTION = (
    "Anda adalah analis kuantitatif portofolio saham Indonesia (Bursa Efek Indonesia/IDX) "
    "yang bertugas menjelaskan hasil optimasi Algoritma Genetika (GA) kepada investor "
    "ritel berbahasa Indonesia.\n"
    "ATURAN WAJIB:\n"
    "1. Gunakan HANYA angka dan fakta yang tertulis pada DATA PORTOFOLIO. Jangan "
    "menghitung ulang, jangan menambah angka, dan jangan mengarang berita, rumor, "
    "atau kondisi pasar yang tidak diberikan.\n"
    "2. Jika ada metrik yang bernilai 'tidak tersedia', sebutkan keterbatasan itu "
    "apa adanya, jangan menebak.\n"
    "3. Selalu tulis satuan (x, %, rasio) dan jangan mencampur satuan ROE (%) dengan "
    "DER (rasio).\n"
    "4. Jelaskan KETERKAITAN sebab-akibat: angka metrik -> kontribusi ke fitness -> "
    "kenapa saham tersebut terpilih/tertinggal dalam portofolio.\n"
    "5. Bahasa Indonesia yang jelas, mengalir, untuk pembaca non-teknis; hindari "
    "jargon berlebihan dan hindari format Markdown (tanpa '#', '*', '-' sebagai "
    "penanda daftar).\n"
    "6. Akhiri dengan satu kalimat disclaimer bahwa ini hasil simulasi optimasi, "
    "bukan nasihat keuangan.\n"
    "7. Jangan pernah menyebut bahwa Anda sebuah model bahasa atau menyebut nama model."
)



# ----------------------------------------------------------------------
# Utilitas format (semua nilai NaN/inf -> None agar aman untuk JSON & prompt)
# ----------------------------------------------------------------------
def _num(value: Any, digits: int = 4) -> Optional[float]:
    """Konversi nilai apa pun menjadi float bersih: None/NaN/inf -> None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, digits)


def _rupiah(value: Any) -> str:
    """Format rupiah tanpa desimal, mis. 10.000.000 -> 'Rp10.000.000'."""
    v = _num(value, 2)
    if v is None:
        return "tidak tersedia"
    return f"Rp{v:,.0f}".replace(",", ".")


def _pct(value: Optional[float], digits: int = 2) -> str:
    """Format fraksi (0.1234) menjadi persen ('12.34%'); None -> teks ramah."""
    v = _num(value, 8)
    if v is None:
        return "tidak tersedia"
    return f"{v * 100:.{digits}f}%"


def _num_str(value: Optional[float], digits: int = 3) -> str:
    """Format angka biasa; None -> 'tidak tersedia'."""
    v = _num(value, 8)
    if v is None:
        return "tidak tersedia"
    return f"{v:.{digits}f}"


def _to_iso(value: Any) -> Optional[str]:
    """Ubah date/datetime/objek ber-isoformat menjadi string ISO."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class PortfolioPrompt:
    """Hasil akhir prompt siap kirim ke LLM (system instruction + user prompt)."""

    system_instruction: str
    user_prompt: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Bentuk dict (memudahkan logging / penyimpanan / pengujian)."""
        return {
            "system_instruction": self.system_instruction,
            "user_prompt": self.user_prompt,
            "context": self.context,
        }

    @property
    def full_text(self) -> str:
        """System instruction + user prompt sebagai satu teks (untuk debugging)."""
        return f"{self.system_instruction}\n\n{self.user_prompt}"


# ----------------------------------------------------------------------
# Perhitungan statistik portofolio (semua deterministik, dari data GA)
# ----------------------------------------------------------------------
def _weights_from_lots(lots: Sequence[int], prices: Sequence[float]) -> np.ndarray:
    """Bobot alokasi = (lot x harga per lot) / total; nol bila total 0."""
    lots_arr = np.asarray(list(lots), dtype=float)
    prices_arr = np.asarray(list(prices), dtype=float)
    alloc = lots_arr * prices_arr
    total = float(alloc.sum())
    if total <= 0:
        return np.zeros_like(alloc)
    return alloc / total


def _daily_series(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rangkaian return harian portofolio (bobot statis) tanpa NaN/inf."""
    ret = np.asarray(returns, dtype=float)
    if ret.size == 0 or weights.size == 0 or ret.shape[0] != weights.size:
        return np.asarray([], dtype=float)
    daily = (ret * weights[:, None]).sum(axis=0)
    return daily[~np.isnan(daily) & ~np.isinf(daily)]


def _annual_stats(daily: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Statistik tahunan dari rangkaian return harian (paritas dengan Chromosome):
    return tahunan = rata-rata x 252, volatilitas = std x sqrt(252),
    dan maximum drawdown dari kurva ekuitas kumulatif.
    """
    if daily.size == 0:
        return {"return_tahunan": None, "volatilitas_tahunan": None, "max_drawdown": None}
    ret_ann = float(daily.mean()) * 252.0
    vol_ann = float(daily.std()) * math.sqrt(252.0)
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    mdd = float(dd.min()) if dd.size else 0.0
    if math.isnan(mdd):
        mdd = 0.0
    return {
        "return_tahunan": _num(ret_ann),
        "volatilitas_tahunan": _num(vol_ann),
        "max_drawdown": _num(abs(mdd)),
    }


def _stock_stats(returns: np.ndarray, index: int) -> Dict[str, Optional[float]]:
    """Statistik tahunan satu emiten dari baris `index` matriks returns."""
    row = np.asarray(returns, dtype=float)[index]
    row = row[~np.isnan(row) & ~np.isinf(row)]
    return _annual_stats(row)


def _pair_correlations(
    correlation: np.ndarray, indices: Sequence[int], max_pairs: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pasangan korelasi TERTINGGI & TERENDAH antar emiten terpilih.

    Dipakai untuk menjelaskan term `- gamma * RataKorelasi`: pasangan berkorelasi
    tinggi-lah yang "menarik" nilai rata-rata korelasi naik (dan fitness turun).
    """
    idx = list(indices)
    if len(idx) < 2:
        return [], []
    corr = np.asarray(correlation, dtype=float)
    pairs: List[Dict[str, Any]] = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            try:
                value = float(corr[idx[a], idx[b]])
            except (IndexError, ValueError):
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            pairs.append({"i": int(idx[a]), "j": int(idx[b]), "korelasi": _num(value, 4)})

    tertinggi = sorted(pairs, key=lambda p: p["korelasi"], reverse=True)[:max_pairs]
    terendah = sorted(pairs, key=lambda p: p["korelasi"])[:max_pairs]
    return tertinggi, terendah


def _fitness_breakdown(solution, config, indices: Sequence[int]) -> Dict[str, Any]:
    """
    Bedah fitness menjadi komponen-komponen yang benar-benar dijumlahkan
    Chromosome.evaluate_fitness, sehingga LLM bisa menjelaskan dari mana nilai
    fitness berasal (dan komponen mana yang paling menentukan).
    """
    lam = float(getattr(config, "lambda_mdd", 0.0) or 0.0)
    gamma = float(getattr(config, "correlation_penalty", 0.0) or 0.0)
    alpha = float(getattr(config, "fundamental_bonus", 0.0) or 0.0)
    death = float(getattr(config, "death_penalty", 0.0) or 0.0)

    sharpe = _num(getattr(solution, "sharpe_ratio", None)) or 0.0
    mdd = abs(_num(getattr(solution, "max_drawdown", None)) or 0.0)
    avg_corr = _num(getattr(solution, "avg_correlation", None)) or 0.0
    funda = _num(getattr(solution, "skor_fundamental", None)) or 0.0

    penalti_mdd = -lam * mdd
    penalti_korelasi = -gamma * avg_corr
    bonus_fundamental = alpha * funda
    budget_ok = bool(getattr(solution, "budget_ok", True))
    div_ok = bool(getattr(solution, "diversification_ok", True))
    penalti_budget = 0.0 if budget_ok else -death
    penalti_div = 0.0 if div_ok else -death

    komponen = {
        "sharpe_ratio": _num(sharpe),
        "penalti_max_drawdown": _num(penalti_mdd),
        "penalti_korelasi": _num(penalti_korelasi),
        "bonus_fundamental": _num(bonus_fundamental),
        "penalti_budget": _num(penalti_budget),
        "penalti_diversifikasi": _num(penalti_div),
    }
    total = _num(getattr(solution, "fitness", None))
    jumlah_komponen = _num(sum(v for v in komponen.values() if v is not None))

    # Komponen dengan pengaruh mutlak terbesar -> penentu utama nilai fitness.
    penentu = max(
        ((k, abs(v)) for k, v in komponen.items() if v is not None),
        key=lambda kv: kv[1],
        default=(None, 0.0),
    )
    return {
        "koefisien": {
            "lambda_mdd": _num(lam),
            "gamma_korelasi": _num(gamma),
            "alpha_fundamental": _num(alpha),
            "death_penalty": _num(death, 2),
        },
        "komponen": komponen,
        "jumlah_komponen": jumlah_komponen,
        "fitness_dilaporkan": total,
        "komponen_paling_menentukan": penentu[0],
        "jumlah_emiten_dinilai": len(list(indices)),
    }



def _emiten_rows(
    solution,
    market_data,
    allocations: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[int]]:
    """
    Susun baris metrik untuk SETIAP emiten pemenang (lot > 0).

    Angka lot/harga/bobot diambil dari objek hasil GA (``solution.lots`` +
    ``market_data.prices_per_lot``) sehingga identik dengan yang disimpan
    controller; parameter `allocations` (payload controller) hanya dipakai
    sebagai cadangan bila `solution.lots` kosong.

    Returns:
        (rows, lots, weights, active_indices)
    """
    codes = list(market_data.stock_codes)
    prices = np.asarray(market_data.prices_per_lot, dtype=float)
    returns = np.asarray(market_data.returns, dtype=float)
    corr = np.asarray(market_data.correlation, dtype=float)
    fund_scores = np.asarray(market_data.fundamental_scores, dtype=float)
    fund_metrics = getattr(market_data, "fundamental_metrics", None)
    n = len(codes)

    lots_raw = getattr(solution, "lots", None)
    if lots_raw is not None and len(lots_raw) == n:
        lots = np.asarray(lots_raw, dtype=int)
    elif allocations:
        lots = np.zeros(n, dtype=int)
        pos = {c: i for i, c in enumerate(codes)}
        for row in allocations:
            ticker = row.get("ticker")
            if ticker in pos:
                lots[pos[ticker]] = int(row.get("lots", 0) or 0)
    else:
        lots = np.zeros(n, dtype=int)

    weights = _weights_from_lots(lots, prices)
    active = [i for i in range(n) if int(lots[i]) > 0]

    rows: List[Dict[str, Any]] = []
    for i in active:
        # Korelasi: rata-rata NILAI ABSOLUT dengan emiten terpilih lainnya
        # (sama seperti perhitungan Chromosome: np.abs(vals).mean()).
        corr_vals: List[float] = []
        for j in active:
            if j == i:
                continue
            try:
                v = float(corr[i, j])
            except (IndexError, ValueError):
                continue
            if not (math.isnan(v) or math.isinf(v)):
                corr_vals.append(abs(v))
        corr_avg = float(np.mean(corr_vals)) if corr_vals else None

        if fund_metrics is not None and np.asarray(fund_metrics).shape[0] > i:
            row_metrics = np.asarray(fund_metrics, dtype=float)[i]
            per_val = _num(row_metrics[0], 4)
            pbv_val = _num(row_metrics[1], 4)
            roe_val = _num(row_metrics[2], 4)
            der_val = _num(row_metrics[3], 4)
            div_val = _num(row_metrics[4] * 100.0, 4) if len(row_metrics) > 4 else None
        else:
            per_val = pbv_val = roe_val = der_val = div_val = None

        stats = _stock_stats(returns, i)
        rows.append({
            "ticker": str(codes[i]),
            "lots": int(lots[i]),
            "harga_per_lot": _num(prices[i], 2),
            "harga_per_lembar": _num(prices[i] / 100.0, 2),
            "alokasi": _num(lots[i] * prices[i], 2),
            "bobot": _num(weights[i], 6),
            "skor_fundamental_0_1": _num(fund_scores[i], 4) if fund_scores.size > i else None,
            "fundamental": {
                "per_x": per_val,
                "pbv_x": pbv_val,
                "roe_persen": roe_val,
                "der_rasio": der_val,
                "dividend_yield_persen": div_val,
            },
            "return_tahunan": stats["return_tahunan"],
            "volatilitas_tahunan": stats["volatilitas_tahunan"],
            "max_drawdown": stats["max_drawdown"],
            "korelasi_abs_rata2_dengan_emiten_lain": _num(corr_avg, 4),
            "kontribusi_ke_return_portofolio": _num(weights[i] * (stats["return_tahunan"] or 0.0), 6),
        })

    rows.sort(key=lambda r: r["alokasi"] or 0.0, reverse=True)
    return rows, lots, weights, active



def build_portfolio_context(
    solution,
    market_data,
    config=None,
    *,
    allocations: Optional[Sequence[Dict[str, Any]]] = None,
    mode: str = "live",
    date_ref: Optional[Any] = None,
    budget: Optional[float] = None,
    expected_return: Optional[float] = None,
    generations_run: Optional[int] = None,
    max_pairs: int = 5,
) -> Dict[str, Any]:
    """
    Rakit KONTEKS lengkap hasil GA menjadi dict siap-JSON untuk dijelaskan LLM.

    Semua angka dihitung di sini (deterministik) supaya LLM tidak perlu
    menghitung apa pun dan tidak bisa berhalusinasi nilai.

    Args:
        solution        : Chromosome terbaik hasil GA (punya .fitness, .lots, dst).
        market_data     : MarketData yang dipakai GA (kode, harga, returns, dst).
        config          : GAConfig (koefisien fitness, batas saham, parameter GA).
        allocations     : opsional, daftar alokasi dari controller (cadangan bila
                          `solution.lots` tidak tersedia).
        mode            : "live" atau "backtest" (konteks waktu portofolio).
        date_ref        : tanggal acuan (date/datetime/str) untuk mode backtest.
        budget          : modal pengguna (default: config.budget bila ada).
        expected_return : estimasi return tahunan (default: dihitung dari returns).
        generations_run : jumlah generasi yang benar-benar dijalankan (opsional).
        max_pairs       : jumlah pasangan korelasi ekstrem yang dilaporkan.

    Returns:
        Dict[str, Any] berisi: mode, periode, profil_risiko, modal, kinerja,
        dekomposisi_fitness, emiten, korelasi, parameter_ga, catatan.
    """
    rows, lots, weights, active = _emiten_rows(solution, market_data, allocations)
    prices = np.asarray(market_data.prices_per_lot, dtype=float)

    total_terpakai = float((np.asarray(lots, dtype=float) * prices).sum())
    total_budget = _num(budget, 2)
    if total_budget is None:
        total_budget = _num(getattr(config, "budget", None), 2) or total_terpakai
    sisa = (total_budget - total_terpakai) if total_budget is not None else None

    daily = _daily_series(np.asarray(market_data.returns, dtype=float), weights)
    port_stats = _annual_stats(daily)
    if expected_return is None:
        expected_return = port_stats["return_tahunan"]

    tertinggi, terendah = _pair_correlations(
        np.asarray(market_data.correlation, dtype=float), active, max_pairs
    )
    kode_emiten = list(market_data.stock_codes)
    for pasangan in (tertinggi, terendah):
        for p in pasangan:
            p["pasangan"] = [str(kode_emiten[p["i"]]), str(kode_emiten[p["j"]])]

    breakdown = _fitness_breakdown(solution, config, active)

    risk_free = _num(getattr(market_data, "risk_free_rate", None), 6)
    excess = None
    if port_stats["return_tahunan"] is not None and risk_free is not None:
        excess = _num(port_stats["return_tahunan"] - risk_free, 6)

    context: Dict[str, Any] = {
        "mode": str(mode).lower(),
        "tanggal_acuan": _to_iso(date_ref),
        "profil_risiko": getattr(config, "risk_profile", None),
        "rumus_fitness": FITNESS_FORMULA_TEXT,
        "penjelasan_koefisien": dict(COEFFICIENT_NOTES),
        "modal": {
            "budget": total_budget,
            "terpakai": _num(total_terpakai, 2),
            "sisa": _num(sisa, 2),
            "persen_terpakai": _num(total_terpakai / total_budget, 6) if total_budget else None,
        },
        "kinerja_portofolio": {
            "fitness": _num(getattr(solution, "fitness", None)),
            "sharpe_ratio": _num(getattr(solution, "sharpe_ratio", None)),
            "return_tahunan_historis": port_stats["return_tahunan"],
            "expected_return_tahunan": _num(expected_return, 6),
            "volatilitas_tahunan": port_stats["volatilitas_tahunan"],
            "max_drawdown": _num(abs(getattr(solution, "max_drawdown", 0.0) or 0.0), 6),
            "max_drawdown_dari_seri_harga": port_stats["max_drawdown"],
            "rata_korelasi": _num(getattr(solution, "avg_correlation", None)),
            "skor_fundamental_0_1": _num(getattr(solution, "skor_fundamental", None)),
            "risk_free_rate_tahunan": risk_free,
            "excess_return_di_atas_risk_free": excess,
            "jumlah_hari_return": int(daily.size) if daily.size else 0,
        },
        "dekomposisi_fitness": breakdown,
        "constraint": {
            "min_saham": getattr(config, "min_stocks", None),
            "max_saham": getattr(config, "max_stocks", None),
            "n_saham_aktif": int(getattr(solution, "n_active", len(active))),
            "budget_terpenuhi": bool(getattr(solution, "budget_ok", True)),
            "diversifikasi_terpenuhi": bool(getattr(solution, "diversification_ok", True)),
        },
        "kandidat_saham_awal": int(getattr(market_data, "n_stocks", 0)),
        "emiten_terpilih": rows,
        "korelasi": {
            "pasangan_tertinggi": tertinggi,
            "pasangan_terendah": terendah,
        },
        "parameter_ga": {
            "population_size": getattr(config, "population_size", None),
            "generations": getattr(config, "generations", None),
            "generations_dijalankan": generations_run,
            "crossover_rate": getattr(config, "crossover_rate", None),
            "tournament_size": getattr(config, "tournament_size", None),
            "elitism_count": getattr(config, "elitism_count", None),
            "mutation_rate_awal": getattr(config, "mutation_rate_start", None),
            "mutation_rate_akhir": getattr(config, "mutation_rate_end", None),
            "early_stop_patience": getattr(config, "early_stop_patience", None),
            "seed": getattr(config, "seed", None),
        },
    }
    context["catatan_otomatis"] = _auto_insights(context)
    return context



def _auto_insights(context: Dict[str, Any]) -> List[str]:
    """
    Fakta terhitung (bukan opini) yang membantu LLM menjelaskan hasil GA:
    dekomposisi fitness, komponen dominan, korelasi ekstrem, pemakaian budget,
    diversifikasi, emiten ber bobot terbesar, dan sebaran skor fundamental.
    """
    kin = context["kinerja_portofolio"]
    dek = context["dekomposisi_fitness"]
    komp = dek["komponen"]
    modal = context["modal"]
    cons = context["constraint"]
    catatan: List[str] = []

    if kin.get("fitness") is None:
        catatan.append("Nilai fitness belum tersedia (kromosom belum dievaluasi).")
    else:
        catatan.append(
            "Fitness " + _num_str(kin["fitness"]) + " = Sharpe " + _num_str(komp["sharpe_ratio"])
            + " + penalti MDD " + _num_str(komp["penalti_max_drawdown"])
            + " + penalti korelasi " + _num_str(komp["penalti_korelasi"])
            + " + bonus fundamental " + _num_str(komp["bonus_fundamental"])
            + " + penalti budget " + _num_str(komp["penalti_budget"])
            + " + penalti diversifikasi " + _num_str(komp["penalti_diversifikasi"])
            + "."
        )
    if dek.get("komponen_paling_menentukan"):
        catatan.append(
            "Komponen dengan pengaruh absolut terbesar pada fitness: "
            + str(dek["komponen_paling_menentukan"]) + "."
        )
    if kin.get("excess_return_di_atas_risk_free") is not None:
        catatan.append(
            "Return tahunan portofolio " + _pct(kin["return_tahunan_historis"])
            + " dibandingkan risk-free " + _pct(kin["risk_free_rate_tahunan"])
            + " -> selisih " + _pct(kin["excess_return_di_atas_risk_free"])
            + " dengan volatilitas tahunan " + _pct(kin["volatilitas_tahunan"]) + "."
        )
    tertinggi = context["korelasi"]["pasangan_tertinggi"]
    terendah = context["korelasi"]["pasangan_terendah"]
    if tertinggi:
        p = tertinggi[0]
        catatan.append(
            "Pasangan paling bergerak bersama: " + " & ".join(p["pasangan"])
            + " (korelasi " + _num_str(p["korelasi"]) + ")."
        )
    if terendah:
        p = terendah[0]
        catatan.append(
            "Pasangan paling saling menyeimbangkan: " + " & ".join(p["pasangan"])
            + " (korelasi " + _num_str(p["korelasi"]) + ")."
        )
    catatan.append(
        "Dana terpakai " + _rupiah(modal["terpakai"]) + " dari budget "
        + _rupiah(modal["budget"]) + " (" + _pct(modal["persen_terpakai"])
        + "), sisa " + _rupiah(modal["sisa"]) + "."
    )
    catatan.append(
        "Jumlah saham aktif " + str(cons["n_saham_aktif"]) + " dari batas "
        + str(cons["min_saham"]) + "-" + str(cons["max_saham"]) + " (diversifikasi "
        + ("terpenuhi" if cons["diversifikasi_terpenuhi"] else "TIDAK terpenuhi")
        + "; budget " + ("terpenuhi" if cons["budget_terpenuhi"] else "TIDAK terpenuhi") + ")."
    )
    emiten = context["emiten_terpilih"]
    if emiten:
        utama = emiten[0]
        catatan.append(
            "Alokasi terbesar: " + str(utama["ticker"]) + " " + _pct(utama["bobot"])
            + " (" + str(utama["lots"]) + " lot) dengan skor fundamental "
            + _num_str(utama["skor_fundamental_0_1"]) + "."
        )
        berskor = [e for e in emiten if e.get("skor_fundamental_0_1") is not None]
        if berskor:
            terbaik = max(berskor, key=lambda e: e["skor_fundamental_0_1"])
            terlemah = min(berskor, key=lambda e: e["skor_fundamental_0_1"])
            catatan.append(
                "Skor fundamental tertinggi: " + str(terbaik["ticker"]) + " ("
                + _num_str(terbaik["skor_fundamental_0_1"]) + "); terendah: "
                + str(terlemah["ticker"]) + " (" + _num_str(terlemah["skor_fundamental_0_1"]) + ")."
            )
    if str(context.get("mode")) == "backtest":
        catatan.append(
            "Portofolio ini hasil SIMULASI BACKTEST per " + str(context.get("tanggal_acuan"))
            + " memakai data historis (bukan kondisi pasar hari ini)."
        )
    else:
        catatan.append("Portofolio ini dihitung dari data pasar terbaru (mode LIVE).")
    return catatan



def _cell(value: Any, digits: int = 3) -> str:
    """Isi sel tabel: angka rapi, atau '-' bila data tidak tersedia."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _num_str(value, digits)
    return str(value)


def render_context_text(context: Dict[str, Any]) -> str:
    """
    Render konteks (dict dari :func:`build_portfolio_context`) menjadi teks
    berisi tabel-tabel rapi, siap ditempel ke prompt LLM.
    """
    kin = context["kinerja_portofolio"]
    dek = context["dekomposisi_fitness"]
    komp, koef = dek["komponen"], dek["koefisien"]
    modal, cons, ga = context["modal"], context["constraint"], context["parameter_ga"]
    emiten = context.get("emiten_terpilih", [])

    lines: List[str] = []
    add = lines.append
    add("=== DATA PORTOFOLIO HASIL ALGORITMA GENETIKA (GA) ===")
    add("Mode data: " + ("BACKTEST / SIMULASI HISTORIS" if str(context.get("mode")) == "backtest"
                         else "LIVE (data pasar terbaru)"))
    add("Tanggal acuan: " + str(context.get("tanggal_acuan") or "sekarang"))
    add("Profil risiko: " + str(context.get("profil_risiko")))
    add("Jumlah saham kandidat yang diseleksi GA: " + str(context.get("kandidat_saham_awal")))
    add("Jumlah saham terpilih (aktif): " + str(cons["n_saham_aktif"])
        + " (batas " + str(cons["min_saham"]) + "-" + str(cons["max_saham"]) + ")")
    add("")
    add("RUMUS FITNESS YANG DIPAKAI GA: " + str(context.get("rumus_fitness")))
    add("Koefisien: lambda_MDD = " + _num_str(koef["lambda_mdd"]) + " ; gamma_korelasi = "
        + _num_str(koef["gamma_korelasi"]) + " ; alpha_fundamental = "
        + _num_str(koef["alpha_fundamental"]) + " ; death_penalty = " + _num_str(koef["death_penalty"], 0))
    for kunci, catatan in (context.get("penjelasan_koefisien") or {}).items():
        add("  - " + str(catatan))
    add("")
    add("TABEL 1 - MODAL & ALOKASI")
    add("Budget pengguna: " + _rupiah(modal["budget"]))
    add("Dana terpakai: " + _rupiah(modal["terpakai"]) + " (" + _pct(modal["persen_terpakai"]) + " dari budget)")
    add("Sisa dana: " + _rupiah(modal["sisa"]))
    add("Budget terpenuhi: " + ("ya" if cons["budget_terpenuhi"] else "TIDAK"))
    add("")
    add("TABEL 2 - METRIK KINERJA PORTOFOLIO")
    add("Nilai fitness: " + _num_str(kin["fitness"]))
    add("Sharpe ratio (tahunan): " + _num_str(kin["sharpe_ratio"]))
    add("Return tahunan historis: " + _pct(kin["return_tahunan_historis"]))
    add("Estimasi return tahunan dari alokasi: " + _pct(kin["expected_return_tahunan"]))
    add("Volatilitas tahunan: " + _pct(kin["volatilitas_tahunan"]))
    add("Maximum drawdown (MDD): " + _pct(kin["max_drawdown"]))
    add("Rata-rata korelasi antar saham (absolut): " + _num_str(kin["rata_korelasi"]))
    add("Skor fundamental portofolio (0-1): " + _num_str(kin["skor_fundamental_0_1"]))
    add("Risk-free rate tahunan (BI): " + _pct(kin["risk_free_rate_tahunan"]))
    add("Excess return di atas risk-free: " + _pct(kin["excess_return_di_atas_risk_free"]))
    add("Jumlah hari data return yang dipakai: " + str(kin["jumlah_hari_return"]))
    add("")
    add("TABEL 3 - DEKOMPOSISI FITNESS (jumlah komponen = nilai fitness)")
    add("Sharpe ratio (positif): " + _num_str(komp["sharpe_ratio"]))
    add("Penalti MDD (lambda x |MDD|): " + _num_str(komp["penalti_max_drawdown"]))
    add("Penalti korelasi (gamma x rata2 korelasi): " + _num_str(komp["penalti_korelasi"]))
    add("Bonus fundamental (alpha x skor): " + _num_str(komp["bonus_fundamental"]))
    add("Penalti budget: " + _num_str(komp["penalti_budget"]))
    add("Penalti diversifikasi: " + _num_str(komp["penalti_diversifikasi"]))
    add("Jumlah komponen: " + _num_str(dek["jumlah_komponen"]) + " (fitness dilaporkan: "
        + _num_str(dek["fitness_dilaporkan"]) + ")")
    add("Komponen paling menentukan: " + str(dek.get("komponen_paling_menentukan")))
    add("")
    add("TABEL 4 - EMITEN PEMENANG: ALOKASI & BOBOT (urut alokasi terbesar)")
    add("| Ticker | Lot | Harga per lot (Rp) | Alokasi (Rp) | Bobot | Skor fundamental 0-1 |")
    add("|---|---|---|---|---|---|")
    for e in emiten:
        add("| " + str(e["ticker"]) + " | " + str(e["lots"]) + " | " + _rupiah(e["harga_per_lot"])
            + " | " + _rupiah(e["alokasi"]) + " | " + _pct(e["bobot"])
            + " | " + _cell(e["skor_fundamental_0_1"]) + " |")
    add("")
    add("TABEL 5 - EMITEN PEMENANG: FUNDAMENTAL & RISIKO HISTORIS")
    add("| Ticker | PER (x) | PBV (x) | ROE (%) | DER (rasio) | Div yield (%) | Return thn | "
        "Volatilitas thn | MDD | Korelasi abs rata2 | Kontribusi ke return portofolio |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in emiten:
        f = e["fundamental"]
        add("| " + str(e["ticker"]) + " | " + _cell(f["per_x"]) + " | " + _cell(f["pbv_x"])
            + " | " + _cell(f["roe_persen"]) + " | " + _cell(f["der_rasio"])
            + " | " + _cell(f["dividend_yield_persen"]) + " | " + _pct(e["return_tahunan"])
            + " | " + _pct(e["volatilitas_tahunan"]) + " | " + _pct(e["max_drawdown"])
            + " | " + _cell(e["korelasi_abs_rata2_dengan_emiten_lain"])
            + " | " + _pct(e["kontribusi_ke_return_portofolio"]) + " |")
    add("")
    add("TABEL 6 - KORELASI ANTAR EMITEN TERPILIH (1 = bergerak sama, 0 = bebas)")
    for label, daftar in (("Tertinggi", context["korelasi"]["pasangan_tertinggi"]),
                          ("Terendah", context["korelasi"]["pasangan_terendah"])):
        if not daftar:
            add("  Pasangan " + label.lower() + ": tidak tersedia (saham aktif < 2).")
            continue
        add("  Pasangan " + label.lower() + ":")
        for p in daftar:
            add("    " + " & ".join(p["pasangan"]) + " = " + _num_str(p["korelasi"]))
    add("")
    add("TABEL 7 - PARAMETER ALGORITMA GENETIKA")
    add("Ukuran populasi: " + str(ga["population_size"]) + " kromosom")
    add("Maksimum generasi: " + str(ga["generations"]) + " (dijalankan: "
        + str(ga["generations_dijalankan"] if ga["generations_dijalankan"] is not None else "sesuai hasil")
        + ")")
    add("Probabilitas crossover: " + _num_str(ga["crossover_rate"]))
    add("Ukuran turnamen seleksi: " + str(ga["tournament_size"]))
    add("Jumlah elit yang dipertahankan: " + str(ga["elitism_count"]))
    add("Mutation rate (awal -> akhir): " + _num_str(ga["mutation_rate_awal"]) + " -> "
        + _num_str(ga["mutation_rate_akhir"]) + " (adaptif per generasi)")
    add("Ambang early stop (generasi stagnan): " + str(ga["early_stop_patience"]))
    add("Seed acak: " + str(ga["seed"]))
    add("")
    add("CATATAN TERHITUNG (fakta dari sistem, bukan opini):")
    for i, catatan in enumerate(context.get("catatan_otomatis", []), 1):
        add("  " + str(i) + ". " + str(catatan))
    return "\n".join(lines)



# ----------------------------------------------------------------------
# Instruksi tugas (bagian "TUGAS ANDA" pada user prompt)
# ----------------------------------------------------------------------
# TASK_INSTRUCTIONS = (
#     "TUGAS ANDA:\n"
#     "Jelaskan hasil pemilihan portofolio oleh Algoritma Genetika di bawah ini kepada {audience}.\n"
#     "Tulis narasi Bahasa Indonesia, teks polos tanpa penanda Markdown (tanpa '#', '*', '-'), "
#     "panjang sekitar 350-550 kata, dengan urutan bagian berikut:\n"
#     "1) Ringkasan: profil risiko, modal, jumlah saham terpilih, dan nilai fitness beserta artinya.\n"
#     "2) Mengapa portofolio ini terpilih: uraikan kontribusi setiap komponen fitness (Sharpe, "
#     "penalti maximum drawdown, penalti korelasi, bonus fundamental) memakai angka dari TABEL 3, "
#     "lalu sebutkan komponen mana yang paling menentukan beserta alasannya. Jelaskan juga mengapa "
#     "batasan budget dan diversifikasi terpenuhi.\n"
#     "3) Peran setiap emiten pemenang: jelaskan alasan tiap saham masuk portofolio berdasarkan "
#     "bobot alokasi, skor fundamental, PER, PBV, ROE, DER, dividend yield, serta korelasinya dengan "
#     "saham lain. Bedakan saham dengan bobot terbesar dan bobot kecil, tanpa membuat tabel.\n"
#     "4) Risiko dan keterbatasan: bahas maximum drawdown, volatilitas, rata-rata korelasi, dana yang "
#     "belum terpakai, dan keterbatasan data (mis. hasil backtest memakai data historis).\n"
#     "5) Kesimpulan praktis: dua sampai tiga kalimat tentang karakter portofolio ini (mis. cenderung "
#     "konservatif atau agresif) dan hal yang perlu dipantau investor, ditutup dengan disclaimer "
#     "singkat bahwa ini hasil simulasi optimasi dan bukan nasihat keuangan.\n"
#     "Gunakan angka PERSIS seperti pada data (jangan mengubah atau membulatkan berbeda), tuliskan "
#     "satuannya, dan sebut nilai '-' sebagai 'tidak tersedia'. Jangan mengarang berita, rekomendasi "
#     "harga target, atau data lain di luar DATA PORTOFOLIO di bawah ini.\n"
# )
TASK_INSTRUCTIONS = (
    "TUGAS ANDA:\n"
    "Jelaskan mengapa saham-saham yang dihasilkan oleh Algoritma Genetika membentuk portofolio tersebut kepada {audience}.\n"
    "Fokus utama penjelasan harus berada pada INTERPRETASI FINANSIAL setiap saham dan peran masing-masing saham dalam portofolio, bukan pada penjelasan teknis mengenai mekanisme Algoritma Genetika.\n"
    "Algoritma Genetika digunakan sebagai metode optimasi untuk menentukan kombinasi dan bobot saham. Tugas Anda adalah menjelaskan secara finansial mengapa saham-saham tersebut masuk ke solusi akhir berdasarkan data yang tersedia.\n"
    "\n"
    "Tulis narasi Bahasa Indonesia yang jelas, mengalir, dan mudah dipahami oleh pembaca non-teknis. Gunakan teks polos tanpa penanda Markdown (tanpa '#', '*', atau '-' sebagai penanda daftar), dengan panjang sekitar 600-900 kata.\n"
    "\n"
    "Susun penjelasan dengan urutan berikut:\n"
    "\n"
    "1) GAMBARAN UMUM PORTOFOLIO\n"
    "Jelaskan secara singkat profil risiko, modal, jumlah saham terpilih, dan tujuan optimasi.\n"
    "Jelaskan komponen utama fitness function secara singkat, yaitu Sharpe Ratio, penalti Maximum Drawdown, penalti korelasi, dan bonus fundamental.\n"
    "Jangan menghabiskan sebagian besar narasi untuk menjelaskan mekanisme Algoritma Genetika seperti populasi, crossover, mutation, atau seleksi. Fokuskan penjelasan pada hasil pemilihan saham dan alasan finansial di baliknya.\n"
    "\n"
    "2) ANALISIS FINANSIAL SETIAP SAHAM TERPILIH\n"
    "Untuk SETIAP saham yang terpilih, jelaskan secara terpisah mengapa saham tersebut menjadi bagian dari portofolio.\n"
    "Untuk setiap saham WAJIB membahas sebanyak mungkin informasi yang tersedia mengenai:\n"
    "a. Bobot alokasi dan posisi saham dalam portofolio.\n"
    "b. PER dan PBV sebagai indikator valuasi.\n"
    "c. ROE sebagai indikator profitabilitas dan efisiensi penggunaan modal.\n"
    "d. DER sebagai indikator struktur utang dan leverage.\n"
    "e. Dividend Yield sebagai karakteristik distribusi keuntungan kepada pemegang saham.\n"
    "f. Skor fundamental saham dan hubungannya dengan metrik fundamental yang tersedia.\n"
    "g. Return tahunan historis sebagai gambaran kinerja historis.\n"
    "h. Volatilitas tahunan sebagai gambaran fluktuasi historis.\n"
    "i. Maximum Drawdown sebagai gambaran penurunan historis terbesar.\n"
    "j. Korelasi saham dengan saham lain dalam portofolio dan implikasinya terhadap diversifikasi.\n"
    "k. Kontribusi saham terhadap return portofolio berdasarkan bobot dan return historisnya.\n"
    "\n"
    "JANGAN hanya menyebutkan angka. Setiap angka yang relevan harus diikuti dengan interpretasi finansialnya.\n"
    "Jelaskan bagaimana kombinasi karakteristik fundamental, valuasi, return, risiko, dan korelasi tersebut menggambarkan PERAN saham tersebut di dalam portofolio.\n"
    "Bedakan antara saham yang memperoleh bobot besar dan saham yang memperoleh bobot kecil. Bobot yang lebih besar tidak otomatis berarti saham tersebut lebih baik; jelaskan berdasarkan data yang tersedia.\n"
    "\n"
    "Gunakan pola penalaran berikut sebagai panduan, tetapi jangan menyalinnya secara kaku:\n"
    "'[Ticker] memiliki bobot X% dalam portofolio. Dari sisi fundamental, ROE sebesar X% menunjukkan ..., sedangkan DER sebesar X menunjukkan .... Dari sisi valuasi, PER sebesar X dan PBV sebesar X memberikan informasi mengenai .... Dari sisi historis, return sebesar X% dengan volatilitas X% dan maximum drawdown X% menunjukkan .... Korelasi sebesar X dengan saham lain menunjukkan .... Kombinasi karakteristik tersebut membuat saham ini berperan sebagai ... dalam portofolio.'\n"
    "\n"
    "3) HUBUNGAN ANTAR SAHAM DAN DIVERSIFIKASI\n"
    "Jelaskan bagaimana saham-saham yang terpilih saling melengkapi berdasarkan korelasi.\n"
    "Identifikasi pasangan saham dengan korelasi tinggi dan rendah berdasarkan DATA PORTOFOLIO.\n"
    "Jelaskan bahwa korelasi yang lebih rendah dapat membantu mengurangi kecenderungan saham bergerak secara bersamaan, sedangkan korelasi yang lebih tinggi menunjukkan pergerakan yang lebih serupa.\n"
    "Hubungkan informasi tersebut dengan penalti korelasi dalam fitness function.\n"
    "Jangan menyimpulkan bahwa korelasi rendah selalu berarti saham tersebut lebih baik. Jelaskan hanya implikasinya terhadap diversifikasi portofolio.\n"
    "\n"
    "4) MENGAPA KOMBINASI PORTOFOLIO INI TERPILIH OLEH GA\n"
    "Setelah analisis setiap saham selesai, jelaskan bagaimana karakteristik saham-saham tersebut secara bersama-sama menghasilkan solusi portofolio yang dipilih.\n"
    "Hubungkan karakteristik saham dengan komponen fitness secara eksplisit:\n"
    "a. Sharpe Ratio menunjukkan hubungan antara excess return dan risiko portofolio.\n"
    "b. Penalti Maximum Drawdown mengurangi fitness ketika risiko penurunan maksimum semakin besar.\n"
    "c. Penalti korelasi mengurangi fitness ketika saham-saham dalam portofolio memiliki korelasi yang semakin tinggi.\n"
    "d. Bonus fundamental meningkatkan fitness berdasarkan skor fundamental portofolio.\n"
    "e. Constraint budget dan diversifikasi memastikan solusi memenuhi batasan yang ditentukan.\n"
    "\n"
    "PENTING: Jangan menjadikan nilai fitness sebagai satu-satunya alasan mengapa suatu saham dipilih. Jelaskan terlebih dahulu karakteristik finansial saham, kemudian tunjukkan bagaimana karakteristik tersebut berkontribusi terhadap objective function dan kombinasi portofolio secara keseluruhan.\n"
    "\n"
    "5) PERAN MASING-MASING SAHAM DALAM PORTOFOLIO\n"
    "Berikan rangkuman deskriptif mengenai peran setiap saham berdasarkan data yang tersedia.\n"
    "Misalnya, suatu saham dapat memiliki karakteristik return historis yang relatif tinggi, volatilitas yang relatif tinggi, karakteristik fundamental tertentu, atau korelasi yang membantu diversifikasi.\n"
    "Gunakan istilah seperti 'berperan sebagai komponen pertumbuhan', 'memberikan kontribusi terhadap diversifikasi', atau 'memiliki karakteristik defensif secara historis' HANYA jika didukung oleh data yang tersedia.\n"
    "Jangan menyebut suatu saham sebagai 'terbaik', 'terburuk', 'paling bagus', atau 'paling jelek'.\n"
    "\n"
    "6) RISIKO DAN KETERBATASAN\n"
    "Jelaskan risiko portofolio berdasarkan Maximum Drawdown, volatilitas, korelasi, dan karakteristik saham yang tersedia.\n"
    "Jelaskan juga dana yang belum terpakai jika terdapat sisa budget.\n"
    "Jika mode yang digunakan adalah BACKTEST, tegaskan bahwa return, volatilitas, dan Maximum Drawdown merupakan hasil historis dan tidak menjamin hasil masa depan.\n"
    "Sebutkan keterbatasan data apabila terdapat metrik yang tidak tersedia.\n"
    "\n"
    "7) KESIMPULAN\n"
    "Simpulkan karakteristik portofolio berdasarkan data yang tersedia dan jelaskan hal-hal utama yang perlu diperhatikan investor.\n"
    "Kesimpulan harus merangkum hubungan antara fundamental, return, risiko, korelasi, dan fitness function.\n"
    "Jangan memberikan rekomendasi beli atau jual, target harga, atau prediksi harga masa depan.\n"
    "Akhiri dengan disclaimer bahwa hasil ini merupakan simulasi optimasi berdasarkan data yang tersedia dan bukan nasihat keuangan.\n"
    "\n"
    "ATURAN PENTING:\n"
    "1. Gunakan HANYA angka dan fakta yang terdapat dalam DATA PORTOFOLIO.\n"
    "2. Jangan mengarang berita, kondisi perusahaan, prospek bisnis, target harga, atau informasi eksternal yang tidak terdapat dalam DATA PORTOFOLIO.\n"
    "3. Jangan mengatakan suatu saham pasti naik, pasti turun, atau pasti menghasilkan keuntungan.\n"
    "4. Jangan hanya mengulang angka. Setiap metrik harus dijelaskan makna finansialnya.\n"
    "5. Jangan menganggap skor fundamental tinggi sebagai satu-satunya alasan suatu saham terpilih.\n"
    "6. Jelaskan hubungan antara karakteristik finansial saham dan perannya dalam kombinasi portofolio.\n"
    "7. Bedakan fakta historis dari interpretasi. Jangan menyajikan interpretasi sebagai kepastian mengenai masa depan.\n"
    "8. Jangan membuat ranking atau penilaian subjektif terhadap saham.\n"
    "9. Gunakan angka PERSIS seperti pada DATA PORTOFOLIO dan jangan membulatkan dengan cara berbeda.\n"
    "10. Selalu tuliskan satuan yang sesuai, seperti x, %, atau rasio.\n"
    "11. Jika suatu nilai adalah '-', sebutkan sebagai 'tidak tersedia' dan jangan menebak nilainya.\n"
    "12. Jika informasi yang diperlukan untuk membuat suatu kesimpulan tidak tersedia, nyatakan keterbatasan tersebut.\n"
)


def build_explanation_prompt(
    context: Optional[Dict[str, Any]] = None,
    *,
    solution=None,
    market_data=None,
    config=None,
    allocations: Optional[Sequence[Dict[str, Any]]] = None,
    mode: str = "live",
    date_ref: Optional[Any] = None,
    budget: Optional[float] = None,
    expected_return: Optional[float] = None,
    generations_run: Optional[int] = None,
    audience: str = "investor ritel",
    extra_instruction: Optional[str] = None,
    system_instruction: Optional[str] = None,
) -> PortfolioPrompt:
    """
    Bangun prompt lengkap (system instruction + user prompt) untuk meminta LLM
    menjelaskan hasil GA.

    Bisa dipakai dengan dua cara:
        1. Langsung dari objek GA:
           build_explanation_prompt(solution=sol, market_data=md, config=cfg)
        2. Dari konteks yang sudah dirakit (mis. untuk logging/uji):
           build_explanation_prompt(context=ctx)

    Args:
        context     : hasil build_portfolio_context (opsional).
        solution    : Chromosome terbaik (wajib bila `context` kosong).
        market_data : MarketData yang dipakai GA (wajib bila `context` kosong).
        config      : GAConfig (opsional, disarankan diisi).
        lainnya     : diteruskan ke build_portfolio_context.
        audience    : sasaran pembaca (mis. "investor ritel", "manajer investasi").
        extra_instruction : instruksi tambahan yang ditempel di akhir tugas.
        system_instruction: override system instruction (default teks baku modul).

    Raises:
        ValueError: bila `context` kosong dan `solution`/`market_data` tidak diberi.
    """
    if context is None:
        if solution is None or market_data is None:
            raise ValueError(
                "build_explanation_prompt memerlukan 'context', atau pasangan "
                "'solution' + 'market_data' untuk merakit konteks."
            )
        context = build_portfolio_context(
            solution,
            market_data,
            config,
            allocations=allocations,
            mode=mode,
            date_ref=date_ref,
            budget=budget,
            expected_return=expected_return,
            generations_run=generations_run,
        )

    tugas = TASK_INSTRUCTIONS.replace("{audience}", str(audience))
    if extra_instruction:
        tugas = tugas + "INSTRUKSI TAMBAHAN: " + str(extra_instruction) + "\n"

    user_prompt = tugas + "\n" + render_context_text(context)
    return PortfolioPrompt(
        system_instruction=system_instruction or DEFAULT_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        context=context,
    )


def build_narrative_prompt(*args, **kwargs) -> PortfolioPrompt:
    """
    Alias praktis :func:`build_explanation_prompt` (dipakai
    :mod:`src.backend.services.gemini_llm_caller` agar pemanggilan service
    tetap ringkas).
    """
    return build_explanation_prompt(*args, **kwargs)


__all__ = [
    "PortfolioPrompt",
    "FUNDAMENTAL_LABELS",
    "FITNESS_FORMULA_TEXT",
    "COEFFICIENT_NOTES",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "TASK_INSTRUCTIONS",
    "build_portfolio_context",
    "render_context_text",
    "build_explanation_prompt",
    "build_narrative_prompt",
]

