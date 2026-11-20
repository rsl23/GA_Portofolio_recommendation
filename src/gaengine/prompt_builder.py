"""
Prompt Builder - Bahan Penjelasan Portofolio untuk LLM
=====================================================

Modul ini merakit BAHAN (konteks terstruktur) + PROMPT yang dikirim ke LLM
(OpenAI, lihat :mod:`src.backend.services.openai_llm_caller`) supaya hasil
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
                                      oleh openai_llm_caller)

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

from .narrative_evidence import build_selection_evidence, risk_notes
from .narrative_facts import mark_numeric_facts
from .news_context import normalize_news_context
from .narrative_style import NARRATIVE_VERSION, fundamental_contrast, narrative_opening

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
    "Fitness = SharpePenilaian - lambda*KomponenRisiko - PenaltiBuktiHilang "
    "- gamma*RataKorelasi + alpha*SkorFundamental - PenaltiTurnover "
    "+ BonusDividen + BonusMomentum + UtilityRebalance + RewardReturnNAV "
    "- DeathPenalty_Budget - DeathPenalty_Diversifikasi. "
    "KomponenRisiko menggunakan campuran MDD residual dan risiko beberapa event IHSG "
    "bila tersedia; tanpa bukti event, memakai MDD penuh."
)

# Peran tiap koefisien, supaya LLM menjelaskan bobot penalti dengan tepat.
COEFFICIENT_NOTES = {
    "lambda_mdd": "lambda = bobot penalti risiko historis (MDD residual dan stress jika "
                  "model market_stress aktif; ditentukan profil risiko).",
    "gamma_korelasi": "gamma = bobot penalti korelasi rata-rata antar saham "
                      "(makin besar = makin menekan risiko portofolio bergerak seragam).",
    "alpha_fundamental": "alpha = bobot bonus skor fundamental (PER/PBV/ROE/DER/DivYield "
                         "yang sudah dinormalisasi min-max menjadi skor 0-1).",
    "death_penalty": "DeathPenalty = nilai penalti besar ketika alokasi melebihi budget "
                     "atau jumlah saham aktif di luar batas minimum/maksimum.",
}

DEFAULT_SYSTEM_INSTRUCTION = (
    "Anda adalah penulis analisis hasil optimasi portofolio untuk investor ritel Indonesia. "
    "Bantu pembaca memahami hasil, dengan bahasa tenang, alami, dan mudah diikuti. "
    "Mulai dari satu kesimpulan bermakna tanpa angka, lalu jelaskan maknanya dan bukti terpilih. "
    "Gunakan hubungan antarfakta, bukan daftar rasio atau definisi buku teks.\n"
    "FOKUS: fundamental saham, kompromi risikonya, dan metrik historis GA. "
    "Hasil GA sudah tetap; Anda tidak mengubah saham, alokasi, atau keputusan transaksi. "
    "Rincian bobot, kas, biaya, dan lot tersedia terpisah. Berita juga tampil sebagai konteks "
    "terpisah dan tidak dimasukkan ke narasi utama.\n"
    "FAKTA: hanya gunakan data yang diberikan. Setiap angka output harus menggunakan "
    "penanda [[N001]] dan seterusnya. Pertahankan satuan; tambahkan x hanya jika belum "
    "tercakup dalam nilai penanda. Jangan menghitung nilai baru, mengarang data emiten, "
    "menebak sektor, atau menganggap data hilang sebagai nol. Hubungkan nilai dengan "
    "saham dan metrik pemiliknya secara tepat.\n"
    "INTERPRETASI: bandingkan rasio secara relatif, bukan memberi label murah atau aman "
    "tanpa acuan. PER negatif bukan valuasi murah. Yield nol tidak menjelaskan kebijakan "
    "pembayaran dividen. Korelasi saja tidak membuktikan risiko turun. MDD adalah pengalaman "
    "historis, bukan batas kerugian mendatang. Hasil pencarian bukan jaminan kelayakan "
    "atau optimum global. Aturan ini menjaga interpretasi, bukan daftar peringatan yang "
    "harus dibacakan. Batasi kalimat pembatas pada satu di bagian fundamental dan satu "
    "di bagian risiko yang paling relevan. Pembuka tidak berisi disclaimer.\n"
    "BENTUK: tiga atau empat paragraf pendek bila bahan cukup; dua sampai lima diperbolehkan. "
    "Panjang mengikuti kebutuhan penjelasan; sekitar 300–450 kata biasanya cukup, bukan target wajib. Hindari melebihi 600 kata. Pilih enam sampai delapan angka pendukung, "
    "maksimum dua belas. Kalimat pertama tanpa angka atau penanda. Tanpa heading, daftar, "
    "tautan, kode policy, atau rincian fitness internal. Tutup dengan satu kalimat: "
    "Hasil optimasi tidak menjamin kinerja mendatang dan bukan nasihat keuangan."
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
    numeric_facts: Dict[str, str] = field(default_factory=dict)
    news_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Bentuk dict (memudahkan logging / penyimpanan / pengujian)."""
        return {
            "system_instruction": self.system_instruction,
            "user_prompt": self.user_prompt,
            "context": self.context,
            "numeric_facts": self.numeric_facts,
            "news_sources": self.news_sources,
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
    eq = np.r_[1.0, np.cumprod(1.0 + daily)]
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

    sharpe = float(getattr(solution, "score_sharpe", getattr(solution, "sharpe_ratio", 0)) or 0)
    risk_component = getattr(solution, "risk_component", None)
    mdd = float(risk_component if risk_component is not None else (getattr(solution, "max_drawdown", 0) or 0))
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
        "bonus_dividen": _num(getattr(solution, "dividend_bonus", 0.0)),
        "penalti_budget": _num(penalti_budget),
        "penalti_diversifikasi": _num(penalti_div),
        "penalti_bukti_hilang": _num(-getattr(solution, "market_stress_missing_penalty", 0.0)),
        "penalti_turnover": _num(-getattr(config, "turnover_penalty_beta", 0.0) * getattr(solution, "turnover_rate", 0.0)),
        "bonus_momentum": _num(getattr(config, "momentum_bonus", 0.0) * getattr(solution, "momentum_score", 0.0)),
        "utility_rebalance": _num(getattr(config, "rebalance_utility_weight", 0.0) * getattr(solution, "rebalance_utility", 0.0)),
        "reward_return_nav": _num(getattr(config, "cash_return_weight", 0.0) * getattr(solution, "roi_annual", 0.0)
                                 if getattr(config, "allow_cash", False) and config.fitness_mode != "baseline" else 0.0),
        "penalti_cvar": _num(-(getattr(solution, "fitness_diagnostics", None) or {}).get("cvar_penalty", 0.0)),
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
            "bobot_dividen": _num(getattr(config, "dividend_bonus_weight", 0.0)),
            "batas_yield_dividen": _num(getattr(config, "dividend_yield_cap", 0.10)),
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

    Jika tersedia, payload allocations dari controller adalah sumber lot dan
    harga eksekusi aktual. Pemanggilan mandiri memakai lot/harga keputusan GA.

    Returns:
        (rows, lots, weights, active_indices)
    """
    codes = list(market_data.stock_codes)
    prices = np.asarray(market_data.prices_per_lot, dtype=float).copy()
    returns = np.asarray(market_data.returns, dtype=float)
    corr = np.asarray(market_data.correlation, dtype=float)
    fund_scores = np.asarray(market_data.fundamental_scores, dtype=float)
    fund_metrics = getattr(market_data, "fundamental_metrics", None)
    n = len(codes)

    lots_raw = getattr(solution, "lots", None)
    if allocations is not None:
        # Controller rows describe actual fills, including backtest slippage.
        lots = np.zeros(n, dtype=int)
        pos = {c: i for i, c in enumerate(codes)}
        for row in allocations:
            if row.get("ticker") in pos:
                i = pos[row["ticker"]]
                lots[i] = int(row.get("lots", 0) or 0)
                if row.get("price_per_lot") is not None:
                    prices[i] = float(row["price_per_lot"])
    elif lots_raw is not None and len(lots_raw) == n:
        lots = np.asarray(lots_raw, dtype=int)
    else:
        lots = np.zeros(n, dtype=int)

    weights = _weights_from_lots(lots, prices)
    active = [i for i in range(n) if int(lots[i]) > 0]
    annual_returns = np.array([
        float(r[np.isfinite(r)].mean() * 252) if np.isfinite(r).any() else np.nan
        for r in returns
    ])
    basket_return = float(np.dot(weights, np.nan_to_num(annual_returns)))

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
        dividend_yields = getattr(market_data, "dividend_yields", None)
        if dividend_yields is not None:
            div_val = _num(dividend_yields[i] * 100.0, 4)
        contribution = float(weights[i] * annual_returns[i])
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
            "kontribusi_return_keranjang_poin_persen": _num(contribution * 100.0, 6),
            "bagian_return_keranjang": _num(contribution / basket_return, 6) if basket_return > 1e-12 else None,
            "sumber_dividend_yield": "arsip dividen TTM / harga keputusan" if dividend_yields is not None else "cache fundamental",
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
    rebalance_deltas: Optional[List[Dict[str, Any]]] = None,
    reporting: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rakit KONTEKS lengkap hasil GA menjadi dict siap-JSON untuk dijelaskan LLM.

    Angka dihitung di Python agar LLM tidak perlu menghitung ulang. Penanda
    fakta dan pemeriksaan keluaran membatasi kesalahan angka pada narasi.

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
    reporting = reporting or {}
    rows, lots, weights, active = _emiten_rows(solution, market_data, allocations)
    known_investment = float(sum(row["alokasi"] for row in rows))
    total_terpakai = float(reporting.get("investment_total", known_investment))
    total_budget = _num(budget, 2)
    if total_budget is None:
        total_budget = _num(getattr(config, "budget", None), 2) or total_terpakai
    fees = reporting.get("transaction_costs")
    sisa = reporting.get("cash_after")
    if sisa is None and fees is not None:
        sisa = total_budget - total_terpakai - float(fees)
    uncovered = list(reporting.get("uncovered_positions") or [])
    for row in rows:
        row["bobot_dana_saham"] = _num(row["alokasi"] / total_terpakai, 6) if total_terpakai else None
        row["bobot_modal"] = _num(row["alokasi"] / total_budget, 6) if total_budget else None
    daily_basket = _daily_series(np.asarray(market_data.returns, dtype=float), weights)
    basket_stats = _annual_stats(daily_basket)
    nav_weights = weights * (known_investment / total_budget) if total_budget else np.zeros_like(weights)
    daily = _daily_series(np.asarray(market_data.returns, dtype=float), nav_weights)
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
    sigma = float(daily.std() * math.sqrt(252)) if daily.size else 0.0
    sharpe_nav = ((float(daily.mean()) * 252 - risk_free) / sigma
                  if daily.size and sigma > 1e-12 and risk_free is not None else None)
    if uncovered:
        port_stats = dict.fromkeys(port_stats)
        basket_stats = dict.fromkeys(basket_stats)
        sharpe_nav = excess = None
        for row in rows:
            row["bagian_return_keranjang"] = None
            row["kontribusi_return_keranjang_poin_persen"] = None
    execution_prices = reporting.get("execution_prices") or {}
    decision_prices = dict(zip(market_data.stock_codes, np.asarray(market_data.prices_per_lot) / 100))
    different_prices = any(code in decision_prices and not math.isclose(
        float(value), float(decision_prices[code]), rel_tol=1e-9, abs_tol=1e-6)
        for code, value in execution_prices.items())
    comparison_reason = ("Posisi di luar data keputusan belum tercakup: " + ", ".join(uncovered)) if uncovered else None
    if different_prices:
        comparison_reason = "Harga eksekusi berbeda dari harga keputusan; perbandingan skor lintas basis tidak ditampilkan."

    context: Dict[str, Any] = {
        "mode": str(mode).lower(),
        "tanggal_acuan": _to_iso(date_ref),
        "profil_risiko": getattr(config, "risk_profile", None),
        "jenis_keputusan": "rebalance" if getattr(config, "is_rebalance", False) else "pembentukan_awal",
        "tanggal_harga_keputusan": _to_iso(getattr(market_data, "price_date", None)),
        "basis_metrik": "Simulasi historis bobot komposisi tetap terhadap modal acuan; kas ber-return nol, biaya transaksi dilaporkan terpisah. Bukan kinerja akun aktual.",
        "posisi_belum_tercakup": uncovered,
        "rumus_fitness": FITNESS_FORMULA_TEXT,
        "penjelasan_koefisien": dict(COEFFICIENT_NOTES),
        "modal": {
            "budget": total_budget,
            "terpakai": _num(total_terpakai, 2),
            "sisa": _num(sisa, 2),
            "sisa_sebelum_biaya": _num(total_budget - total_terpakai, 2),
            "biaya_transaksi": _num(fees, 2),
            "kas_setelah_biaya_terverifikasi": sisa is not None and fees is not None,
            "batas_exposure": _num(reporting.get("exposure_limit"), 6),
            "basis_budget": "NAV sebelum transaksi termasuk setoran yang diterapkan" if getattr(config, "is_rebalance", False) else "modal pembentukan awal",
            "persen_terpakai": _num(total_terpakai / total_budget, 6) if total_budget else None,
        },
        "kinerja_portofolio": {
            "fitness": _num(getattr(solution, "fitness", None)),
            "sharpe_ratio": _num(sharpe_nav, 6),
            "return_tahunan_historis": port_stats["return_tahunan"],
            "expected_return_tahunan": _num(expected_return, 6),
            "volatilitas_tahunan": port_stats["volatilitas_tahunan"],
            "max_drawdown": port_stats["max_drawdown"],
            "max_drawdown_dari_seri_harga": port_stats["max_drawdown"],
            "rata_korelasi": _num(getattr(solution, "avg_correlation", None)),
            "skor_fundamental_0_1": _num(getattr(solution, "skor_fundamental", None)),
            "risk_free_rate_tahunan": risk_free,
            "excess_return_di_atas_risk_free": excess,
            "jumlah_hari_return": int(daily.size) if daily.size else 0,
        },
        "dekomposisi_fitness": breakdown,
        "kinerja_keranjang_saham": basket_stats,
        "bukti_pembanding": build_selection_evidence(solution, config, unavailable_reason=comparison_reason),
        "batasan_risiko": risk_notes(solution, config),
        "uji_cvar": dict(getattr(solution, "fitness_diagnostics", None) or {}),
        "constraint": {
            "min_saham": getattr(config, "min_stocks", None),
            "max_saham": getattr(config, "max_stocks", None),
            "n_saham_aktif": int(getattr(solution, "n_active", len(active))),
            "budget_terpenuhi": bool(getattr(solution, "budget_ok", True)),
            "diversifikasi_terpenuhi": bool(getattr(solution, "diversification_ok", True)),
            "penalti_risiko_profil": _num(getattr(config, "lambda_mdd", None)),
            "kebijakan_profil": getattr(config, "profile_policy", "legacy"),
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
    if rebalance_deltas is not None:
        context["rebalance_deltas"] = rebalance_deltas
    evidence = getattr(market_data, "market_stress", None)
    if evidence is not None:
        context["uji_stress"] = {
            "as_of": evidence.as_of.isoformat(), "history_end": str(evidence.history_end),
            "lookback_years": evidence.lookback_years,
            "window_start": evidence.window_start.isoformat(),
            "window_end": evidence.as_of.isoformat(),
            "status": evidence.status, "reason": evidence.reason, "events": evidence.events,
            "risk_model": config.risk_model,
            "mdd_residual_weight": config.mdd_residual_weight,
            "stress_risk": getattr(solution, "market_stress_risk", None),
            "coverage": getattr(solution, "market_stress_coverage", None),
            "uncertainty_penalty": getattr(solution, "market_stress_missing_penalty", 0.0),
        }
        
    context["catatan_otomatis"] = _auto_insights(context)
    return context



def _auto_insights(context: Dict[str, Any]) -> List[str]:
    """Explicit interpretation boundaries and already-computed facts."""
    notes = [
        "Sharpe portofolio mengukur return relatif terhadap fluktuasi terhadap seluruh modal. Komponen Sharpe penilaian hanya bagian rumus skor; jangan menyebut keduanya sebagai metrik yang sama.",
        "Return yang disediakan adalah rerata return harian dikali 252, bukan return kumulatif atau ramalan. Volatilitas saham, keranjang saham, dan seluruh modal mempunyai penyebut berbeda.",
        "Kontribusi return dalam poin persentase bukan persentase bagian total return. Gunakan bagian_return_keranjang bila ingin menyebut proporsi kontribusi; bila tidak tersedia jangan menebaknya.",
        "Korelasi dekat nol menunjukkan hubungan linear lemah dalam sampel; tidak membuktikan independensi atau perlindungan saat semua saham turun.",
    ]
    if context.get("jenis_keputusan") == "pembentukan_awal":
        notes.append("Ini pembentukan awal. Jangan menyatakan mempertahankan saham lama atau menyimpulkan pembelian terjadi karena penalti turnover nol.")
    else:
        notes.append("Ini rebalance. Jelaskan BUY/SELL/HOLD dari delta aktual; laba/rugi akuntansi bukan sebab otomatis transaksi.")
    if context.get("constraint", {}).get("kebijakan_profil") == "legacy":
        notes.append("Profil risiko mengatur bobot penalti risiko; label profil tidak otomatis menjadi batas keras volatilitas setiap saham.")
    modal = context["modal"]
    if modal.get("batas_exposure") is not None:
        notes.append("Batas maksimum dana saham yang aktif: " + _pct(modal["batas_exposure"]) + " dari modal. Pembulatan lot dan biaya memengaruhi realisasi; jangan mengarang sebab pasar dari batas ini.")
    else:
        notes.append("Sumber alasan pemakaian kas tidak diberikan sebagai batas exposure aktif. Jangan mengklaim sisa kas disengaja atau berasal dari bear market hanya dari jumlahnya.")
    if not modal.get("kas_setelah_biaya_terverifikasi"):
        notes.append("Kas setelah biaya belum terverifikasi; hanya kas sebelum biaya tersedia. Jangan menyebut angka sebelum biaya sebagai kas final.")
    if context.get("posisi_belum_tercakup"):
        notes.append("Metrik tidak mencakup seluruh posisi; jangan menyatakan ukuran risiko ini mewakili portofolio lengkap.")
    notes.extend(context.get("batasan_risiko", []))
    return notes



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
    """Render units and bases explicitly; never ask the LLM to calculate them."""
    import json

    kin = context["kinerja_portofolio"]
    modal = context["modal"]
    breakdown = context["dekomposisi_fitness"]
    basket = context.get("kinerja_keranjang_saham", {})
    cons = context["constraint"]
    lines = [
        "=== DATA PORTOFOLIO HASIL GA ===",
        "Mode: " + ("BACKTEST / SIMULASI HISTORIS" if context.get("mode") == "backtest" else "LIVE"),
        "Jenis keputusan: " + str(context.get("jenis_keputusan")),
        "Tanggal acuan: " + str(context.get("tanggal_acuan") or "tidak diberikan"),
        "Tanggal harga keputusan: " + str(context.get("tanggal_harga_keputusan")),
        "Profil risiko: " + str(context.get("profil_risiko")),
        "Kandidat awal: " + str(context.get("kandidat_saham_awal")),
        "Saham aktif: " + str(cons["n_saham_aktif"]) + "; batas jumlah " + str(cons["min_saham"]) + "–" + str(cons["max_saham"]),
        "Jumlah saham dalam batas tidak membuktikan risiko rendah.",
        "BASIS: " + str(context.get("basis_metrik")),
        "",
        "MODAL DAN TRANSAKSI",
        "Modal acuan (" + str(modal.get("basis_budget")) + "): " + _rupiah(modal["budget"]),
        "Nilai saham pada harga eksekusi: " + _rupiah(modal["terpakai"]) + " = " + _pct(modal["persen_terpakai"]) + " modal.",
        "Biaya transaksi aktual: " + _rupiah(modal.get("biaya_transaksi")),
        "Kas setelah biaya: " + _rupiah(modal.get("sisa")),
        "Kas sebelum biaya (jangan disebut kas final): " + _rupiah(modal.get("sisa_sebelum_biaya")),
        "Batas maksimum exposure aktif: " + _pct(modal.get("batas_exposure")),
        "",
        "METRIK HISTORIS SELURUH MODAL TERMASUK KAS",
        "Sharpe portofolio (metrik untuk investor): " + _num_str(kin["sharpe_ratio"]),
        "Return aritmetika tahunan historis: " + _pct(kin["return_tahunan_historis"]),
        "Volatilitas tahunan seluruh modal: " + _pct(kin["volatilitas_tahunan"]),
        "MDD seluruh modal (penurunan maksimum dari puncak historis): " + _pct(kin["max_drawdown"]),
        "Risk-free tahunan: " + _pct(kin["risk_free_rate_tahunan"]),
        "Jumlah observasi return: " + str(kin["jumlah_hari_return"]),
        "",
        "METRIK KERANJANG SAHAM SAJA (100% DANA SAHAM, TANPA KAS)",
        "Return aritmetika tahunan keranjang: " + _pct(basket.get("return_tahunan")),
        "Volatilitas tahunan keranjang: " + _pct(basket.get("volatilitas_tahunan")),
        "MDD keranjang saham: " + _pct(basket.get("max_drawdown")),
        "",
        "PENILAIAN GA PADA HARGA KEPUTUSAN (RINCIAN, BUKAN METRIK INVESTOR TAMBAHAN)",
        "Fitness hasil: " + _num_str(kin["fitness"]),
        "Komponen Sharpe penilaian, bukan Sharpe portofolio: " + _num_str(breakdown["komponen"]["sharpe_ratio"]),
        "Rumus: " + str(context.get("rumus_fitness")),
        "Komponen bertanda, termasuk penalti CVaR bila aktif: " + json.dumps(breakdown["komponen"], ensure_ascii=False),
        "Koefisien: " + json.dumps(breakdown["koefisien"], ensure_ascii=False),
        "Korelasi absolut rata-rata: " + _num_str(kin["rata_korelasi"]),
        "",
        "SAHAM TERPILIH (urut nilai alokasi aktual)",
        "Bobot dana saham memakai total nilai saham sebagai penyebut. Bobot modal memakai seluruh modal acuan.",
        "Kontribusi di bawah adalah terhadap return aritmetika KERANJANG SAHAM; satuan poin persentase berbeda dari bagian total (%).",
    ]
    for row in context.get("emiten_terpilih", []):
        f = row["fundamental"]
        lines.append(
            str(row["ticker"]) + ": " + str(row["lots"]) + " lot; alokasi " + _rupiah(row["alokasi"])
            + "; bobot dana saham " + _pct(row.get("bobot_dana_saham"))
            + "; bobot modal " + _pct(row.get("bobot_modal"))
            + "; skor fundamental " + _num_str(row.get("skor_fundamental_0_1"))
            + "; dividend yield " + _cell(f.get("dividend_yield_persen")) + "% (" + row.get("sumber_dividend_yield", "tidak tersedia") + ")"
            + "; DER " + _cell(f.get("der_rasio"))
            + "; return aritmetika tahunan saham " + _pct(row.get("return_tahunan"))
            + "; volatilitas tahunan saham " + _pct(row.get("volatilitas_tahunan"))
            + "; MDD saham " + _pct(row.get("max_drawdown"))
            + "; kontribusi keranjang " + _num_str(row.get("kontribusi_return_keranjang_poin_persen")) + " poin persentase"
            + "; bagian total return keranjang " + _pct(row.get("bagian_return_keranjang"))
        )
    lines.append("\nKORELASI PASANGAN HISTORIS (0 = hubungan linear lemah, bukan bukti independen)")
    for label in ("pasangan_tertinggi", "pasangan_terendah"):
        for pair in context["korelasi"][label]:
            lines.append(" & ".join(pair["pasangan"]) + ": " + _num_str(pair["korelasi"]))
    proof = context.get("bukti_pembanding", {})
    lines.extend(["\nBUKTI PEMBANDING LOKAL", proof.get("batas_interpretasi", "")])
    if proof.get("status") != "tersedia":
        lines.append("Pembanding tidak tersedia: " + str(proof.get("alasan", "belum dihitung")))
    else:
        alternatives = [("Alokasi merata pada saham yang sama", proof.get("bobot_merata") or {})]
        alternatives.extend(("Tanpa " + item["ticker"] + ", dana menjadi kas", item)
                            for item in proof.get("tanpa_satu_saham", []))
        for label, item in alternatives:
            if not item.get("layak"):
                lines.append(label + ": TIDAK LAYAK; " + str(item.get("alasan")) + ". Jangan dipakai sebagai bukti keunggulan hasil.")
                continue
            lines.append(label + ": fitness " + _num_str(item.get("fitness"))
                + "; selisih fitness hasil dikurangi pembanding " + _num_str(item.get("selisih_fitness_hasil_dikurangi_pembanding"))
                + "; Sharpe portofolio pembanding " + _num_str(item.get("sharpe_portofolio"))
                + "; volatilitas modal pembanding " + _pct(item.get("volatilitas_modal_tahunan"))
                + "; return aritmetika modal pembanding " + _pct(item.get("return_aritmetika_modal_tahunan"))
                + "; MDD modal pembanding " + _pct(item.get("mdd_modal")))
    progress = context.get("parameter_ga", {})
    lines.append("Pencarian berjalan " + str(progress.get("generations_dijalankan")) + " generasi; hanya solusi terbaik yang ditemukan, tanpa jaminan optimum global.")
    if "rebalance_deltas" in context:
        lines.extend(["TABEL 8 - REBALANCING (delta transaksi aktual)", json.dumps(context["rebalance_deltas"], ensure_ascii=False, default=str)])
        lines.append("Realized P/L adalah akibat akuntansi, bukan alasan otomatis menjual.")
    if context.get("uji_cvar"):
        cvar = context["uji_cvar"]
        lines.extend(["\nUJI SKENARIO KERUGIAN HISTORIS",
            "Jumlah skenario: " + str(cvar.get("scenario_count"))
            + "; panjang setiap skenario: " + str(cvar.get("horizon_days")) + " sesi"
            + "; bagian ekor yang dirata-ratakan: " + _pct(cvar.get("tail_fraction"))
            + "; rata-rata kerugian downside pada ekor skenario: " + _pct(cvar.get("downside_cvar")),
            "Gunakan batasan pengujian dalam bahasa investor, tanpa kode policy atau rincian skor teknis."])
    if context.get("uji_stress"):
        stress = context["uji_stress"]
        lines.extend(["\nUJI TEKANAN PASAR HISTORIS",
            "Tanggal penilaian: " + str(stress.get("as_of")) + "; arsip terakhir: " + str(stress.get("history_end")),
            "Jumlah event IHSG: " + str(len(stress.get("events", [])))
            + "; cakupan bukti saham: " + _pct(stress.get("coverage")),
            "Return event IHSG berikut bukan kerugian portofolio atau saham. Risiko gabungan berbeda dari MDD saja."])
        for event in stress.get("events", []):
            lines.append("IHSG " + str(event.get("start")) + " sampai " + str(event.get("end"))
                         + ": " + _pct(event.get("market_return")))
    lines.append("\nBATAS INTERPRETASI DAN FAKTA PENTING")
    lines.extend(context.get("catatan_otomatis", []))
    return "\n".join(lines)



# ----------------------------------------------------------------------
# Instruksi tugas (bagian "TUGAS ANDA" pada user prompt)
# ----------------------------------------------------------------------
def render_investor_context(context: Dict[str, Any]) -> str:
    """Meaning-first fundamentals and observed GA metrics; news stays in the UI."""
    kin = context["kinerja_portofolio"]
    rows = context.get("emiten_terpilih", [])
    lines = [
        "DATA UNTUK NARASI UTAMA — FUNDAMENTAL DAN METRIK GA",
        "Mode: " + ("BACKTEST / SIMULASI HISTORIS" if context.get("mode") == "backtest" else "LIVE"),
        "Profil: " + str(context.get("profil_risiko")) + "; jenis: " + str(context.get("jenis_keputusan")),
        ("Konteks waktu: jelaskan dari sudut pandang tanggal keputusan simulasi; jangan memakai kejadian sesudahnya."
         if context.get("mode") == "backtest" else "Konteks waktu: ini rekomendasi live berbasis snapshot yang tersedia; metrik risiko tetap historis, bukan data real-time atau prediksi."),
        "Tanggal acuan: " + str(context.get("tanggal_acuan") or context.get("tanggal_harga_keputusan") or "tidak tersedia"),
        "Saham terpilih: " + ", ".join(row["ticker"] for row in rows),
        "METRIK PORTOFOLIO: simulasi historis seluruh modal, bukan prediksi atau kinerja akun aktual.",
        "Sharpe portofolio: " + _num_str(kin.get("sharpe_ratio")),
        "Penurunan terbesar historis (MDD): " + _pct(kin.get("max_drawdown")),
        "Korelasi absolut rata-rata antar saham: " + _num_str(kin.get("rata_korelasi")),
    ]
    breakdown = context.get("dekomposisi_fitness") or {}
    components = breakdown.get("komponen") or {}
    labels = {"bonus_fundamental": "fundamental", "bonus_dividen": "dividen", "bonus_momentum": "momentum"}
    additions = [label for key, label in labels.items() if (components.get(key) or 0) > 0]
    if additions:
        lines.append("Kriteria tambahan yang aktif pada penilaian GA: " + ", ".join(additions) + ".")
    lines.append("Rincian fitness, koefisien dan Sharpe penilaian tersedia dalam diagnostik; "
                 "jangan membacakan skor internal atau mencampurnya dengan Sharpe portofolio.")
    contrast = fundamental_contrast(context)
    lines.append("ARAH PENJELASAN BERDASARKAN DATA (bukan klaim sebab pemilihan): " + narrative_opening(context, contrast))
    if contrast:
        for observation in contrast["observations"]:
            lines.append("Perbandingan terverifikasi: " + observation["label"] + " " + contrast["first"]["ticker"]
                         + " " + observation["relation"] + " daripada " + contrast["second"]["ticker"] + ".")
    lines.append("PEDOMAN INTERPRETASI (jelaskan yang dipakai, bukan seluruh kamus): "
                 "PER membandingkan harga dengan laba per saham; lebih rendah berarti harga relatif terhadap laba lebih rendah, "
                 "belum otomatis murah. PBV membandingkan harga dengan nilai buku ekuitas per saham; lebih rendah berarti "
                 "harga relatif terhadap nilai buku lebih rendah, bukan jaminan murah. "
                 "ROE menggambarkan laba pada periode data terhadap ekuitas/modal sendiri; lebih tinggi umumnya "
                 "menunjukkan profitabilitas terhadap ekuitas lebih tinggi, bukan lebih rendah lebih baik. "
                 "ROE tinggi dapat dipengaruhi leverage atau ekuitas kecil sehingga perlu dibaca bersama DER. "
                 "DER membandingkan utang dengan ekuitas; lebih rendah berarti ketergantungan relatif pada utang lebih rendah, "
                 "bukan bukti seluruh risiko saham lebih rendah. Dividend yield membandingkan dividen tercatat dengan harga; "
                 "lebih tinggi memberi imbal hasil dividen historis relatif lebih besar, bukan jaminan pembayaran mendatang. "
                 "Sharpe mengukur imbal hasil lebih dari tingkat bebas risiko relatif terhadap fluktuasi; lebih tinggi "
                 "umumnya lebih baik pada basis dan periode sebanding. MDD adalah penurunan terdalam dari puncak selama "
                 "simulasi; nilai penurunan lebih kecil berarti pengalaman penurunan historis lebih ringan. "
                 "Jangan menganggap periode ROE tahunan jika periodenya tidak diberikan. Jangan mengubah skala ROE sendiri.")
    lines.append("FUNDAMENTAL SAHAM TERPILIH:")
    ranked = sorted(rows, key=lambda row: (row.get("skor_fundamental_0_1") is not None,
                                          row.get("skor_fundamental_0_1") or 0), reverse=True)
    for row in ranked:
        f = row.get("fundamental") or {}
        lines.append(row["ticker"] + ": PER " + _cell(f.get("per_x"), 2) + " x; PBV " + _cell(f.get("pbv_x"), 2)
                     + " x; ROE " + _cell(f.get("roe_persen"), 2) + "%; DER " + _cell(f.get("der_rasio"), 2)
                     + " x; dividend yield " + _cell(f.get("dividend_yield_persen"), 2)
                     + "%")
    proof = context.get("bukti_pembanding", {})
    equal = proof.get("bobot_merata") or {}
    if proof.get("status") == "tersedia" and equal.get("layak"):
        delta = equal["selisih_fitness_hasil_dikurangi_pembanding"]
        relation = "lebih tinggi" if delta > 1e-9 else "lebih rendah" if delta < -1e-9 else "setara"
        lines.append("Skor hasil " + relation + " daripada pembanding merata pada saham yang sama; "
                     "ini pembanding lokal, bukan bukti unggul pada seluruh metrik atau optimum global.")
    else:
        lines.append("Pembanding setara tidak tersedia; jangan mengklaim keunggulan terhadap alternatif.")
    if context.get("rebalance_deltas"):
        lines.append("Perubahan aktual, boleh disebut singkat jika relevan:")
        for delta in context["rebalance_deltas"]:
            lines.append(str(delta["kode"]) + " " + str(delta["action"]) + "; alasan: " + str(delta["reason"]))
    if context.get("posisi_belum_tercakup"):
        lines.append("Sebagian posisi belum tercakup: risiko portofolio lengkap belum dapat dinilai.")
    lines.extend(context.get("batasan_risiko", []))
    cvar = context.get("uji_cvar") or {}
    if cvar.get("downside_cvar") is not None:
        lines.append("CVaR downside historis: " + _pct(cvar["downside_cvar"]))
    return "\n".join(lines)


TASK_INSTRUCTIONS = """
TUGAS UNTUK {audience}:
Tulis penjelasan yang nyaman dibaca. Sekitar 300–450 kata biasanya cukup, tetapi jangan memotong penjelasan demi mengejar jumlah kata. Pembaca harus memahami makna
hasil sebelum menghadapi angka. Kalimat pertama adalah kesimpulan tentang karakteristik
atau kompromi yang DIDUKUNG data, tanpa angka, tanggal, atau penanda numerik.

Alur yang disarankan:
- Pembuka menjelaskan temuan utama. Jika rebalance, dahulukan makna perubahan aktual.
- Pilih dua saham dengan kontras yang relevan. Untuk setiap rasio yang disebut, jelaskan
  singkat artinya dalam bahasa sehari-hari, arah interpretasi lebih tinggi/rendah, lalu
  hubungkan dengan angka saham dan implikasinya. Definisi cukup sekali, bukan diulang
  per saham. Contoh pola: ROE menunjukkan kemampuan menghasilkan laba dari modal sendiri;
  nilai yang lebih tinggi pada saham terkait menunjukkan laba relatif terhadap ekuitas
  lebih besar, lalu jelaskan pertimbangan DER-nya. Jangan hanya menulis daftar angka.
  Pilih dua atau tiga rasio utama yang membantu; rasio lain boleh dihilangkan jika tidak
  sempat dijelaskan. Jangan menganggap seluruh rasio lebih rendah selalu lebih baik.
- Jelaskan satu atau dua metrik portofolio seperti Sharpe dan MDD dalam bahasa sehari-hari,
  termasuk cara membacanya. Sharpe lebih tinggi umumnya menunjukkan imbal hasil berlebih
  per fluktuasi lebih tinggi pada basis sebanding. MDD lebih kecil menunjukkan penurunan
  historis yang lebih ringan. Jangan memberi label aman/kuat tanpa pembanding.
  Setiap angka harus menjawab pertanyaan pembaca; korelasi tidak wajib disebut.
- Rangkuman implikasi singkat dan satu disclaimer penutup.

Pilih sekitar enam sampai delapan fakta numerik; maksimum dua belas, dan jangan menumpuk
lebih dari empat penanda dalam satu kalimat. Setiap paragraf menambah penjelasan, bukan
mengulang fakta atau batasan. Maksimum satu kalimat pembatas di bagian fundamental dan
satu di bagian risiko, selain disclaimer penutup. Pembuka hanya kesimpulan bermakna,
tanpa peringatan. Jangan membahas keterbatasan skor internal, berita, atau data yang tidak
sedang dijelaskan. Hindari label kuat/aman/unggul untuk metrik GA tanpa pembanding.

Semua angka hanya melalui penanda yang diberikan, pada saham dan metrik yang tepat.
Jangan membahas persentase alokasi, bobot saham, kas, modal, biaya, exposure, atau lot.
Jangan menampilkan nilai fitness, Sharpe penilaian, koefisien, atau besar bonus/penalti.
Kriteria GA aktif boleh dijelaskan singkat secara kualitatif. Berita dan sumbernya tersedia
terpisah di aplikasi: jangan merangkum, mengutip, atau membahas ketersediaan berita dalam
narasi utama. Jangan mengambil berita dari ingatan.
""".strip() + "\n"


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
    rebalance_deltas: Optional[List[Dict[str, Any]]] = None,
    reporting: Optional[Dict[str, Any]] = None,
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
            rebalance_deltas=rebalance_deltas,
            reporting=reporting,
        )

    context = dict(context)
    context["narrative_focus"] = NARRATIVE_VERSION
    context["news_context"] = normalize_news_context(context, context.get("news_context"))
    tugas = TASK_INSTRUCTIONS.replace("{audience}", str(audience))
    if extra_instruction:
        tugas = tugas + "INSTRUKSI TAMBAHAN: " + str(extra_instruction) + "\n"

    marked_context, numeric_facts = mark_numeric_facts(render_investor_context(context))
    user_prompt = (tugas + "\nATURAN PENULISAN ANGKA: Setiap nilai pada data mempunyai penanda [[N...]]. "
        "Dalam jawaban, tulis HANYA penandanya untuk angka yang dipilih; aplikasi akan menggantinya "
        "dengan nilai dan satuan persis. Jangan tulis angka literal, jangan menghitung atau menyingkat rupiah, "
        "dan jangan menulis ulang satuan persen yang sudah termasuk penanda. Nama saham tetap ditulis biasa.\n"
        + marked_context + "\nDAFTAR NILAI PENANDA (untuk memahami data; pada jawaban salin hanya penandanya):\n"
        + "\n".join(token + " = " + value for token, value in numeric_facts.items()))
    return PortfolioPrompt(
        system_instruction=system_instruction or DEFAULT_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        context=context,
        numeric_facts=numeric_facts,
        news_sources={},  # News remains available in metadata, outside the main narrative.
    )


def build_narrative_prompt(*args, **kwargs) -> PortfolioPrompt:
    """
    Alias praktis :func:`build_explanation_prompt` (dipakai
    :mod:`src.backend.services.openai_llm_caller` agar pemanggilan service
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

