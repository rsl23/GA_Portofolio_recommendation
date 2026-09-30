"""
Gemini LLM Caller - Penjelasan Hasil GA (library google-genai)
==============================================================

Pemanggil model Gemini (default: ``gemini-3.8-flash``) untuk MENJELASKAN hasil
Algoritma Genetika: mengapa dan bagaimana portofolio itu terpilih, berdasarkan
metrik fitness function serta daftar emiten saham pemenang.

Pembagian tugas:
    src/gaengine/prompt_builder.py -> merakit KONTEKS + PROMPT (murni, tanpa I/O)
    modul ini                      -> memanggil API Gemini (+ retry & fallback)

Konfigurasi .env (semua opsional kecuali kunci API):
    GEMINI_API_KEY           : kunci API Gemini (fallback: GOOGLE_API_KEY)
    GEMINI_ENABLED           : saklar utama narasi LLM. Nilai yang dianggap
                               NONAKTIF: false / 0 / off / no / tidak.
                               Bila nonaktif, LLM tidak dipanggil sama sekali dan
                               controller memakai narasi template GA.
                               Default: true (aktif) bila variabel tidak ada.
    GEMINI_MODEL             : default "gemini-3.8-flash"
    GEMINI_TEMPERATURE       : default 0.7
    GEMINI_MAX_OUTPUT_TOKENS : default 8192 (token thinking ikut dihitung,
                               batas kecil membuat narasi terpotong)
    GEMINI_THINKING_LEVEL    : default "low" ("off"/"" -> thinking dinonaktifkan,
                               sehingga respons lebih cepat)
    GEMINI_TIMEOUT_MS        : default 120000 (2 menit)
    GEMINI_MAX_RETRIES       : default 3 (total percobaan = retries + 1 per model)
    GEMINI_FALLBACK_MODELS   : model cadangan bila model utama gagal total,
                               default "gemini-3.7-flash,gemini-3.5-flash"
    GEMINI_BACKOFF_BASE/CAP  : dasar & batas exponential backoff (detik)

Ketahanan (anti-gagal) -- mengikuti gaya :func:`src.gaengine.data_loader._call_with_retry`:
    Kegagalan (kunci hilang, timeout, server sibuk, respons kosong/diblokir)
    TIDAK melempar exception ke controller. Fungsi penjelasan mengembalikan
    ``None`` (atau ``fallback`` bila diberikan), sehingga proses generate
    portofolio tetap berjalan walau LLM sedang tidak bisa dihubungi.

Contoh pemakaian:
    from src.backend.services.gemini_llm_caller import explain_portfolio

    narasi = explain_portfolio(
        solution=solution, market_data=market_data, config=config,
        allocations=allocations, mode="live", budget=request.budget,
        fallback=narasi_template,
    )
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from src.gaengine.prompt_builder import build_narrative_prompt

# Load variabel dari file .env (konsisten dengan service lain, mis. api_bi.py)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# Peringatan bawaan SDK (AFC pada generate_content) hanya informasi internal SDK
# dan tidak relevan untuk pemakaian kita -> naikkan levelnya agar terminal bersih.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)




# ----------------------------------------------------------------------
# Utilitas baca .env (aman terhadap nilai kosong / rusak)
# ----------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    """Baca integer dari .env; nilai kosong/rusak -> pakai default."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        logger.warning("_env_int: %s='%s' tidak valid -> pakai default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Baca float dari .env; nilai kosong/rusak -> pakai default."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("_env_float: %s='%s' tidak valid -> pakai default %s", name, raw, default)
        return default


# Nilai teks yang dianggap True / False pada variabel .env bertipe boolean.
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "aktif", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "tidak", "disable", "disabled", "nonaktif"}


def _as_bool(value: Any, default: bool = True) -> bool:
    """
    Ubah nilai apa pun (bool / teks .env) menjadi bool dengan aman.

    PENTING: fungsi ini dipakai supaya saklar seperti GEMINI_ENABLED benar-benar
    bekerja meski nilainya berupa TEKS. Contoh jebakan: nilai "false" (string)
    bersifat TRUTHY di Python (``not "false"`` = False), sehingga pengecekan
    ``if not GEMINI_ENABLED`` akan selalu lolos dan saklar tampak "tidak jalan".
    Teks yang tidak dikenal -> `default` (disertai peringatan).
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    teks = str(value).strip().lower()
    if teks in _TRUE_VALUES:
        return True
    if teks in _FALSE_VALUES:
        return False
    logger.warning("_as_bool: nilai '%s' tidak dikenal -> pakai default %s", value, default)
    return default


def _env_bool(name: str, default: bool) -> bool:
    """Baca boolean dari .env; kosong/rusak -> pakai default."""
    return _as_bool(os.getenv(name), default)


# ----------------------------------------------------------------------
# Konfigurasi model (dibaca saat import; bisa di-override per pemanggilan)
# ----------------------------------------------------------------------
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").strip()
# Saklar utama narasi LLM. GEMINI_ENABLED=false / 0 / off / tidak -> LLM sama
# sekali TIDAK dipanggil (controller otomatis memakai narasi template GA).
# Default: true (aktif) bila variabel ini tidak ada di .env.
# Ditulis memakai _env_bool agar teks "false" tetap dibaca sebagai bool False
# (teks "false" bersifat truthy di Python -> jangan dibandingkan langsung).
GEMINI_ENABLED = _env_bool("GEMINI_ENABLED", True)
GEMINI_TEMPERATURE = _env_float("GEMINI_TEMPERATURE", 0.7)
# PENTING: token "thinking" model Gemini 3.x IKUT dihitung ke max_output_tokens.
# Batas kecil (mis. 2048) membuat narasi terpotong di tengah kalimat karena
# jatah token habis dipakai berpikir -> default 8192 (cukup untuk ~500 kata
# narasi + penalaran).
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 8192)
GEMINI_THINKING_LEVEL = (os.getenv("GEMINI_THINKING_LEVEL") or "low").strip()
GEMINI_TIMEOUT_MS = _env_int("GEMINI_TIMEOUT_MS", 120_000)
GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 3)
GEMINI_BACKOFF_BASE = _env_float("GEMINI_BACKOFF_BASE", 2.0)
GEMINI_BACKOFF_CAP = _env_float("GEMINI_BACKOFF_CAP", 30.0)
# Rantai model CADANGAN, dipakai HANYA bila model utama gagal total setelah
# semua percobaan ulang (mis. 503 "This model is currently experiencing high
# demand"). Isi lewat .env: GEMINI_FALLBACK_MODELS="gemini-3.7-flash,gemini-3.5-flash".
# Isi dengan string kosong untuk mematikan fallback (hanya model utama).
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in str(os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash")).split(",")
    if m.strip()
]

# Nilai GEMINI_THINKING_LEVEL yang berarti "tanpa thinking" (respons lebih cepat).
_THINKING_OFF = {"", "off", "none", "false", "0", "disable", "disabled"}

# Status SDK (di-cache agar import google-genai hanya dicoba sekali).
_sdk: Optional[tuple] = None
_client: Optional[Any] = None


def _load_sdk():
    """
    Import library ``google-genai`` secara LAZY (hanya saat dipakai).

    Returns:
        tuple (genai, types) bila tersedia, atau (None, None) bila library
        tidak terpasang sehingga aplikasi tetap bisa berjalan (LLM nonaktif).
    """
    global _sdk
    if _sdk is not None:
        return _sdk
    try:
        from google import genai
        from google.genai import types
        _sdk = (genai, types)
    except ImportError as e:  # pragma: no cover - bergantung environment
        logger.error("Library google-genai tidak tersedia (%s) -> narasi LLM dilewati.", e)
        _sdk = (None, None)
    return _sdk


def is_available() -> bool:
    """
    True bila narasi LLM bisa dipakai: saklar GEMINI_ENABLED aktif, kunci API
    tersedia, DAN library google-genai bisa diimpor.
    """
    if not _as_bool(GEMINI_ENABLED, default=True):
        logger.info("is_available: GEMINI_ENABLED nonaktif -> narasi LLM dinonaktifkan.")
        return False
    genai, _ = _load_sdk()
    return bool(GEMINI_API_KEY) and genai is not None


def get_client(api_key: Optional[str] = None, timeout_ms: Optional[int] = None):
    """
    Ambil (dan cache) klien Google GenAI.

    Args:
        api_key    : override kunci API (default GEMINI_API_KEY dari .env).
        timeout_ms : timeout request milidetik (default GEMINI_TIMEOUT_MS).

    Returns:
        genai.Client, atau None bila library/kunci tidak tersedia.
    """
    global _client
    key = (api_key or GEMINI_API_KEY or "").strip()
    if not key:
        logger.warning("get_client: GEMINI_API_KEY tidak ditemukan di .env -> LLM dilewati.")
        return None
    if _client is not None:
        return _client

    genai, types = _load_sdk()
    if genai is None:
        return None
    try:
        _client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout_ms or GEMINI_TIMEOUT_MS)),
        )
        logger.info("get_client: klien Gemini siap (model default=%s, timeout=%dms).",
                    GEMINI_MODEL, int(timeout_ms or GEMINI_TIMEOUT_MS))
    except Exception as e:  # noqa: BLE001 - konfigurasi klien gagal -> LLM dilewati
        logger.error("get_client: gagal membuat klien Gemini (%s) -> LLM dilewati.", e)
        _client = None
    return _client


# ----------------------------------------------------------------------
# Retry & pembersihan respons
# ----------------------------------------------------------------------
def _retry_delay(attempt: int) -> float:
    """
    Jeda sebelum percobaan ulang ke-`attempt` (1-indexed): exponential backoff
    (GEMINI_BACKOFF_BASE ** attempt) dibatasi GEMINI_BACKOFF_CAP, ditambah
    jitter 0-25% agar request tidak menumpuk pada detik yang sama.
    """
    delay = min(GEMINI_BACKOFF_BASE ** attempt, GEMINI_BACKOFF_CAP)
    return delay + random.uniform(0.0, delay * 0.25)


def _build_config(
    types,
    system_instruction: Optional[str],
    temperature: float,
    max_output_tokens: int,
    thinking_level: str,
):
    """Susun GenerateContentConfig (system instruction, temperatur, thinking)."""
    kwargs: Dict[str, Any] = {
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
    }
    if system_instruction:
        kwargs["system_instruction"] = str(system_instruction)

    level = str(thinking_level or "").strip().lower()
    if level in _THINKING_OFF:
        # thinking_budget=0 mematikan penalaran internal -> respons lebih cepat.
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    else:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(level.upper())
            )
        except (ValueError, AttributeError):
            logger.warning("_build_config: thinking_level '%s' tidak dikenal -> dimatikan.", level)
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def _extract_text(response) -> Optional[str]:
    """Ambil teks dari GenerateContentResponse; None bila kosong/diblokir."""
    if response is None:
        return None
    teks = getattr(response, "text", None)
    if isinstance(teks, str) and teks.strip():
        return teks.strip()

    # Fallback: rakit manual dari parts (mis. saat .text None karena multi-part).
    potongan: list[str] = []
    for kandidat in getattr(response, "candidates", None) or []:
        isi = getattr(kandidat, "content", None)
        for part in (getattr(isi, "parts", None) or []):
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip():
                potongan.append(t.strip())
    gabung = "\n".join(potongan).strip()
    return gabung or None


def _sanitize_narrative(text: str) -> str:
    """
    Rapikan narasi LLM menjadi teks polos (sesuai permintaan pada prompt):
    buang penanda Markdown (heading, bullet, bold, garis pemisah) namun
    JANGAN mengubah angka/kata penting.
    """
    if not text:
        return ""
    baris_bersih: list[str] = []
    for baris in str(text).replace("\r\n", "\n").split("\n"):
        b = baris.rstrip()
        b = re.sub(r"^\s{0,3}#{1,6}\s*", "", b)          # heading "# ..."
        b = re.sub(r"^\s{0,3}[-*•]\s+", "", b)            # bullet "- ", "* ", "• "
        b = re.sub(r"^\s{0,3}\d+[\.\)]\s+", "", b)        # penomoran "1. ", "2) "
        if b.strip() in {"---", "***", "___"}:
            continue
        baris_bersih.append(b)
    hasil = "\n".join(baris_bersih)
    hasil = hasil.replace("**", "").replace("__", "")
    hasil = re.sub(r"\n{3,}", "\n\n", hasil)
    return hasil.strip()



# ----------------------------------------------------------------------
# API publik
# ----------------------------------------------------------------------
def _model_chain(
    model: Optional[str] = None,
    fallback_models: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Susun urutan model yang akan dicoba: model utama lebih dulu, lalu model
    cadangan (tanpa duplikat).
    """
    utama = (model or GEMINI_MODEL).strip()
    cadangan = GEMINI_FALLBACK_MODELS if fallback_models is None else list(fallback_models)
    rantai: List[str] = [utama] if utama else []
    for m in cadangan:
        nama = str(m).strip()
        if nama and nama not in rantai:
            rantai.append(nama)
    return rantai


def _generate_with_model(
    client,
    config,
    nama_model: str,
    prompt_teks: str,
    total: int,
) -> Optional[str]:
    """
    Jalankan SATU model dengan retry + exponential backoff + jitter.

    Returns:
        Teks jawaban (belum dibersihkan), atau None bila seluruh percobaan
        pada model ini gagal.
    """
    percobaan_ulang = total - 1
    for attempt in range(1, total + 1):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=nama_model,
                contents=prompt_teks,
                config=config,
            )
            teks = _extract_text(response)
            if teks:
                logger.info("generate_text: SUKSES model=%s pada percobaan %d/%d dalam %.2fs "
                            "(%d karakter jawaban).",
                            nama_model, attempt, total, time.perf_counter() - t0, len(teks))
                return teks
            alasan = "respons kosong / diblokir safety filter"
        except Exception as e:  # noqa: BLE001 - semua kegagalan API layak dicoba ulang
            alasan = f"exception: {type(e).__name__}: {e}"

        if attempt < total:
            delay = _retry_delay(attempt)
            print(f"   [RETRY {attempt}/{percobaan_ulang}] gemini {nama_model}: {alasan} "
                  f"-> ulangi dalam {delay:.1f}s...")
            logger.warning("generate_text: model=%s percobaan %d/%d gagal (%s) -> ulang dalam %.1fs.",
                           nama_model, attempt, total, alasan, delay)
            time.sleep(delay)
        else:
            print(f"   [GAGAL] gemini {nama_model}: tetap gagal setelah {total} percobaan ({alasan}).")
            logger.error("generate_text: model=%s GAGAL setelah %d percobaan (%s).",
                         nama_model, total, alasan)
    return None


def generate_text(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = None,
    max_retries: Optional[int] = None,
    fallback_models: Optional[Sequence[str]] = None,
    plain_text: bool = False,
) -> Optional[str]:
    """
    Kirim satu prompt ke Gemini dan kembalikan teks jawabannya.

    Ketahanan dua lapis:
      1) RETRY per model   : exponential backoff + jitter untuk gangguan sesaat
                             (timeout, server sibuk 5xx, respons kosong).
      2) FALLBACK MODEL    : bila model utama gagal total, otomatis dicoba model
                             cadangan berikutnya (lihat GEMINI_FALLBACK_MODELS).

    Saklar GEMINI_ENABLED (di .env) juga dihormati di sini: bila nonaktif,
    panggilan API dibatalkan dan fungsi mengembalikan None (tanpa exception).

    Args:
        prompt            : teks prompt (wajib).
        system_instruction: instruksi sistem (peran & aturan output).
        model             : nama model (default GEMINI_MODEL = gemini-3.8-flash).
        temperature       : kreativitas jawaban (default GEMINI_TEMPERATURE).
        max_output_tokens : batas panjang jawaban (default dari .env).
        thinking_level    : "off"/"low"/"medium"/"high" (default dari .env).
        max_retries       : jumlah percobaan ULANG per model (default GEMINI_MAX_RETRIES).
        fallback_models   : override daftar model cadangan.
        plain_text        : True -> buang penanda Markdown dari jawaban.

    Returns:
        str jawaban (sudah di-strip), atau None bila saklar mati / SEMUA model
        & percobaan gagal.
    """
    if not prompt or not str(prompt).strip():
        logger.warning("generate_text: prompt kosong -> dilewati.")
        return None

    if not _as_bool(GEMINI_ENABLED, default=True):
        logger.warning("generate_text: GEMINI_ENABLED nonaktif -> panggilan API dibatalkan.")
        return None

    client = get_client()
    if client is None:
        return None
    _, types = _load_sdk()

    percobaan_ulang = GEMINI_MAX_RETRIES if max_retries is None else max(0, int(max_retries))
    total = percobaan_ulang + 1
    config = _build_config(
        types,
        system_instruction,
        GEMINI_TEMPERATURE if temperature is None else float(temperature),
        GEMINI_MAX_OUTPUT_TOKENS if max_output_tokens is None else int(max_output_tokens),
        GEMINI_THINKING_LEVEL if thinking_level is None else thinking_level,
    )
    prompt_teks = str(prompt)
    rantai = _model_chain(model, fallback_models)
    logger.info("generate_text: rantai model=%s, panjang_prompt=%d karakter, maks %d percobaan/model.",
                rantai, len(prompt_teks), total)

    for urutan, nama_model in enumerate(rantai, 1):
        if urutan > 1:
            print(f"   [FALLBACK] model sebelumnya gagal -> coba model cadangan "
                  f"'{nama_model}' ({urutan}/{len(rantai)})...")
            logger.warning("generate_text: beralih ke model cadangan '%s' (%d/%d).",
                           nama_model, urutan, len(rantai))
        teks = _generate_with_model(client, config, nama_model, prompt_teks, total)
        if teks:
            return _sanitize_narrative(teks) if plain_text else teks

    logger.error("generate_text: SEMUA model gagal (%s).", rantai)
    return None



def explain_portfolio(
    solution=None,
    market_data=None,
    config=None,
    *,
    context: Optional[Dict[str, Any]] = None,
    allocations: Optional[Sequence[Dict[str, Any]]] = None,
    mode: str = "live",
    date_ref: Optional[Any] = None,
    budget: Optional[float] = None,
    expected_return: Optional[float] = None,
    generations_run: Optional[int] = None,
    fallback: Optional[str] = None,
    audience: str = "investor ritel",
    extra_instruction: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    thinking_level: Optional[str] = None,
    fallback_models: Optional[Sequence[str]] = None,
    log_prompt: bool = False,
) -> Optional[str]:
    """
    Minta LLM MENJELASKAN hasil GA: mengapa & bagaimana portofolio terpilih.

    Alur: rakit konteks metrik (prompt_builder) -> susun prompt -> panggil
    Gemini -> rapikan menjadi teks polos.

    Args:
        solution, market_data, config : objek hasil GA (Chromosome, MarketData, GAConfig).
        context        : bila sudah punya konteks dari build_portfolio_context,
                         boleh langsung diberikan (tanpa solution/market_data).
        allocations    : daftar alokasi dari controller (cadangan bila perlu).
        mode, date_ref : konteks waktu ("live"/"backtest", tanggal acuan).
        budget         : modal pengguna (default config.budget).
        expected_return: estimasi return tahunan (default dihitung dari returns).
        fallback       : teks yang dikembalikan bila LLM tidak tersedia/gagal
                         (mis. narasi template controller) -> generate portofolio
                         TIDAK PERNAH gagal karena masalah LLM.
        audience       : sasaran pembaca narasi.
        log_prompt     : True -> cetak prompt lengkap (untuk debugging kualitas prompt).
        model/temperature/max_output_tokens/thinking_level/fallback_models :
                         override konfigurasi LLM (default dari .env).

    Returns:
        str narasi penjelasan, atau `fallback` (boleh None) bila LLM gagal.
    """
    if not is_available():
        logger.warning("explain_portfolio: LLM tidak tersedia (kunci API / library) -> pakai fallback.")
        return fallback

    try:
        prompt = build_narrative_prompt(
            context,
            solution=solution,
            market_data=market_data,
            config=config,
            allocations=allocations,
            mode=mode,
            date_ref=date_ref,
            budget=budget,
            expected_return=expected_return,
            generations_run=generations_run,
            audience=audience,
            extra_instruction=extra_instruction,
        )
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("explain_portfolio: gagal merakit prompt (%s) -> pakai fallback.", e)
        return fallback

    konteks = prompt.context or {}
    kinerja = konteks.get("kinerja_portofolio", {})
    logger.info(
        "explain_portfolio: mode=%s, n_emiten=%d, fitness=%s, sharpe=%s -> memanggil %s.",
        konteks.get("mode"), len(konteks.get("emiten_terpilih", [])),
        kinerja.get("fitness"), kinerja.get("sharpe_ratio"),
        (model or GEMINI_MODEL),
    )
    if log_prompt:
        print("================= PROMPT KE GEMINI =================")
        print(prompt.full_text)
        print("===================================================")

    t0 = time.perf_counter()
    teks = generate_text(
        prompt.user_prompt,
        system_instruction=prompt.system_instruction,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
        fallback_models=fallback_models,
        plain_text=True,
    )
    if not teks:
        logger.error("explain_portfolio: LLM gagal/ kosong -> pakai fallback.")
        return fallback

    logger.info("explain_portfolio: narasi LLM siap (%d karakter) dalam %.2fs.",
                len(teks), time.perf_counter() - t0)
    return teks


__all__ = [
    "GEMINI_API_KEY",
    "GEMINI_ENABLED",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODELS",
    "GEMINI_TEMPERATURE",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_THINKING_LEVEL",
    "GEMINI_TIMEOUT_MS",
    "GEMINI_MAX_RETRIES",
    "is_available",
    "get_client",
    "generate_text",
    "explain_portfolio",
]

