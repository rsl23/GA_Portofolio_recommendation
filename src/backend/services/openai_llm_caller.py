"""Penjelasan hasil GA melalui OpenAI Responses API (GPT-6 Luna).

Prompt dan angka portofolio tetap disusun oleh ``src.gaengine.prompt_builder``.
Konfigurasi .env:
    OPENAI_API_KEY           : kunci API OpenAI (wajib untuk narasi LLM).
    OPENAI_ENABLED           : default true; false memakai narasi template GA.
    OPENAI_MODEL             : default gpt-6-luna.
    OPENAI_REASONING_EFFORT : none / low / medium / high / xhigh / max;
                             default medium untuk analisis.
    OPENAI_MAX_OUTPUT_TOKENS : default 4096, termasuk token penalaran.
    OPENAI_TIMEOUT_MS        : default 120000 (milidetik).
    OPENAI_MAX_RETRIES       : default 3, dipakai bila retry aktif.
    OPENAI_RETRY_ENABLED     : default false.
    OPENAI_RETURN_ERROR      : default false; true mengembalikan [LLM ERROR]
                             untuk diagnosis bila layanan belum siap.
    OPENAI_FALLBACK_MODELS   : daftar model cadangan dipisah koma;
                            default kosong = tanpa pergantian model.
                            Dipakai bila OPENAI_RETURN_ERROR=false.
    OPENAI_BACKOFF_BASE/CAP  : default 2.0 / 30.0 detik.

SDK diimpor saat diperlukan sehingga backend tetap dapat berjalan tanpa SDK/key.
Retry SDK dimatikan agar jumlah percobaan mengikuti konfigurasi layanan ini.
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
from src.gaengine.narrative_facts import resolve_narrative
from src.gaengine.verified_narrative import NarrativeText, render_verified_narrative
from src.gaengine.narrative_style import NARRATIVE_VERSION
from src.backend.services.portfolio_news import load_portfolio_news
from src.backend.services.cancellable_io import cancellable_io
from src.gaengine.progress import AnalysisCancelled, check_cancelled, cancellable_sleep

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        logger.warning("%s tidak valid -> pakai default %d", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning("%s tidak valid -> pakai default %s", name, default)
        return default


_TRUE_VALUES = {"1", "true", "yes", "y", "on", "aktif", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "tidak", "disable", "disabled", "nonaktif"}


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    teks = str(value).strip().lower()
    if teks in _TRUE_VALUES:
        return True
    if teks in _FALSE_VALUES:
        return False
    logger.warning("Nilai boolean '%s' tidak dikenal -> pakai default %s", value, default)
    return default


OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_ENABLED = _as_bool(os.getenv("OPENAI_ENABLED"), True)
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-6-luna").strip()
OPENAI_REASONING_EFFORT = (os.getenv("OPENAI_REASONING_EFFORT") or "medium").strip().lower()
if OPENAI_REASONING_EFFORT not in {"none", "low", "medium", "high", "xhigh", "max"}:
    OPENAI_REASONING_EFFORT = "medium"
OPENAI_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 4096)
OPENAI_TIMEOUT_MS = _env_int("OPENAI_TIMEOUT_MS", 120_000)
OPENAI_MAX_RETRIES = _env_int("OPENAI_MAX_RETRIES", 3)
OPENAI_RETRY_ENABLED = _as_bool(os.getenv("OPENAI_RETRY_ENABLED"), False)
OPENAI_RETURN_ERROR = _as_bool(os.getenv("OPENAI_RETURN_ERROR"), False)
OPENAI_BACKOFF_BASE = _env_float("OPENAI_BACKOFF_BASE", 2.0)
OPENAI_BACKOFF_CAP = _env_float("OPENAI_BACKOFF_CAP", 30.0)
OPENAI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("OPENAI_FALLBACK_MODELS", "").split(",")
    if model.strip()
]
LLM_ERROR_PREFIX = "[LLM ERROR]"

_sdk: Optional[tuple] = None
_client: Optional[Any] = None


def _load_sdk():
    """Impor SDK OpenAI sekali; kegagalan impor tidak menggagalkan startup."""
    global _sdk
    if _sdk is None:
        try:
            from openai import OpenAI

            _sdk = (OpenAI,)
        except ImportError as exc:
            logger.error("Library openai tidak tersedia (%s) -> narasi LLM dilewati.", exc)
            _sdk = (None,)
    return _sdk[0]


def is_available() -> bool:
    return (
        _as_bool(OPENAI_ENABLED, True)
        and bool(OPENAI_API_KEY)
        and _load_sdk() is not None
    )


def get_client(api_key: Optional[str] = None, timeout_ms: Optional[int] = None):
    """Ambil klien OpenAI; timeout .env dikonversi dari milidetik ke detik."""
    global _client
    key = (api_key or OPENAI_API_KEY or "").strip()
    if not key:
        logger.warning("OPENAI_API_KEY tidak ditemukan -> LLM dilewati.")
        return None
    if _client is not None and api_key is None and timeout_ms is None:
        return _client
    sdk = _load_sdk()
    if sdk is None:
        return None
    try:
        client = sdk(
            api_key=key,
            timeout=float(OPENAI_TIMEOUT_MS if timeout_ms is None else timeout_ms) / 1000.0,
            max_retries=0,
        )
    except Exception as exc:
        logger.error("Gagal membuat klien OpenAI (%s) -> LLM dilewati.", exc)
        return None
    if api_key is None and timeout_ms is None:
        _client = client
    return client


def _retry_delay(attempt: int) -> float:
    delay = min(OPENAI_BACKOFF_BASE ** attempt, OPENAI_BACKOFF_CAP)
    return delay + random.uniform(0.0, delay * 0.25)


def _extract_text(response) -> Optional[str]:
    """Ambil teks Responses API yang selesai, tanpa menerima jawaban terpotong."""
    if getattr(response, "status", None) != "completed":
        return None
    text = getattr(response, "output_text", None)
    return text.strip() if isinstance(text, str) and text.strip() else None


def _sanitize_narrative(text: str) -> str:
    """Hapus penanda Markdown tanpa mengubah angka dan kata dalam narasi."""
    baris_bersih: list[str] = []
    for baris in str(text).replace("\r\n", "\n").split("\n"):
        b = baris.rstrip()
        b = re.sub(r"^\s{0,3}#{1,6}\s*", "", b)
        b = re.sub(r"^\s{0,3}[-*•]\s+", "", b)
        b = re.sub(r"^\s{0,3}\d+[\.)]\s+", "", b)
        if b.strip() in {"---", "***", "___"}:
            continue
        baris_bersih.append(b)
    hasil = "\n".join(baris_bersih).replace("**", "").replace("__", "")
    return re.sub(r"\n{3,}", "\n\n", hasil).strip()


def _model_chain(
    model: Optional[str] = None,
    fallback_models: Optional[Sequence[str]] = None,
) -> List[str]:
    utama = (model or OPENAI_MODEL).strip()
    cadangan = OPENAI_FALLBACK_MODELS if fallback_models is None else fallback_models
    rantai = [utama] if utama else []
    for item in cadangan:
        nama = str(item).strip()
        if nama and nama not in rantai:
            rantai.append(nama)
    return rantai


def _generate_with_model(client, config, nama_model, messages, total, deadline):
    alasan = "tidak ada percobaan yang dijalankan"
    for attempt in range(1, total + 1):
        check_cancelled()
        if time.monotonic() >= deadline:
            return None, "batas waktu narasi habis"
        try:
            request_config = dict(config)
            response = cancellable_io(lambda: client.responses.create(
                model=nama_model,
                input=messages,
                **request_config,
            ), timeout=deadline - time.monotonic())
            usage = getattr(response, "usage", None)
            if usage is not None:
                logger.info(
                    "OpenAI model=%s tokens input=%s output=%s reasoning=%s total=%s",
                    nama_model, usage.input_tokens, usage.output_tokens,
                    getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None),
                    usage.total_tokens,
                )
            teks = _extract_text(response)
            if teks:
                return teks, None
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            alasan = ("batas token keluaran habis sebelum narasi tersedia"
                      if reason == "max_output_tokens"
                      else f"respons kosong / diblokir / belum selesai (status={getattr(response, 'status', None)})")
        except AnalysisCancelled:
            raise
        except Exception as exc:
            alasan = f"exception: {type(exc).__name__}: {exc}"
        logger.warning("OpenAI model=%s percobaan=%d/%d: %s", nama_model, attempt, total, alasan)
        if attempt < total:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            cancellable_sleep(min(_retry_delay(attempt), remaining))
    return None, alasan


def generate_text(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
    fallback_models: Optional[Sequence[str]] = None,
    plain_text: bool = False,
    reasoning_effort: Optional[str] = None,
    _deadline: Optional[float] = None,
) -> Optional[str]:
    """Kirim prompt ke OpenAI; kegagalan tidak melempar exception ke controller.

    Retry eksplisit mengesampingkan OPENAI_RETRY_ENABLED. OPENAI_RETURN_ERROR=true
    mempertahankan mode debugging: error model pertama langsung dikembalikan.
    Bila false, coba model cadangan lalu kembalikan None jika semuanya gagal.
    Pembatalan analisis diteruskan agar worker berhenti sebelum menyimpan hasil.
    """
    check_cancelled()
    deadline = _deadline if _deadline is not None else time.monotonic() + max(.001, OPENAI_TIMEOUT_MS / 1000.)
    if not prompt or not str(prompt).strip() or not _as_bool(OPENAI_ENABLED, True):
        return None
    client = get_client()
    if client is None:
        if _as_bool(OPENAI_RETURN_ERROR, False):
            return (
                f"{LLM_ERROR_PREFIX} klien OpenAI tidak siap "
                "(OPENAI_API_KEY kosong / library openai tidak tersedia)."
            )
        return None

    if max_retries is not None:
        percobaan_ulang = max(0, int(max_retries))
    elif _as_bool(OPENAI_RETRY_ENABLED, False):
        percobaan_ulang = max(0, OPENAI_MAX_RETRIES)
    else:
        percobaan_ulang = 0
    config = {
        "max_output_tokens": (
            OPENAI_MAX_OUTPUT_TOKENS if max_output_tokens is None else int(max_output_tokens)
        ),
        "store": False,
        "reasoning": {"effort": reasoning_effort or OPENAI_REASONING_EFFORT},
    }
    messages = []
    if system_instruction:
        messages.append({"role": "developer", "content": str(system_instruction)})
    messages.append({"role": "user", "content": str(prompt)})

    for nama_model in _model_chain(model, fallback_models):
        check_cancelled()
        if time.monotonic() >= deadline:
            break
        teks, pesan_error = _generate_with_model(
            client, config, nama_model, messages, percobaan_ulang + 1, deadline,
        )
        if teks:
            return _sanitize_narrative(teks) if plain_text else teks
        if _as_bool(OPENAI_RETURN_ERROR, False):
            return f"{LLM_ERROR_PREFIX} {nama_model}: {pesan_error}"
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
    rebalance_deltas: Optional[List[Dict[str, Any]]] = None,
    reporting: Optional[Dict[str, Any]] = None,
    fallback: Optional[str] = None,
    audience: str = "investor ritel",
    extra_instruction: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    fallback_models: Optional[Sequence[str]] = None,
    log_prompt: bool = False,
) -> Optional[str]:
    """Susun prompt hasil GA, minta penjelasan OpenAI, lalu rapikan menjadi teks.

    Jika OpenAI dinonaktifkan, gunakan fallback tanpa memanggil API. Jika OpenAI
    gagal, OPENAI_RETURN_ERROR menentukan apakah error atau fallback dikembalikan.
    Konteks portofolio dan delta rebalancing diteruskan ke prompt seperti semula.
    """
    check_cancelled()
    deadline = time.monotonic() + max(.001, OPENAI_TIMEOUT_MS / 1000.)
    available = is_available()
    if not available:
        if not _as_bool(OPENAI_ENABLED, True):
            logger.warning("Narasi OpenAI dinonaktifkan: OPENAI_ENABLED=false -> pakai template GA.")
        if _as_bool(OPENAI_ENABLED, True) and _as_bool(OPENAI_RETURN_ERROR, False):
            return (
                f"{LLM_ERROR_PREFIX} narasi LLM tidak dapat dijalankan: "
                "OPENAI_API_KEY tidak tersedia / library openai tidak terpasang."
            )
        if fallback is not None:
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
            rebalance_deltas=rebalance_deltas,
            reporting=reporting,
            audience=audience,
            extra_instruction=extra_instruction,
        )
        if available and (context is None or "news_context" not in context):
            # Separate display metadata: news is not sent to the main narrator.
            try:
                prompt.context["news_context"] = cancellable_io(
                    lambda: load_portfolio_news(prompt.context),
                    timeout=min(15., max(0., deadline - time.monotonic())))
            except TimeoutError:
                from src.gaengine.news_context import normalize_news_context
                prompt.context["news_context"] = normalize_news_context(prompt.context,
                    {"reason": "Pencarian berita melewati batas waktu."})
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.error("Gagal merakit prompt (%s) -> pakai fallback.", exc)
        return fallback

    if not available:
        return NarrativeText(render_verified_narrative(prompt.context), "verified_template", prompt.context["news_context"])

    logger.info("Narasi provider=openai model=%s mode=%s decision=%s version=%s",
                model or OPENAI_MODEL, prompt.context.get("mode"),
                prompt.context.get("jenis_keputusan"), NARRATIVE_VERSION)

    if log_prompt:
        print("================ PROMPT KE OPENAI ================")
        print(prompt.full_text)
        print("=================================================")

    t0 = time.perf_counter()
    request_text = prompt.user_prompt
    teks = None
    tickers = [row["ticker"] for row in prompt.context.get("emiten_terpilih", [])]
    for attempt in range(2):
        check_cancelled()
        draft = generate_text(
            request_text,
            system_instruction=prompt.system_instruction,
            model=model,
            max_output_tokens=max_output_tokens,
            fallback_models=fallback_models,
            plain_text=True,
            reasoning_effort=OPENAI_REASONING_EFFORT,
            _deadline=deadline,
        )
        if not draft or draft.startswith(LLM_ERROR_PREFIX):
            teks = draft
            break
        resolved, issues = resolve_narrative(draft, prompt.numeric_facts, tickers, prompt.news_sources, prompt.context)
        if not issues:
            teks = resolved
            break
        logger.warning("Narasi OpenAI belum valid (percobaan %d/2): %s", attempt + 1, "; ".join(issues))
        request_text = (prompt.user_prompt + "\nPERBAIKI DRAF SEBELUMNYA. " + " ".join(issues)
                        + "\nKembalikan hanya narasi ringkas yang diperbaiki, dengan rujukan berita bila digunakan.\nDRAF:\n" + draft)
    if not teks:
        logger.warning("OpenAI gagal/kosong atau narasi belum valid -> pakai narasi fakta terverifikasi.")
        return fallback if fallback is not None else NarrativeText(render_verified_narrative(prompt.context), "verified_template", prompt.context["news_context"])
    if teks.startswith(LLM_ERROR_PREFIX):
        logger.error("%s", teks)
        return fallback if fallback is not None else NarrativeText(render_verified_narrative(prompt.context), "verified_template", prompt.context["news_context"])
    logger.info("Narasi OpenAI siap (%d karakter) dalam %.2fs.", len(teks), time.perf_counter() - t0)
    return NarrativeText(teks, news_context=prompt.context["news_context"])


__all__ = [
    "OPENAI_API_KEY", "OPENAI_ENABLED", "OPENAI_MODEL", "OPENAI_FALLBACK_MODELS",
    "OPENAI_REASONING_EFFORT", "OPENAI_MAX_OUTPUT_TOKENS", "OPENAI_TIMEOUT_MS",
    "OPENAI_MAX_RETRIES", "OPENAI_RETRY_ENABLED", "OPENAI_RETURN_ERROR",
    "LLM_ERROR_PREFIX", "is_available", "get_client", "generate_text", "explain_portfolio",
]
