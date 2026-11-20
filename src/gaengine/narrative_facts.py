"""Numeric references keep LLM prose from calculating or reformatting facts."""
import re
from .news_context import NEWS_TOKEN
from .narrative_style import NARRATIVE_VERSION

TOKEN = re.compile(r"\[\[N\d{3,}\]\]")
NUMBER = re.compile(
    r"(?<![\w])(?:\d{4}-\d{2}-\d{2}|Rp-?\d+(?:\.\d{3})*|-?\d+(?:\.\d+)?%?)(?![\w])"
)

_RATIO_KEYS = {"PER": "per_x", "PBV": "pbv_x", "ROE": "roe_persen", "DER": "der_rasio"}


def _check_explicit_ratio_claims(text, facts, context):
    """Check unambiguous stock/ratio/token statements, without guessing prose.

    This is bounded validation, not a semantic proof of every sentence. Compound
    comparisons without an explicit owner still require editorial evaluation.
    """
    issues = []
    gap = r"\s*(?:(?:memiliki|mencatatkan|mencatat|menunjukkan|sebesar|mencapai|tercatat|adalah|rasio|nilai|saham|dengan)\s+){0,3}"
    for row in context.get("emiten_terpilih", []):
        ticker = re.escape(row["ticker"])
        ratio = r"(?P<ratio>PER|PBV|ROE|DER)"
        token = r"(?P<token>\[\[N\d{3,}\]\])"
        patterns = (r"\b" + ticker + r"\b" + gap + ratio + r"\b" + gap + token,
                    r"\b" + ratio + r"\b" + gap + ticker + r"\b" + gap + token)
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                label = match.group("ratio").upper()
                expected = (row.get("fundamental") or {}).get(_RATIO_KEYS[label])
                supplied = facts.get(match.group("token"), "")
                value = re.match(r"-?\d+(?:\.\d+)?", supplied)
                wrong_unit = ("%" in supplied) != (label == "ROE")
                if (expected is None or not value or wrong_unit
                        or abs(float(value.group()) - round(float(expected), 2)) > .000001):
                    issues.append(f"Periksa {label} {row['ticker']}: gunakan penanda milik saham dan metrik tersebut.")
    return list(dict.fromkeys(issues))


def mark_numeric_facts(text):
    facts, reverse = {}, {}

    def mark(match):
        value = match.group()
        if value not in reverse:
            token = f"[[N{len(facts)+1:03d}]]"
            reverse[value] = token
            facts[token] = value
        return reverse[value]

    return NUMBER.sub(mark, text), facts


def resolve_narrative(text, facts, tickers=(), news_sources=None, context=None):
    """Return validated prose or reasons to request one bounded correction."""
    issues = []
    modern = context is not None and context.get("narrative_focus") == NARRATIVE_VERSION
    news_sources = news_sources or {}
    markers = re.findall(r"\[\[.*?\]\]", text)
    if any(marker not in facts and marker not in news_sources for marker in markers):
        issues.append("Ada penanda angka atau sumber berita yang tidak diberikan pada data.")
    without_tokens = NEWS_TOKEN.sub("", TOKEN.sub("", text))
    for ticker in tickers:
        without_tokens = re.sub(r"\b" + re.escape(ticker) + r"\b", "", without_tokens)
    if re.search(r"\d", without_tokens):
        issues.append("Semua angka harus menggunakan penanda [[N...]], tanpa angka literal atau perhitungan sendiri.")
    resolved = TOKEN.sub(lambda match: facts.get(match.group(), match.group()), text)
    resolved = NEWS_TOKEN.sub(lambda match: news_sources.get(match.group(), match.group()), resolved)
    word_limit = 600 if modern else 320
    if len(resolved.split()) > word_limit:
        issues.append(f"Narasi melebihi {word_limit} kata; ringkas dengan mengurangi angka dan rincian.")
    numeric_limit = 18 if context is not None and context.get("narrative_focus") == "fundamental_ga_news" else 12
    if len(set(TOKEN.findall(text))) > numeric_limit:
        issues.append(f"Terlalu banyak angka; gunakan maksimal {numeric_limit} fakta numerik, utamakan yang membantu investor.")
    if not 2 <= len([part for part in re.split(r"\n\s*\n", text) if part.strip()]) <= 5:
        issues.append("Gunakan dua sampai lima paragraf pendek, sesuai bukti yang tersedia.")
    if re.search(r"https?://|www\.", text, re.I):
        issues.append("Jangan menciptakan tautan; gunakan penanda sumber berita yang disediakan.")
    if re.search(r"\[B\d+\]", without_tokens):
        issues.append("Rujuk berita dengan penanda lengkap [[B001]], bukan nomor sumber literal.")
    news_claim = re.search(r"akuisisi|merger|menurut\s+(?:berita|pemberitaan)|sektor\s+\w+\s+(?:sedang|menguat)|mengumumkan", text, re.I)
    if news_claim and not any(marker in news_sources for marker in NEWS_TOKEN.findall(text)):
        issues.append("Klaim peristiwa/berita harus didukung penanda sumber yang tersedia; hapus jika tidak ada bukti.")
    if context is not None and re.search(r"\bCVaR\b", text, re.I):
        cvar = context.get("uji_cvar") or {}
        if cvar.get("downside_cvar") is None:
            issues.append("Hapus pembahasan CVaR; metrik ini tidak tersedia pada hasil portofolio.")
        elif cvar["downside_cvar"] != 0 and re.search(r"CVaR\s+(?:downside\s+)?(?:nol|zero)\b", text, re.I):
            issues.append("Jangan menyebut CVaR nol karena nilai pengujiannya bukan nol.")
    if context is not None and context.get("narrative_focus") in {"fundamental_ga_news", NARRATIVE_VERSION}:
        if re.search(r"bobot\s+(?:modal|saham)|persentase\s+alokasi|proporsi\s+(?:terhadap\s+)?modal|porsi\s+(?:saham|kas)|sisa\s+kas|kas\s+setelah\s+biaya|batas\s+(?:maksimum\s+)?exposure|pembulatan\s+lot", text, re.I):
            issues.append("Hapus uraian alokasi dan kas; fokus pada rasio fundamental, metrik GA, dan berita bersumber.")
        available = any(any((row.get("fundamental") or {}).get(key) is not None
                            for key in ("per_x", "pbv_x", "roe_persen", "der_rasio", "dividend_yield_persen"))
                        for row in context.get("emiten_terpilih", []))
        if available and not re.search(r"\b(?:PER|PBV|ROE|DER)\b|dividend\s+yield|imbal\s+hasil\s+dividen", text, re.I):
            issues.append("Tambahkan analisis rasio fundamental yang tersedia; jangan hanya menyebut skor fundamental umum.")
        for sentence in re.split(r"[.!?\n]", text):
            if (re.search(r"tidak\s+membayar\s+dividen", sentence, re.I) and not NEWS_TOKEN.search(sentence)
                    and not re.search(r"tidak membuktikan|bukan bukti|belum membuktikan|tidak berarti", sentence, re.I)):
                issues.append("Yield nol tidak membuktikan perusahaan tidak membayar dividen; jelaskan hanya yield yang tercatat.")
            if (re.search(r"korelasi.*(?:menurunkan|mengurangi)\s+risiko", sentence, re.I)
                    and not re.search(r"tidak|bukan|belum|tanpa", sentence, re.I)):
                issues.append("Jangan menyimpulkan korelasi menurunkan risiko tanpa pembanding risiko yang setara.")
    if modern:
        first = re.split(r"[.!?](?:\s|$)|\n", text.strip(), maxsplit=1)[0]
        if TOKEN.search(first) or re.search(r"\d", first):
            issues.append("Awali dengan kesimpulan bermakna tanpa angka; tempatkan angka sebagai bukti pada kalimat berikutnya.")
        if re.search(r"\bberita\b|\bpemberitaan\b|\bakuisisi\b|\bmerger\b|\[\[B", text, re.I):
            issues.append("Berita tersedia pada bagian terpisah; hapus pembahasan dan rujukan berita dari narasi utama.")
        if re.search(r"\bfitness\b|Sharpe penilaian|\bkoefisien\b", text, re.I):
            issues.append("Jelaskan metrik portofolio dan maknanya, tanpa skor fitness atau rincian penilaian internal.")
        if any(len(TOKEN.findall(sentence)) > 4 for sentence in re.split(r"[.!?\n]", text)):
            issues.append("Terlalu banyak angka dalam satu kalimat; pilih bukti utama dan jelaskan maknanya.")
        issues.extend(_check_explicit_ratio_claims(text, facts, context))
    if re.search(r"legacy_cvar_v1|skor lama|penalti turnover|tanpa mengorbankan Sharpe", text, re.I):
        issues.append("Hapus kode teknis, sebab pembelian dari turnover, dan klaim mempertahankan Sharpe tanpa pengujian setara.")
    if re.search(r"(?:bobot|komposisi)\s+optimal\b", text, re.I):
        issues.append("Sebut hasil pencarian, bukan bobot atau komposisi optimal.")
    return resolved, issues
