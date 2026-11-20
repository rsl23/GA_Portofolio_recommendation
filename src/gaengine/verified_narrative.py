"""Investor-facing fallback built only from the computed explanation context."""
from .narrative_style import fundamental_contrast, narrative_opening


class NarrativeText(str):
    def __new__(cls, text, source="openai", news_context=None):
        obj = super().__new__(cls, text)
        obj.source = source
        obj.news_context = news_context
        return obj


def render_verified_narrative(context):
    def number(value, suffix=""):
        return f"{value:.2f}".replace(".", ",") + suffix

    contrast = fundamental_contrast(context)
    opening = narrative_opening(context, contrast)
    if context.get("jenis_keputusan") == "rebalance":
        actions = {"BUY": "dibeli", "SELL": "dijual"}
        changes = [row["kode"] + " " + actions[row["action"]] for row in context.get("rebalance_deltas", [])
                   if row.get("action") in actions][:3]
        if changes:
            opening = "Rebalance ini menyesuaikan posisi yang tercatat: " + ", ".join(changes) + ". " + opening
        elif context.get("rebalance_deltas"):
            opening = "Hasil rebalance mempertahankan posisi yang tercatat pada data keputusan. " + opening
    paragraphs = [opening]
    if contrast:
        a, b = contrast["first"]["ticker"], contrast["second"]["ticker"]
        sentences = []
        explanations = {
            "per_x": "PER membandingkan harga saham dengan laba per saham. Nilai lebih rendah menunjukkan harga relatif terhadap laba lebih rendah, tetapi belum otomatis murah.",
            "roe_persen": "ROE menunjukkan laba pada periode data relatif terhadap ekuitas atau modal sendiri. Nilai lebih tinggi umumnya menunjukkan profitabilitas terhadap ekuitas lebih besar; pengaruh utang juga perlu diperhatikan.",
            "der_rasio": "DER membandingkan utang dengan ekuitas. Nilai lebih rendah menunjukkan ketergantungan relatif pada utang yang lebih kecil, bukan seluruh risiko saham lebih rendah.",
            "dividend_yield_persen": "Dividend yield menunjukkan dividen tercatat relatif terhadap harga saham. Nilai lebih tinggi berarti imbal hasil dividen historis lebih besar, tanpa memastikan pembayaran berikutnya.",
        }
        for row in contrast["observations"][:3]:
            unit = "%" if row["key"] in {"roe_persen", "dividend_yield_persen"} else "x"
            sentences.append(explanations[row["key"]])
            sentences.append(row["label"] + " " + a + " " + number(row["a"], unit)
                             + ", dibandingkan " + b + " " + number(row["b"], unit) + ".")
        if len(contrast["observations"]) == 1:
            same = []
            for label, key in (("PER", "per_x"), ("PBV", "pbv_x"), ("ROE", "roe_persen"), ("DER", "der_rasio")):
                left = (contrast["first"].get("fundamental") or {}).get(key)
                right = (contrast["second"].get("fundamental") or {}).get(key)
                if left is not None and right is not None and round(left, 2) == round(right, 2):
                    same.append(label)
            if same:
                sentences.append(", ".join(same) + " kedua saham sama pada data yang tersedia, sehingga tidak menjadi dasar pembedaan di sini.")
        sentences.append("Perbandingan ini bersifat relatif pada data yang tersedia; acuannya belum mencakup standar industri.")
        paragraphs.append("Sebagai gambaran, " + " ".join(sentences))
    else:
        rows = context.get("emiten_terpilih", [])
        facts = []
        for row in sorted(rows, key=lambda row: row["ticker"]):
            for label, key, unit in (("ROE", "roe_persen", "%"), ("DER", "der_rasio", "x")):
                value = (row.get("fundamental") or {}).get(key)
                if value is not None:
                    facts.append(label + " " + row["ticker"] + " " + number(value, unit))
            if len(facts) >= 2:
                break
        if facts:
            paragraphs.append("Pada data yang tersedia, " + " dan ".join(facts[:2])
                              + ". ROE menunjukkan laba relatif terhadap modal sendiri; lebih tinggi umumnya menunjukkan profitabilitas terhadap ekuitas lebih besar. "
                              "DER menunjukkan utang relatif terhadap ekuitas; lebih rendah berarti ketergantungan pada utang lebih kecil. "
                              "Belum ada kontras rasio yang cukup untuk menilai keunggulan antar saham.")

    kin = context["kinerja_portofolio"]
    risk = []
    if kin.get("sharpe_ratio") is not None:
        risk.append("Sharpe membandingkan imbal hasil di atas tingkat bebas risiko dengan fluktuasi portofolio. "
                    "Pada basis dan periode sebanding, nilai lebih tinggi umumnya menunjukkan imbal hasil berlebih per fluktuasi lebih besar. "
                    "Sharpe portofolio yang dihitung adalah " + number(kin["sharpe_ratio"]) + ".")
    if kin.get("max_drawdown") is not None:
        risk.append("Dalam simulasi komposisi, penurunan terbesar (MDD) dari puncak tercatat "
                    + number(kin["max_drawdown"] * 100, "%") + ". MDD lebih kecil berarti penurunan historis lebih ringan; ini bukan batas kerugian mendatang.")
    if not risk:
        risk.append("Metrik yang tersedia belum cukup untuk menjelaskan risiko portofolio lengkap.")
    paragraphs.append("Pada tingkat portofolio, " + " ".join(risk))
    limitations = []
    if context.get("posisi_belum_tercakup"):
        limitations.append("Sebagian posisi belum tercakup dalam perhitungan risiko.")
    cvar = context.get("uji_cvar") or {}
    if cvar.get("scenario_count") is not None:
        limitations.append(f"Uji kerugian memakai {cvar['scenario_count']} skenario historis.")
        if cvar.get("downside_cvar") == 0:
            limitations.append("Kerugian ekor nol tidak berarti peluang rugi nol.")
    proof = context.get("bukti_pembanding", {})
    equal = proof.get("bobot_merata") or {}
    if proof.get("status") == "tersedia" and equal.get("layak"):
        delta = equal["selisih_fitness_hasil_dikurangi_pembanding"]
        if delta < -1e-9:
            limitations.append("Pada pengujian lokal, skor hasil lebih rendah daripada pembanding merata pada saham yang sama.")
    limitations.append("Hasil optimasi tidak menjamin kinerja mendatang dan bukan nasihat keuangan.")
    paragraphs.append(" ".join(limitations))
    return "\n\n".join(paragraphs)
