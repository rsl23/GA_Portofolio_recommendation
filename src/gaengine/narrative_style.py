"""Small explanation aids derived from existing facts; never GA inputs."""
from itertools import combinations
import math

NARRATIVE_VERSION = "investor_narrative_v3"

RATIOS = (
    ("PER", "per_x", -1),
    ("ROE", "roe_persen", 1),
    ("DER", "der_rasio", -1),
    ("dividend yield", "dividend_yield_persen", 1),
)


def fundamental_contrast(context):
    """Find a descriptive contrast, without ranking investment suitability.

    Prefer different dimensions and opposing relative characteristics. Negative
    PER/DER and missing values are not interpreted as cheap/low debt. Selection
    is independent of allocation size. PBV is available in the main fact table.
    """
    rows = sorted(context.get("emiten_terpilih", []), key=lambda row: row["ticker"])
    best, best_score = None, (-1, -1.0)
    for first, second in combinations(rows, 2):
        observations, directions, separation = [], set(), 0.0
        for label, key, preference in RATIOS:
            a, b = (row.get("fundamental", {}).get(key) for row in (first, second))
            if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
                continue
            if (key == "per_x" and min(a, b) <= 0) or (key != "roe_persen" and min(a, b) < 0):
                continue
            # Match the precision exposed to the model, avoiding invisible differences.
            a, b = round(a, 2), round(b, 2)
            if a == b:
                continue
            relation = "lebih tinggi" if a > b else "lebih rendah"
            observations.append({"label": label, "key": key, "a": a, "b": b, "relation": relation})
            directions.add((1 if a > b else -1) * preference)
            separation += abs(a - b) / max(abs(a), abs(b), .01)
        score = (len(observations) + (3 if len(directions) > 1 else 0), separation)
        if observations and score > best_score:
            best = {"first": first, "second": second, "observations": observations}
            best_score = score
    return best


def narrative_opening(context, contrast=None):
    contrast = contrast or fundamental_contrast(context)
    if contrast:
        first, second = contrast["first"]["ticker"], contrast["second"]["ticker"]
        if len(contrast["observations"]) == 1 and contrast["observations"][0]["key"] == "dividend_yield_persen":
            return (f"Pada data ini, perbedaan {first} dan {second} yang dapat dijelaskan "
                    "terutama terletak pada dividend yield yang tercatat.")
        per = next((row for row in contrast["observations"] if row["key"] == "per_x"), None)
        debt = next((row for row in contrast["observations"] if row["key"] == "der_rasio"), None)
        if per and debt and (per["a"] - per["b"]) * (debt["a"] - debt["b"]) < 0:
            low = first if per["a"] < per["b"] else second
            other = second if low == first else first
            return (f"Salah satu kompromi pada hasil ini terlihat pada {low}: rasio harga terhadap laba "
                    f"lebih rendah, tetapi rasio utangnya lebih tinggi dibandingkan {other}.")
        return (f"Perbedaan karakteristik {first} dan {second} menunjukkan mengapa penilaian saham "
                "perlu membaca valuasi, profitabilitas, dan utang secara bersama.")
    if not any(any(value is not None for value in (row.get("fundamental") or {}).values())
               for row in context.get("emiten_terpilih", [])):
        return "Data fundamental yang tersedia belum cukup untuk menjelaskan perbedaan kualitas saham terpilih."
    return ("Penilaian hasil ini perlu menghubungkan kualitas fundamental saham dengan risiko historis "
            "kombinasinya; satu rasio saja belum memberi gambaran yang lengkap.")
