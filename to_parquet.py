import pandas as pd
import re
from pathlib import Path

# ============================================================
# Excel -> Parquet Transformer for Historical Fundamental Data
#
# Expected Excel layout:
#   Row 1: Quarter labels, e.g. Q1 2026, Q4 2025, Q3 2025, ...
#   Column A: Ticker
#   Column B: Key Ratio / Metric
#   Columns C onward: Values for each quarter
#
# Output:
#   Ticker | Fiscal_Period | Report_Date | Metric | Value | Unit
#
# IMPORTANT:
# - This script can preserve Fiscal_Period such as "Q1 2026".
# - Report_Date cannot be inferred safely from "Q1 2026" alone.
#   If your Excel does not contain report/publication dates, the
#   script leaves Report_Date as missing (NaT).
# - For point-in-time backtesting, Report_Date should eventually
#   be populated from the actual publication/availability date.
# ============================================================

INPUT_FILE = "data/Kuartalan_lengkap.xlsx"
OUTPUT_FILE = "data/fundamental_quarterly.parquet"

# Excel settings
HEADER_ROW = 0       # Row 1 in Excel (0-based)
TICKER_COL = 0       # Column A
METRIC_COL = 1       # Column B
FIRST_PERIOD_COL = 2 # Column C onward


def parse_period(value):
    """Convert quarter labels into standardized 'Qx YYYY' strings."""
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    # Handles: Q1 2026, Q1-2026, Q1/2026, Q1_2026
    match = re.search(r"\b(Q[1-4])[\s\-_\/]*(20\d{2})\b", text)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    # Handles: 2026 Q1
    match = re.search(r"\b(20\d{2})[\s\-_\/]*(Q[1-4])\b", text)
    if match:
        return f"{match.group(2)} {match.group(1)}"

    return None


def parse_value(value):
    """
    Convert Excel cell values to numeric values where possible.

    Examples:
      12.5        -> 12.5
      '12.5%'     -> 12.5
      '1,234'     -> 1234
      '1,532 B'   -> 1532000000000.0   (B = miliar -> x 1e9)
      '(48.16 B)' -> -48160000000.0    (kurung = negatif)
      '-'         -> NaN
      blank       -> NaN

    NOTE:
    Percent values are stored as their numeric percentage representation.
    For example, 3.2% becomes 3.2, not 0.032.

    Suffix 'B' (miliar/billion) dikalikan 1_000_000_000 agar nilai tersimpan
    sebagai angka rupiah penuh, bukan string. Berlaku juga di dalam tanda
    kurung negatif: '(2,756 B)' -> -2_756_000_000_000.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if text == "" or text in {"-", "—", "N/A", "NA", "NULL", "NONE"}:
        return None

    # Parentheses convention: (123) = -123, also '(2,756 B)' = -2756 miliar
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    # Remove common formatting (thousand separators, percent sign)
    text = text.replace(",", "").replace("%", "").strip()

    # Magnitude suffix 'B' (miliar / billion) -> multiply by 1e9
    # e.g. '1,532 B' -> 1532000000000, '(48.16 B)' -> -48160000000
    multiplier = 1.0
    if re.fullmatch(r"-?[\d.]+\s*[Bb]", text):
        multiplier = 1_000_000_000
        text = text[:-1].strip()

    try:
        result = float(text) * multiplier
    except ValueError:
        # Keep non-numeric values rather than silently destroying data.
        return text

    return -result if negative else result


def infer_unit(metric):
    """
    Optional simple unit inference.
    This is only a helper; review the result for your actual dataset.
    """
    if pd.isna(metric):
        return None

    m = str(metric).lower()

    percentage_keywords = [
        "margin", "margin %", "growth", "growth %", "roe", "roa",
        "roce", "payout", "yield", "rate"
    ]

    for keyword in percentage_keywords:
        if keyword in m:
            return "%"

    if "eps" in m or "bvps" in m or "price" in m:
        return "IDR"

    if "share outstanding" in m or "shares outstanding" in m:
        return "shares"

    return None


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file tidak ditemukan: {input_path.resolve()}"
        )

    # Read Excel without assuming a normal table header.
    raw = pd.read_excel(input_path, header=None)

    if raw.shape[1] <= FIRST_PERIOD_COL:
        raise ValueError(
            "Jumlah kolom Excel tidak sesuai. "
            "Pastikan periode dimulai dari kolom C."
        )

    # --------------------------------------------------------
    # 1. Detect quarter columns from the first row
    # --------------------------------------------------------
    periods = {}

    for col in range(FIRST_PERIOD_COL, raw.shape[1]):
        period = parse_period(raw.iloc[HEADER_ROW, col])

        if period:
            periods[col] = period

    if not periods:
        raise ValueError(
            "Tidak menemukan periode seperti 'Q1 2026' pada baris pertama."
        )

    print("Periode yang ditemukan:")
    for col, period in periods.items():
        print(f"  Column {col + 1}: {period}")

    # --------------------------------------------------------
    # 2. Transform wide -> long
    # --------------------------------------------------------
    records = []

    for row in range(HEADER_ROW + 1, raw.shape[0]):
        ticker = raw.iloc[row, TICKER_COL]
        metric = raw.iloc[row, METRIC_COL]

        if pd.isna(ticker) or pd.isna(metric):
            continue

        ticker = str(ticker).strip().upper()
        metric = str(metric).strip()

        # Skip possible header rows
        if ticker in {"TICKER", "CODE", "STOCK", "EMITEN", "Kode"}:
            continue

        if metric.lower() in {
            "key ratio", "key ratios", "metric", "ratio", "ratios", "key ratio (rp)"
        }:
            continue

        if ticker == "" or metric == "":
            continue

        unit = infer_unit(metric)

        for col, period in periods.items():
            cell_value = raw.iloc[row, col]
            parsed_value = parse_value(cell_value)

            # Keep only cells that actually contain data.
            # If you want explicit NaN rows for every combination,
            # remove this condition.
            if parsed_value is None:
                continue

            records.append({
                "Ticker": ticker,
                "Fiscal_Period": period,
                "Report_Date": pd.NaT,
                "Metric": metric,
                "Value": parsed_value,
                "Unit": unit,
            })

    if not records:
        raise ValueError("Tidak ada data fundamental yang berhasil ditransformasi.")

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # 3. Clean types
    # --------------------------------------------------------
    df["Ticker"] = df["Ticker"].astype("string")
    df["Fiscal_Period"] = df["Fiscal_Period"].astype("string")
    df["Metric"] = df["Metric"].astype("string")
    df["Unit"] = df["Unit"].astype("string")
    df["Report_Date"] = pd.to_datetime(df["Report_Date"], errors="coerce")

    # Value: pastikan seluruhnya numerik. Kolom campuran float + string
    # (mis. salah ketik seperti 'd') membuat pyarrow gagal menulis parquet,
    # jadi nilai yang tetap non-numerik dilaporkan dulu, lalu di-set NaN
    # (tidak dihapus secara diam-diam).
    value_numeric = pd.to_numeric(df["Value"], errors="coerce")

    bad_mask = value_numeric.isna() & df["Value"].notna()
    if bad_mask.any():
        bad = df.loc[bad_mask, ["Ticker", "Fiscal_Period", "Metric", "Value"]]
        print(f"\nPERINGATAN: {len(bad)} nilai non-numerik tidak bisa "
              f"dikonversi dan di-set NaN:")
        print(bad.to_string(index=False))

    df["Value"] = value_numeric

    # --------------------------------------------------------
    # 4. Sort
    # --------------------------------------------------------
    quarter_order = (
        df["Fiscal_Period"]
        .str.extract(r"Q([1-4])\s+(20\d{2})")
    )

    df["_quarter"] = pd.to_numeric(quarter_order[0], errors="coerce")
    df["_year"] = pd.to_numeric(quarter_order[1], errors="coerce")

    df = (
        df.sort_values(
            ["Ticker", "_year", "_quarter", "Metric"],
            ascending=[True, True, True, True]
        )
        .drop(columns=["_year", "_quarter"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 5. Save Parquet
    # --------------------------------------------------------
    try:
        df.to_parquet(output_path, index=False, engine="pyarrow")
    except ImportError:
        raise ImportError(
            "PyArrow belum terinstall. Jalankan:\n"
            "pip install pyarrow"
        )

    # --------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------
    print("\nTransformasi berhasil.")
    print(f"Input : {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Rows  : {len(df):,}")
    print(f"Tickers: {df['Ticker'].nunique():,}")
    print(f"Metrics: {df['Metric'].nunique():,}")
    print(f"Periods: {df['Fiscal_Period'].nunique():,}")

    print("\nPreview:")
    print(df.head(20).to_string(index=False))

    print(
        "\nCATATAN PENTING:\n"
        "Report_Date sengaja dikosongkan karena tanggal publikasi laporan\n"
        "tidak dapat diketahui hanya dari label Q1/Q2/Q3/Q4. Untuk backtesting\n"
        "point-in-time, isi Report_Date dengan tanggal ketika data tersebut\n"
        "benar-benar tersedia bagi investor."
    )


if __name__ == "__main__":
    main()
