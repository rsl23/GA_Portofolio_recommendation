import pandas as pd
from pathlib import Path

# ============================================================
# Excel -> Parquet Transformer for IDX (Indonesia Stock Exchange)
# Activities Data (Idx_Activities.xlsx)
#
# Struktur Excel:
#   Kolom B (index 1) berisi "Year / Month" dengan format:
#     - Baris tahun, contoh: "2006" -> data agregat 1 tahun
#     - Baris bulan berikutnya, contoh: "1", "2", ..., "12"
#       -> detail per bulan untuk tahun pada baris di atasnya
#   Kolom C dst: USD Rate, Total Trading (Volume/Value/Freq),
#   Average Daily Trading (Volume/Value/Freq), days, JCI,
#   Market Cap, Listed Companies, Listed Shares.
#
# Output parquet (struktur tidy, per baris waktu):
#   Period_Type | Year | Month | Period_End_Date | USD_Rate | ...
# - Baris tahun: Period_Type = "Year", Month = NaN
# - Baris bulan: Period_Type = "Month", Month = 1..12
#   (Year diambil dari baris tahun terakhir di atasnya)
# ============================================================

INPUT_FILE = "data/Idx_Activities.xlsx"
OUTPUT_FILE = "data/idx_activities.parquet"

# Excel settings (0-based)
HEADER_ROW_1 = 3   # baris header pertama  ("Year / Month", "USD Rate", ...)
HEADER_ROW_2 = 4   # baris header kedua    ("Volume, m.shares", ...)
FIRST_DATA_ROW = 5 # baris data pertama    ("2006")

COL_LABEL = 1   # Kolom B: Year / Month
DATA_COLS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

OUTPUT_COLUMNS = [
    "Period_Type",
    "Year",
    "Month",
    "Period_End_Date",
    "USD_Rate",
    "Total_Trading_Volume_mshares",
    "Total_Trading_Value_bIDR",
    "Total_Trading_Freq_thx",
    "Avg_Daily_Trading_Volume_mshares",
    "Avg_Daily_Trading_Value_bIDR",
    "Avg_Daily_Trading_Freq_thx",
    "Trading_Days",
    "JCI",
    "Market_Cap_bIDR",
    "Listed_Companies",
    "Listed_Shares_mshares",
]


def build_flat_headers(raw: pd.DataFrame) -> list[str]:
    """
    Kembalikan nama kolom output berdasarkan posisi kolom Excel.

    CATATAN: header Excel memakai merged cells ("Total Trading" dan
    "Average Daily Trading" hanya terbaca di kolom pertama grupnya),
    sehingga pemetaan dilakukan secara eksplisit per posisi kolom:
      C: USD Rate
      D-F: Total Trading (Volume m.shares / Value b.IDR / Freq th.x)
      G-I: Average Daily Trading (Volume / Value / Freq)
      J: days, K: JCI, L: Market Cap b.IDR,
      M: Listed Comp., N: Listed Shares m.shares
    """
    # Verifikasi posisi kolom sesuai harapan (header baris pertama & kedua)
    expected_top = {2: "USD Rate", 3: "Total Trading", 6: "Average Daily Trading",
                    9: "days", 10: "JCI", 11: "Market Capt., b.IDR",
                    12: "Listed Comp.", 13: "Listed Shares, m.shares"}
    for idx, expected in expected_top.items():
        actual = str(raw.iloc[HEADER_ROW_1, idx]).strip()
        if actual != expected:
            raise ValueError(
                f"Struktur Excel tidak sesuai: kolom {chr(65 + idx)} baris "
                f"{HEADER_ROW_1 + 1} berisi '{actual}', diharapkan '{expected}'. "
                f"Periksa kembali file input."
            )

    return [
        "USD_Rate",                      # C
        "Total_Trading_Volume_mshares",  # D
        "Total_Trading_Value_bIDR",      # E
        "Total_Trading_Freq_thx",        # F
        "Avg_Daily_Trading_Volume_mshares",  # G
        "Avg_Daily_Trading_Value_bIDR",      # H
        "Avg_Daily_Trading_Freq_thx",        # I
        "Trading_Days",                  # J
        "JCI",                           # K
        "Market_Cap_bIDR",               # L
        "Listed_Companies",              # M
        "Listed_Shares_mshares",         # N
    ]


def to_number(value):
    """Konversi nilai Excel ke numerik; '-' / kosong -> NaN."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "—", "N/A", "NA"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")

def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path.resolve()}")

    # --------------------------------------------------------
    # 1. Baca Excel tanpa header
    # --------------------------------------------------------
    raw = pd.read_excel(input_path, sheet_name=0, header=None)
    print(f"Membaca {input_path.resolve()} -> shape {raw.shape}")

    col_names = build_flat_headers(raw)
    print("\nMapping kolom:")
    for idx, name in zip(DATA_COLS, col_names):
        print(f"  Kolom {chr(65 + idx)}: {name}")

    # --------------------------------------------------------
    # 2. Interpretasi kolom B (Year / Month) dan bangun records
    # --------------------------------------------------------
    records = []
    current_year = None

    for row in range(FIRST_DATA_ROW, len(raw)):
        label = raw.iloc[row, COL_LABEL]

        # Lewati baris tanpa label di kolom B
        if pd.isna(label):
            continue

        try:
            label_num = int(float(str(label).strip()))
        except ValueError:
            continue

        if 1900 <= label_num <= 2100:
            # --- Baris TAHUN ---
            current_year = label_num
            values = [to_number(raw.iloc[row, c]) for c in DATA_COLS]
            record = {
                "Period_Type": "Year",
                "Year": current_year,
                "Month": pd.NA,
                "Period_End_Date": pd.Timestamp(current_year, 12, 31),
            }
            record.update(dict(zip(col_names, values)))
            records.append(record)
        elif 1 <= label_num <= 12:
            # --- Baris BULAN (tahun merujuk baris tahun di atasnya) ---
            if current_year is None:
                raise ValueError(
                    f"Baris {row + 1}: label bulan '{label_num}' muncul "
                    f"sebelum baris tahun pertama."
                )
            values = [to_number(raw.iloc[row, c]) for c in DATA_COLS]
            month = label_num
            # Period_End_Date = hari terakhir bulan tersebut
            if month == 12:
                period_end = pd.Timestamp(current_year, 12, 31)
            else:
                period_end = pd.Timestamp(current_year, month + 1, 1) - pd.Timedelta(days=1)
            record = {
                "Period_Type": "Month",
                "Year": current_year,
                "Month": month,
                "Period_End_Date": period_end,
            }
            record.update(dict(zip(col_names, values)))
            records.append(record)
        else:
            # Label lain di luar rentang tahun/bulan -> abaikan
            print(f"  PERINGATAN: label '{label}' di baris {row + 1} diabaikan.")

    if not records:
        raise ValueError("Tidak ada data yang berhasil ditransformasi.")

    # --------------------------------------------------------
    # 3. Buat DataFrame, rapikan tipe data
    # --------------------------------------------------------
    df = pd.DataFrame(records)

    # Pastikan semua kolom output ada (walau kosong) dan urut
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OUTPUT_COLUMNS]

    df["Period_Type"] = df["Period_Type"].astype("string")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df["Period_End_Date"] = pd.to_datetime(df["Period_End_Date"])

    value_cols = [c for c in OUTPUT_COLUMNS if c not in
                  {"Period_Type", "Year", "Month", "Period_End_Date"}]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # 4. Urutkan kronologis (tahun lalu bulan)
    # --------------------------------------------------------
    df = df.sort_values(
        ["Year", "Month"],
        na_position="first",  # baris tahun (Month NaN) muncul sebelum bulannya
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 5. Simpan Parquet
    # --------------------------------------------------------
    try:
        df.to_parquet(output_path, index=False, engine="pyarrow")
    except ImportError:
        raise ImportError("PyArrow belum terinstall. Jalankan:\npip install pyarrow")

    # --------------------------------------------------------
    # 6. Ringkasan
    # --------------------------------------------------------
    n_year = (df["Period_Type"] == "Year").sum()
    n_month = (df["Period_Type"] == "Month").sum()

    print("\nTransformasi berhasil.")
    print(f"Input : {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Rows  : {len(df):,} (Year: {n_year}, Month: {n_month})")
    print(f"Rentang: {int(df['Year'].min())} - {int(df['Year'].max())}")

    print("\nPreview (10 baris pertama, kolom kunci):")
    print(df[["Period_Type", "Year", "Month", "USD_Rate", "JCI",
              "Market_Cap_bIDR"]].head(10).to_string(index=False))

    # Sanity check: total tahun = jumlah 12 bulannya
    print("\nSanity check (Total Trading Value: tahun vs jumlah 12 bulan):")
    for year in sorted(df.loc[df["Period_Type"] == "Year", "Year"].unique()):
        y_val = df.loc[(df["Period_Type"] == "Year") & (df["Year"] == year),
                       "Total_Trading_Value_bIDR"].iloc[0]
        m_sum = df.loc[(df["Period_Type"] == "Month") & (df["Year"] == year),
                       "Total_Trading_Value_bIDR"].sum()
        print(f"  {year}: year={y_val:,.2f} | sum(months)={m_sum:,.2f} "
              f"| diff={abs(y_val - m_sum):,.2f}")


if __name__ == "__main__":
    main()
