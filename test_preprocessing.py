import os
import pandas as pd
from src.preprocessing.stock_filtering import run_live_preprocessing

def test_filtering():
    print("========================================")
    print("🚀 MENJALANKAN TESTING LIVE PREPROCESSING")
    print("========================================")
    
    try:
        # Jalankan pipeline preprocessing yang sudah kita buat
        daftar_saham, df_lolos = run_live_preprocessing()
        
        print("\n========================================")
        print("✅ PROSES SELESAI!")
        print("========================================")
        print(f"Total Saham Lolos Filter : {len(daftar_saham)} Saham")
        print(f"Daftar Ticker            : {daftar_saham}")
        
        # Simpan hasilnya ke file Excel agar mudah dibaca dan divalidasi
        file_excel = "hasil_filtering_saham.xlsx"
        
        # Rapikan DataFrame sebelum di-export (tambahkan kolom Index/Kode ke dalam Excel)
        df_export = df_lolos.copy()
        
        # Export ke excel
        df_export.to_excel(file_excel, index=True)
        
        print(f"\n📂 File Excel berhasil dibuat: {os.path.abspath(file_excel)}")
        print("Silakan buka file tersebut untuk melihat detail fundamental & teknikal tiap saham.")
        
    except Exception as e:
        print("\n❌ TERJADI KESALAHAN SAAT TESTING:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_filtering()

