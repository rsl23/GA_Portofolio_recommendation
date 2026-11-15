-- Tambahkan identitas portofolio pada instalasi lama sebelum menjalankan versi baru.
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS name_portofolio VARCHAR(80);
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS description TEXT;
