-- Jalankan sekali sebelum merilis endpoint tambah modal dan rebalance berkala.
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS pending_deposit DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS capital_contribution DOUBLE PRECISION NOT NULL DEFAULT 0;
