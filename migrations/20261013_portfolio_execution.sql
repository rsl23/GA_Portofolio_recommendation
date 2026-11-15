-- Apply once before deploying the accounting code. Existing snapshots remain legacy.
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS execution_date DATE;
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS execution_prices JSONB;
