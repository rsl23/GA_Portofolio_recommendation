-- Additive, idempotent migration; application startup also handles old schemas.
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS transaction_costs DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS rebalance_metadata JSONB;
