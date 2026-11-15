-- Initialize existing series once; future generations preserve user choices.
ALTER TABLE portofolios ADD COLUMN IF NOT EXISTS is_list_active BOOLEAN;
WITH RECURSIVE current_chains AS (
    SELECT id, parent_portofolio_id FROM portofolios
    WHERE status_portofolio IN ('active', 'active_backtest')
    UNION
    SELECT p.id, p.parent_portofolio_id FROM portofolios p
    JOIN current_chains c ON p.id = c.parent_portofolio_id
)
UPDATE portofolios SET is_list_active = id IN (
    SELECT id FROM current_chains WHERE parent_portofolio_id IS NULL
)
WHERE is_list_active IS NULL AND parent_portofolio_id IS NULL;
