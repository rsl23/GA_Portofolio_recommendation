-- Pending live plans are isolated from executed holdings and performance.
-- The application generates UUIDs; no database extension is required.
CREATE TABLE IF NOT EXISTS portfolio_proposals (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL,
    payload JSON NOT NULL,
    portfolio_id UUID REFERENCES portofolios(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_portfolio_proposals_user_id ON portfolio_proposals(user_id);
