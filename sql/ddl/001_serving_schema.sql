CREATE SCHEMA IF NOT EXISTS rees46;

CREATE TABLE IF NOT EXISTS rees46.model_runs (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    profile TEXT NOT NULL,
    git_commit TEXT,
    metrics JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS rees46.item_stats (
    product_id BIGINT PRIMARY KEY,
    category_id BIGINT,
    category_code TEXT,
    brand TEXT,
    mean_price DOUBLE PRECISION,
    events BIGINT,
    views BIGINT,
    carts BIGINT,
    purchases BIGINT,
    unique_users BIGINT,
    unique_sessions BIGINT,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    purchase_per_view DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS rees46.recommendations (
    run_id TEXT NOT NULL REFERENCES rees46.model_runs(run_id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    rank INTEGER NOT NULL,
    product_id BIGINT NOT NULL,
    score DOUBLE PRECISION,
    strategy TEXT NOT NULL,
    PRIMARY KEY (run_id, user_id, rank)
);

CREATE INDEX IF NOT EXISTS recommendations_user_idx
    ON rees46.recommendations(user_id);
