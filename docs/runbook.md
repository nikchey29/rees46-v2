# Local runbook

## First-time finalization

```bash
uv lock
uv sync --all-extras
make quality
uv run rees46 build-gold --profile full
uv run rees46 eda --profile full
uv run rees46 run-experiments --profile full
uv run rees46 freeze-results --profile full
make quality
```

Equivalent one-command wrapper after dependency installation:

```bash
./scripts/finalize_local.sh full
```

## API

```bash
make api
curl http://localhost:8000/health
curl 'http://localhost:8000/recommend/USER_ID?k=20'
```

## PostgreSQL publication

```bash
docker compose up -d postgres
export REES46_POSTGRES_DSN='postgresql://rees46:rees46@localhost:5432/rees46'
uv run rees46 publish-postgres --profile full --dsn "$REES46_POSTGRES_DSN"
```
