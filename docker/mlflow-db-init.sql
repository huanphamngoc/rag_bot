-- Role + database for the MLflow tracking server, inside this project's vector Postgres.
-- Run by the one-shot compose service "mlflow-db-init" before every start of "mlflow"; idempotent:
--   psql -v ON_ERROR_STOP=1 -v mlflow_password=... -f mlflow-db-init.sql
-- MLflow creates and migrates its own tables (alembic) on startup. Its database is separate from the
-- rag/ingest/meta schemas and from the pgvector extension; nothing it stores is queried by this app.

SELECT format('CREATE ROLE mlflow LOGIN PASSWORD %L', :'mlflow_password')
 WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow')
\gexec

-- keeps the role in step with MLFLOW_DB_PASSWORD in .env when that is changed
SELECT format('ALTER ROLE mlflow WITH LOGIN PASSWORD %L', :'mlflow_password')
\gexec

SELECT 'CREATE DATABASE mlflow OWNER mlflow'
 WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')
\gexec

REVOKE ALL ON DATABASE mlflow FROM PUBLIC;
GRANT CONNECT, TEMPORARY ON DATABASE mlflow TO mlflow;
