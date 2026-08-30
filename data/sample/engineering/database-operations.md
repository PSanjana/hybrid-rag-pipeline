# Database Operations Guide

Acme Cloud's production data layer runs on **PostgreSQL**. This guide covers
connection-pool configuration, common failure modes, and routine
maintenance tasks owned by the platform engineering team.

## Connection pooling

Each application service maintains its own connection pool to PostgreSQL,
managed through PgBouncer in transaction-pooling mode. Two configuration
values control pool behavior:

* `DATABASE_POOL_SIZE` — the maximum number of connections a single service
  instance may hold open at once. The default in production is **20**.
* `DATABASE_POOL_TIMEOUT` — how long, in seconds, a request will wait for a
  connection to become available before giving up. The default in
  production is **30 seconds**.

Example configuration:

```yaml
database:
  pool_size: 20
  pool_timeout: 30
  host: prod-pg-primary.internal
  port: 5432
```

Increasing `DATABASE_POOL_SIZE` without also reviewing PostgreSQL's
`max_connections` setting can starve other services, since the primary has a
shared connection budget across all pools.

## ERR_DB_1042: pool exhaustion

`ERR_DB_1042` is returned by the application when it cannot obtain a
connection from the database pool before `DATABASE_POOL_TIMEOUT` elapses.
This almost always means every connection in the pool is already checked
out — either because of a genuine traffic spike, a slow query holding
connections open longer than expected, or a leaked connection from a code
path that doesn't release its connection back to the pool.

### Diagnosing pool exhaustion

1. Check the `db_pool_in_use` metric on the service dashboard. If it is
   pinned at `DATABASE_POOL_SIZE` for a sustained period, the pool is
   exhausted.
2. Look for long-running queries:

   ```sql
   SELECT pid, now() - query_start AS duration, query
   FROM pg_stat_activity
   WHERE state = 'active'
   ORDER BY duration DESC
   LIMIT 20;
   ```

3. If a small number of queries are holding connections for an unusually
   long time, consider terminating them:

   ```sql
   SELECT pg_terminate_backend(pid);
   ```

4. If exhaustion is caused by legitimate load rather than a leak or slow
   query, consider a temporary, reviewed increase to `DATABASE_POOL_SIZE`
   during a deployment (see `deployment-guide.md`).

### Common causes

* A newly deployed code path that opens a transaction but does not commit
  or roll it back on an error path.
* A batch job running outside its expected maintenance window and
  competing with request traffic for pool capacity.
* An underlying PostgreSQL primary failover, during which pooled
  connections briefly become invalid and reconnect storms can temporarily
  exhaust the pool.

## Backups

PostgreSQL full backups run **daily**, and point-in-time recovery (PITR)
data is retained for **7 days**. Backup and restore procedures are detailed
in `backup-recovery.html`; production restores require sign-off from the
incident commander when performed during an active incident.

## Schema migrations

Migrations run automatically as part of the deployment pipeline for
backward-compatible changes. Migrations that are not backward compatible
(column drops, type changes, `NOT NULL` additions without a default) must be
split into multiple deploys and reviewed by a database owner before merge.

## Ownership

The `platform-data` team owns PostgreSQL infrastructure, connection-pool
defaults, and backup policy. File operational questions in `#eng-database`.
