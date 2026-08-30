# Local Development Setup

This guide covers setting up the Acme Cloud web application and API for local
development. It assumes you already have access to the `acme-cloud/platform`
repository and a company GitHub account with SSO enabled.

## Prerequisites

* Docker Desktop 4.x or newer
* Python 3.11+
* Node.js 20 LTS
* PostgreSQL client tools (`psql`)
* An Acme Cloud engineering account with membership in `platform-dev`

## Clone and bootstrap

1. Clone the repository:

   ```bash
   git clone git@github.com:acme-cloud/platform.git
   cd platform
   ```

2. Copy the example environment file and fill in local secrets:

   ```bash
   cp .env.example .env.local
   ```

3. Start the local dependency stack (PostgreSQL, Redis, and a mock webhook
   receiver):

   ```bash
   docker compose up -d postgres redis webhook-mock
   ```

4. Install dependencies and run database migrations:

   ```bash
   make install
   make migrate
   ```

5. Start the API and web application:

   ```bash
   make dev
   ```

## Local configuration notes

Local development uses relaxed defaults so engineers are not blocked by
production-grade limits. Notably:

* `AUTH_TOKEN_TTL` defaults to `3600` (60 minutes) locally, matching
  production, so token-expiry bugs surface early.
* `DATABASE_POOL_SIZE` defaults to `5` locally (much smaller than the
  production default of 20) since a single developer rarely needs more than
  a handful of concurrent connections.
* `DATABASE_POOL_TIMEOUT` defaults to `10` seconds locally rather than the
  production default of 30, so a misconfigured local pool fails fast.
* Local webhook deliveries are sent to the `webhook-mock` container instead
  of a real endpoint.

## Running tests

```bash
make test
```

Unit tests do not require Docker. Integration tests require the local
PostgreSQL container to be running, since they exercise real connection-pool
behavior (including simulating `ERR_DB_1042` under artificial load).

## Troubleshooting

If `make dev` fails with a database connection error, confirm the
`postgres` container is healthy with `docker compose ps` before filing a
ticket in `#eng-platform`.
