# Production Deployment Guide

This document describes how Acme Cloud engineers deploy the web application
and API to production. It applies to all services under the `platform`
umbrella (`api`, `web`, `worker`, and `webhook-dispatcher`).

## Deployment strategy

Production deployments use a **rolling deployment** strategy. New instances
are brought up behind the load balancer and health-checked before old
instances are drained and terminated. This avoids downtime for standard
releases and allows an in-progress rollout to be paused if error rates spike.

Rolling deployments are triggered from the `main` branch via the
`deploy-production` GitHub Actions workflow. A deployment typically takes
between 8 and 15 minutes to fully roll out across all regions.

## The deployment freeze window

Routine production deployments are **prohibited every Friday from 18:00 to
23:59 UTC**. This restriction is tracked internally as `DEPLOY_FREEZE` and is
enforced automatically by the deployment pipeline — the `deploy-production`
workflow will refuse to run during the freeze window unless an override flag
is supplied.

The freeze exists to avoid introducing regressions right before the weekend,
when on-call coverage is reduced.

### Emergency bypass

An emergency fix may bypass `DEPLOY_FREEZE` **only with sign-off from the
current incident commander**. To bypass the freeze:

1. Open or reference an active incident in the incident-management system.
2. Ask the incident commander to approve the bypass in the incident channel.
3. Re-run the `deploy-production` workflow with the `--force-through-freeze`
   flag, citing the incident ID in the deployment description.

Deploys that bypass the freeze without incident-commander approval are
treated as a policy violation and are reviewed by the engineering leadership
team.

## Standard deployment steps

1. Confirm CI is green on `main`.
2. Confirm you are not inside the `DEPLOY_FREEZE` window (or have an approved
   bypass).
3. Run the deployment workflow:

   ```bash
   gh workflow run deploy-production.yml -f service=api
   ```

4. Watch the rollout dashboard for elevated error rates or latency.
5. Once the rollout completes, verify the release in the `#deploys` channel.

## Rollback

If a deployment introduces a regression, roll back with:

```bash
gh workflow run rollback-production.yml -f service=api -f to=previous
```

Rollbacks are exempt from `DEPLOY_FREEZE` restrictions, since they reduce
risk rather than introduce it.

## Configuration changes during deployment

Some deploys also change runtime configuration. Common configuration values
adjusted during deploys include `DATABASE_POOL_SIZE`, `DATABASE_POOL_TIMEOUT`,
`AUTH_TOKEN_TTL`, and `MAX_WEBHOOK_RETRIES`. Configuration changes to these
values should be called out explicitly in the deployment description, since
they can affect error rates for `ERR_DB_1042` and `ERR_WEBHOOK_5003` if
misconfigured.

## Related documents

See `database-operations.md` for connection-pool tuning guidance and
`incident-response.md` for how incident commanders are assigned and what
authority they hold during an active incident.
