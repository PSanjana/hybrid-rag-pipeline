# Incident Response

This document describes how Acme Cloud detects, responds to, and closes out
production incidents.

## Severity levels

* **SEV1** — Full outage or major data-integrity risk affecting most
  customers. Immediate all-hands response.
* **SEV2** — Significant degradation affecting a meaningful subset of
  customers (e.g., elevated `ERR_DB_1042` rates, widespread
  `ERR_WEBHOOK_5003` failures). Response within 15 minutes.
* **SEV3** — Limited-impact issue affecting a small number of customers or
  a non-critical system. Response within one business day.

## Declaring an incident

Any engineer can declare an incident by running `/incident declare` in
Slack, which pages the on-call rotation and creates an incident channel.
When in doubt, declare — it is cheap to stand down a false alarm and
expensive to delay a real one.

## The incident commander

Every SEV1 and SEV2 incident has a designated **incident commander (IC)**,
assigned automatically to the first senior on-call engineer to acknowledge
the page unless explicitly reassigned. The incident commander:

* Owns overall coordination and communication for the incident.
* Decides when to bring in additional responders or escalate severity.
* Has the authority to approve an **emergency deployment that bypasses the
  `DEPLOY_FREEZE` window**, when a fix cannot reasonably wait until the
  freeze lifts. This approval must be recorded in the incident channel and
  referenced in the deployment description.
* Decides when the incident is resolved and hands off follow-up work.
* Approves emergency production database restores performed during an
  active incident (see `backup-recovery.html`).

Engineers should not bypass the deployment freeze on their own judgment
during an incident — bypass authority sits specifically with the incident
commander, precisely so that a stressful moment doesn't lead to an
under-reviewed change.

## Common incident triggers

Several classes of incidents recur often enough to have documented
playbooks:

* Sustained `ERR_DB_1042` responses, usually caused by connection-pool
  exhaustion — see `database-operations.md` for diagnosis steps.
* Elevated `ERR_WEBHOOK_5003` rates affecting many customer endpoints at
  once, which more often points at an Acme Cloud-side delivery problem
  than a customer-side one — see `production-runbook.txt`.
* Authentication failures at unusual volume, which can indicate a signing
  key rotation issue rather than a wave of individually expired tokens.

## Communication

Customer-facing status updates are posted to the public status page by the
incident commander or their delegate. Internal updates are posted to the
incident Slack channel at least every 30 minutes during an active SEV1 or
SEV2, even if the update is "no change."

## Postmortems

Every SEV1 and SEV2 incident requires a written postmortem within 5
business days of resolution. Postmortems are blameless: the goal is
identifying systemic gaps, not individual fault. Postmortems are reviewed
in the weekly engineering sync and action items are tracked to completion.

## After the incident

Once an incident is resolved:

1. The incident commander confirms customer-facing systems are healthy.
2. Any emergency changes made during the incident (including
   freeze-bypassing deploys) are reviewed for follow-up hardening.
3. A postmortem draft is opened within 2 business days.
