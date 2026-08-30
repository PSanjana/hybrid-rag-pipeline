# Production Access Control Policy

This policy defines who may access Acme Cloud production systems, and
under what conditions. It applies to all employees and contractors, and is
owned by the security team.

## Core requirements

* **Multi-factor authentication (MFA) is required for all human access to
  production systems.** This includes the production database, production
  infrastructure consoles, and any administrative API endpoints. Password
  authentication alone is never sufficient for production access.
* **Engineers must be a member of the `production-engineers` access
  group** to obtain production credentials of any kind. Group membership
  is not automatic upon hire — it is requested and approved individually.
* Service accounts and automated tooling are exempt from the MFA
  requirement (MFA is a human-operator control) but are subject to
  stricter secret-rotation and scoping requirements instead.

## Requesting access

1. Open an access request in the internal IT portal, specifying the system
   and the business justification.
2. Your manager approves the request.
3. A member of the security team reviews and grants `production-engineers`
   group membership.
4. You will be prompted to enroll an MFA device (hardware key or
   authenticator app) if you have not already done so.

Access requests are typically resolved within one business day. Emergency
access during an active incident can be expedited by the incident
commander — see `incident-response.md`.

## Review cadence

Privileged production access is **reviewed quarterly**. During each
review:

* Security confirms each member of `production-engineers` still requires
  access based on current role.
* Access that has gone unused for 90 days or more is flagged for removal.
* Managers are asked to confirm continued need for each direct report with
  standing access.

Access is also reviewed, outside the normal quarterly cadence, immediately
upon role change or offboarding.

## Credential and session hygiene

Human operators authenticating to production APIs receive short-lived
access tokens rather than long-lived credentials, consistent with the
token lifetimes described in `authentication-api.md`. Tokens must never be
shared between individuals, stored in plaintext configuration files, or
committed to source control. Any suspected credential exposure must be
reported to `#security-oncall` immediately, regardless of time of day.

## Least privilege

Production access should be scoped to the minimum systems and permissions
required for an individual's role. Broad, standing "just in case" access
is discouraged; temporary elevated access for a specific task should be
requested and time-boxed instead where the tooling supports it.

## Enforcement

Access granted outside this process (for example, through informal
credential sharing) is a policy violation and will be treated as a
security incident.
