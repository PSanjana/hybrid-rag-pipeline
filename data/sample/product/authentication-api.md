# Authentication API

The Acme Cloud API uses OAuth 2.0-style bearer tokens for authentication.
This document describes token lifetimes, the refresh flow, and
production-specific requirements for human operators.

## Token lifetimes

* **Access tokens expire after 60 minutes.** Every request must include a
  valid, unexpired access token in the `Authorization: Bearer <token>`
  header.
* **Refresh tokens expire after 30 days.** A refresh token can be exchanged
  for a new access token without requiring the user to log in again.
* The access-token lifetime is controlled by the `AUTH_TOKEN_TTL`
  configuration value, expressed in seconds. In production this is set to
  `3600`, corresponding to the 60-minute lifetime above.

## Requesting a token

```bash
curl -X POST https://api.acmecloud.com/v1/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "user@example.com",
    "password": "••••••••"
  }'
```

A successful response returns both an `access_token` and a
`refresh_token`:

```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "rtok_9f2c...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

## Refreshing a token

```bash
curl -X POST https://api.acmecloud.com/v1/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "refresh_token",
    "refresh_token": "rtok_9f2c..."
  }'
```

## Expired or invalid tokens

If a request is made with an expired or otherwise invalid access token, the
API returns:

```json
{
  "error": "ERR_AUTH_4017",
  "message": "Access token is expired or invalid."
}
```

with an HTTP `401 Unauthorized` status. Clients should attempt a token
refresh and retry the original request once before surfacing an error to
the end user.

## Multi-factor authentication for production access

Human operators authenticating against **production** API endpoints —
as opposed to end-user traffic through the customer-facing application —
must complete multi-factor authentication (MFA) in addition to normal
credentials. This applies to engineers, support staff with elevated access,
and any automated tooling acting on a human operator's behalf. Service
accounts used for machine-to-machine traffic are not subject to the MFA
requirement, but are subject to tighter scoping and rotation policies
managed separately by the security team.

See `access-control-policy.md` for the broader production access-control
requirements human operators must satisfy beyond authentication alone.
