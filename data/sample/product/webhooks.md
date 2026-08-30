# Webhooks

Acme Cloud can notify your systems of events (subscription changes,
usage-threshold crossings, invoice events) by sending an HTTP `POST`
request to an endpoint you configure.

## Delivery and retries

Webhook delivery is **retried up to 5 times** if your endpoint does not
respond with a `2xx` status code, or does not respond within 10 seconds.
The maximum retry count is controlled internally by the
`MAX_WEBHOOK_RETRIES` configuration value, which defaults to `5` in
production.

Retry intervals use **exponential backoff**: roughly 30 seconds, 2 minutes,
10 minutes, 30 minutes, and finally 2 hours after the initial attempt,
before the delivery is marked as permanently failed.

If all retry attempts are exhausted without a successful delivery, the
event is marked as failed and surfaces internally as `ERR_WEBHOOK_5003`.
Failed events remain visible in the webhook delivery log for 30 days and
can be manually redelivered from the dashboard once your endpoint is
fixed.

## Configuring an endpoint

1. Go to **Settings → Webhooks** in the Acme Cloud dashboard.
2. Click **Add endpoint** and enter your HTTPS URL.
3. Select which event types to subscribe to.
4. Copy the generated signing secret — you'll need it to verify payloads.

## Verifying webhook signatures

Every delivery includes an `Acme-Signature` header computed as an HMAC-SHA256
of the raw request body using your endpoint's signing secret:

```python
import hashlib
import hmac

def verify_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)
```

Reject any request whose computed signature does not match, and always use
a constant-time comparison to avoid timing attacks.

## Example payload

```json
{
  "event": "invoice.payment_failed",
  "webhook_id": "wh_7f3a9c",
  "created_at": "2026-03-11T14:02:31Z",
  "data": {
    "invoice_id": "inv_2291",
    "customer_id": "cus_884a1"
  }
}
```

## Troubleshooting failed deliveries

If you're seeing repeated `ERR_WEBHOOK_5003` failures for an endpoint:

* Confirm your endpoint returns a `2xx` status within 10 seconds, even if
  processing happens asynchronously afterward.
* Check that your TLS certificate is valid and not expired.
* Check firewall rules — Acme Cloud webhook traffic originates from a
  documented set of IP ranges, listed in the dashboard under
  **Settings → Webhooks → Delivery IPs**.

See `production-runbook.txt` for how the on-call team investigates
widespread webhook delivery failures affecting many customers at once.
