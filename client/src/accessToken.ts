/**
 * Access token from the page's own URL, when the operator gated the server
 * with SPIKEFORGE_DASHBOARD_TOKEN (see documentation/usage.md) and shared a
 * link like `https://host/?token=...`. Undefined when unset, so a plain
 * `docker compose up` on localhost behaves exactly as it does today.
 */
export function accessToken(): string | null {
  return new URLSearchParams(location.search).get("token");
}
