/**
 * Access token from the page's own URL, when the operator gated the server
 * with SPIKEFORGE_DASHBOARD_TOKEN (see documentation/usage.md) and shared a
 * link like `https://host/?token=...`. The token is removed from the address
 * bar immediately after it is read. Undefined when unset for local development.
 */
export function accessToken(): string | null {
  const params = new URLSearchParams(location.search);
  const token = params.get("token");
  if (token) {
    params.delete("token");
    const query = params.toString();
    const cleanUrl = `${location.pathname}${query ? `?${query}` : ""}${location.hash}`;
    history.replaceState(null, "", cleanUrl);
  }
  return token;
}
