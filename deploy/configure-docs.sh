#!/usr/bin/env bash
# Update the host-owned, read-only-mounted edge configuration safely.
set -euo pipefail
caddy_container=$(docker ps --filter label=com.docker.compose.service=caddy --format '{{.Names}}' | head -n 1)
test -n "$caddy_container"
config_path=$(docker inspect "$caddy_container" --format '{{range .Mounts}}{{if eq .Destination "/etc/caddy/Caddyfile"}}{{.Source}}{{end}}{{end}}')
test -f "$config_path"
if ! grep -q 'docs.spikeforge.net' "$config_path"; then
  candidate=$(mktemp /tmp/spikeforge-docs-caddy.XXXXXX)
  trap 'rm -f "$candidate"' EXIT
  awk 'FNR==1 {print ""} {print}' "$config_path" /opt/spikeforge/deploy/docs.Caddyfile > "$candidate"
  docker cp "$candidate" "$caddy_container:/tmp/spikeforge-docs-candidate"
  docker exec "$caddy_container" caddy validate --config /tmp/spikeforge-docs-candidate --adapter caddyfile
  cp -p "$config_path" "${config_path}.before-docs-$(date -u +%Y%m%dT%H%M%SZ)"
  # Preserve the inode used by the existing bind mount.
  cp "$candidate" "$config_path"
fi
docker exec "$caddy_container" caddy validate --config /etc/caddy/Caddyfile
docker exec "$caddy_container" caddy reload --config /etc/caddy/Caddyfile
