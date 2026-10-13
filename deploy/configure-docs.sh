#!/usr/bin/env bash
# Update the host-owned, read-only-mounted edge configuration safely.
set -euo pipefail
caddy_container=$(docker ps --filter label=com.docker.compose.service=caddy --format '{{.Names}}' | head -n 1)
test -n "$caddy_container"
config_path=$(docker inspect "$caddy_container" --format '{{range .Mounts}}{{if eq .Destination "/etc/caddy/Caddyfile"}}{{.Source}}{{end}}{{end}}')
test -f "$config_path"
# Share the edge writers' lock before reading a candidate, and hold it
# through reload so a waiting deployment cannot publish an older snapshot.
exec 9>"$(dirname "$config_path")/Caddyfile.lock"
flock -x 9
candidate=$(mktemp /tmp/spikeforge-docs-caddy.XXXXXX)
trap 'rm -f "$candidate"' EXIT
if ! grep -q 'docs.spikeforge.net' "$config_path"; then
  awk 'FNR==1 {print ""} {print}' "$config_path" /opt/spikeforge/deploy/docs.Caddyfile > "$candidate"
else
  cp "$config_path" "$candidate"
fi
docker cp "$candidate" "$caddy_container:/tmp/spikeforge-docs-candidate"
docker exec "$caddy_container" caddy validate --config /tmp/spikeforge-docs-candidate --adapter caddyfile
if ! grep -q 'docs.spikeforge.net' "$config_path"; then
  cp -p "$config_path" "${config_path}.before-docs-$(date -u +%Y%m%dT%H%M%SZ)"
  cp "$candidate" "$config_path"
fi
# The bind mount can point at an older inode after another deploy replaces
# the host file. Reload the same candidate that was validated above.
docker exec "$caddy_container" caddy reload --config /tmp/spikeforge-docs-candidate --adapter caddyfile
