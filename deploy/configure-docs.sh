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
backup=
host_update_started=0
rollback_reload_started=0
cleanup() {
  status=$?
  if [[ $status -ne 0 && $host_update_started -eq 1 ]]; then
    set +e
    if [[ -n "$backup" && -f "$backup" ]]; then
      cat "$backup" > "$config_path"
      if [[ $? -eq 0 ]]; then
        if [[ $rollback_reload_started -eq 1 ]]; then
          docker exec -i "$caddy_container" sh -c \
            'cat > /tmp/spikeforge-docs-rollback' < "$backup"
          docker exec "$caddy_container" caddy reload \
            --config /tmp/spikeforge-docs-rollback --adapter caddyfile
        fi
      else
        echo 'Could not restore the previous host Caddyfile' >&2
      fi
    fi
    set -e
  fi
  rm -f "$candidate"
  exit "$status"
}
trap cleanup EXIT
if ! grep -q 'docs.spikeforge.net' "$config_path"; then
  awk 'FNR==1 {print ""} {print}' "$config_path" /opt/spikeforge/deploy/docs.Caddyfile > "$candidate"
else
  cp "$config_path" "$candidate"
fi
# Docker cp cannot write into this container because its root filesystem is
# read-only. Stream into the writable /tmp tmpfs through the container shell.
docker exec -i "$caddy_container" sh -c \
  'cat > /tmp/spikeforge-docs-candidate' < "$candidate"
docker exec "$caddy_container" caddy validate --config /tmp/spikeforge-docs-candidate --adapter caddyfile
if ! grep -q 'docs.spikeforge.net' "$config_path"; then
  backup=$(mktemp "${config_path}.before-docs.XXXXXX")
  cp -p "$config_path" "$backup"
  chmod 600 "$backup"
  host_update_started=1
  cp "$candidate" "$config_path"
fi
# The bind mount can point at an older inode after another deploy replaces
# the host file. Reload the same candidate that was validated above.
if [[ $host_update_started -eq 1 ]]; then
  rollback_reload_started=1
fi
docker exec "$caddy_container" caddy reload --config /tmp/spikeforge-docs-candidate --adapter caddyfile
host_update_started=0
