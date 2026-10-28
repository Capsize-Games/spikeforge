"""Remove retired emSign C1/C3 roots from the image's active trust stores."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import re
import ssl
import subprocess
from pathlib import Path

RETIRED = {
    "125609aa301da0a249b97a8239cb6a34216f44dcac9f3954b14292f2e8c8608f",
    "bc4d809b15189d78db3e1d8cf4f9726a795da1643ca5f1358e1ddb0edc0d7eb3",
}
EXCLUDED = (
    "mozilla/emSign_Root_CA_-_C1.crt",
    "mozilla/emSign_ECC_Root_CA_-_C3.crt",
)
PEM = re.compile(
    rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----\s*", re.S
)


def certificate_hashes(data: bytes) -> set[str]:
    """Return DER SHA-256 fingerprints from PEM certificates."""
    result = set()
    for block in PEM.findall(data):
        try:
            der = ssl.PEM_cert_to_DER_cert(block.decode())
            result.add(hashlib.sha256(der).hexdigest())
        except (UnicodeDecodeError, ValueError):
            continue
    return result


def filter_file(path: Path) -> None:
    """Rewrite one PEM bundle without any retired certificate blocks."""
    data = path.read_bytes()
    output = []
    offset = 0
    for match in PEM.finditer(data):
        output.append(data[offset:match.start()])
        if not certificate_hashes(match.group()) & RETIRED:
            output.append(match.group())
        offset = match.end()
    output.append(data[offset:])
    path.write_bytes(b"".join(output))


def persist_distro_exclusions() -> None:
    """Keep Debian trust refreshes from reactivating the removed roots."""
    path = Path("/etc/ca-certificates.conf")
    lines = path.read_text().splitlines()
    for name in EXCLUDED:
        active = [i for i, line in enumerate(lines) if line == name]
        if active:
            lines[active[0]] = "!" + name
            for index in reversed(active[1:]):
                del lines[index]
        elif "!" + name not in lines:
            lines.append("!" + name)
    path.write_text("\n".join(lines) + "\n")


def active_bundle_paths() -> set[Path]:
    """Resolve OpenSSL, pip-vendored, and installed certifi bundle paths."""
    paths = set()
    defaults = ssl.get_default_verify_paths()
    override_names = (
        "SSL_CERT_FILE", "PIP_CERT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"
    )
    values = [defaults.cafile, *(os.getenv(name) for name in override_names)]
    for value in values:
        if value and Path(value).is_file():
            paths.add(Path(value))
    for module_name in ("pip._vendor.certifi", "certifi"):
        try:
            paths.add(Path(importlib.import_module(module_name).where()))
        except (ImportError, AttributeError):
            continue
    return paths


def active_capath_paths() -> set[Path]:
    """Return certificate paths from OpenSSL's hashed directory."""
    defaults = ssl.get_default_verify_paths()
    values = [defaults.capath, os.getenv("SSL_CERT_DIR", "")]
    directories = {
        part for value in values if value
        for part in value.split(os.pathsep) if part
    }
    return {
        Path(value) for value in directories
        if value and Path(value).is_dir()
    }


def update_distro_trust() -> None:
    """Regenerate Debian's system bundle after persistent exclusions."""
    persist_distro_exclusions()
    subprocess.run(["update-ca-certificates", "--fresh"], check=True)


def verify_distro_exclusions() -> None:
    """Require persistent package-manager exclusions for both retired roots."""
    lines = Path("/etc/ca-certificates.conf").read_text().splitlines()
    missing = [name for name in EXCLUDED if "!" + name not in lines]
    assert not missing, "Distro CA exclusions missing: " + ",".join(missing)


def clean_active_stores(apply: bool) -> None:
    """Apply or verify the retired-root policy across active stores."""
    if apply:
        update_distro_trust()
    verify_distro_exclusions()
    for path in active_bundle_paths():
        if apply:
            filter_file(path)
        found = certificate_hashes(path.read_bytes()) & RETIRED
        assert not found, f"Retired roots remain in active bundle: {path.name}"
    for directory in active_capath_paths():
        # Remove aliases before their source PEM files. A remaining symlink
        # would otherwise resolve to the now-cleaned source and appear safe.
        entries = sorted(
            directory.iterdir(), key=lambda path: not path.is_symlink()
        )
        for path in entries:
            if not path.is_file():
                continue
            found = certificate_hashes(path.read_bytes()) & RETIRED
            if apply and found and path.is_symlink():
                path.unlink()
                continue
            if apply and found:
                filter_file(path)
                found = certificate_hashes(path.read_bytes()) & RETIRED
            assert not found, (
                f"Retired roots remain in active CApath: {path.name}"
            )


def main() -> None:
    """Run the explicit trust-store apply or check mode."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    clean_active_stores(apply=args.apply)
    print("retired roots absent from system, installer, and certifi stores")


if __name__ == "__main__":
    main()
