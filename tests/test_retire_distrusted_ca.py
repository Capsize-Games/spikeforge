"""Regression tests for exact DER removal and retained-capath handling."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "retire_distrusted_ca", ROOT / "scripts/retire_distrusted_ca.py"
)
assert SPEC and SPEC.loader
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)

C1 = Path("/usr/share/ca-certificates/mozilla/emSign_Root_CA_-_C1.crt")
C3 = Path("/usr/share/ca-certificates/mozilla/emSign_ECC_Root_CA_-_C3.crt")
G1 = Path("/usr/share/ca-certificates/mozilla/emSign_Root_CA_-_G1.crt")


class DistrustedCaTests(unittest.TestCase):
    """Exercise exact trust-store filtering and runtime paths."""

    def test_known_roots_have_the_expected_der_fingerprints(self) -> None:
        """Guard against changes to the two approved DER fingerprints."""
        if not all(path.is_file() for path in (C1, C3)):
            self.skipTest("retired roots are absent from this test host")
        self.assertEqual(
            POLICY.certificate_hashes(C1.read_bytes()),
            {
                "125609aa301da0a249b97a8239cb6a34216f44dcac9f3954b14292f2e8c8608f"
            },
        )
        self.assertEqual(
            POLICY.certificate_hashes(C3.read_bytes()),
            {
                "bc4d809b15189d78db3e1d8cf4f9726a795da1643ca5f1358e1ddb0edc0d7eb3"
            },
        )

    def test_filter_is_idempotent_and_preserves_other_roots(self) -> None:
        """Remove both retired roots without altering the retained root."""
        if not all(path.is_file() for path in (C1, C3, G1)):
            self.skipTest(
                "certificate fixtures are absent from this test host"
            )
        data = C1.read_bytes() + C3.read_bytes() + G1.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.pem"
            path.write_bytes(data)
            POLICY.filter_file(path)
            first = path.read_bytes()
            self.assertFalse(POLICY.certificate_hashes(first) & POLICY.RETIRED)
            self.assertEqual(
                POLICY.certificate_hashes(first),
                POLICY.certificate_hashes(G1.read_bytes()),
            )
            POLICY.filter_file(path)
            self.assertEqual(path.read_bytes(), first)

    def test_cli_accepts_both_explicit_modes(self) -> None:
        """Accept both documented command-line operations."""
        for argument, expected in (("--apply", True), ("--check", False)):
            with self.subTest(argument=argument), patch.object(
                POLICY, "clean_active_stores"
            ) as clean, patch(
                "sys.argv", ["retire_distrusted_ca.py", argument]
            ):
                POLICY.main()
                clean.assert_called_once_with(apply=expected)

    def test_none_default_capath_is_ignored(self) -> None:
        """Handle OpenSSL builds that have no default CApath."""
        with patch.object(
            POLICY.ssl, "get_default_verify_paths",
            return_value=SimpleNamespace(capath=None),
        ), patch.dict("os.environ", {"SSL_CERT_DIR": ""}, clear=True):
            self.assertEqual(POLICY.active_capath_paths(), set())

    def test_bad_capath_hash_symlinks_are_removed(self) -> None:
        """Remove bad aliases before filtering their source certificate."""
        if not C1.is_file():
            self.skipTest("retired root fixture is absent from this test host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "root.pem"
            source.write_bytes(C1.read_bytes())
            alias = root / "406c9bb1.0"
            alias.symlink_to(source)
            with patch.object(
                POLICY, "active_bundle_paths", return_value=set()
            ), patch.object(
                POLICY, "active_capath_paths", return_value={root}
            ), patch.object(
                POLICY, "update_distro_trust"
            ), patch.object(POLICY, "verify_distro_exclusions"):
                POLICY.clean_active_stores(apply=True)
            self.assertFalse(alias.exists())
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
