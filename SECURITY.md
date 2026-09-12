# Security Policy

## Supported versions

`spikeforge` is pre-1.0. Security fixes are made against the latest
released line and `main`; older `0.x` releases are not maintained.

| Version | Supported |
|---|---|
| `0.2.x` (current) | ✅ |
| `0.1.x` | ❌ |

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for a
security problem.**

Use GitHub's private vulnerability reporting: on the repository, go to
**Security → Advisories → Report a vulnerability**. This opens a private
advisory visible only to the maintainers.

If private reporting is unavailable, email the maintainer at
**`contact@capsizegames.com`** (a monitored inbox).

Please include:

- the affected version or commit,
- a description of the issue and its impact,
- a minimal reproduction (a command or a short snippet),
- any suggested mitigation, if you have one.

## What to expect

- **Acknowledgement** within a few days.
- **Assessment** of the report and a decision on severity.
- **A fix and disclosure** coordinated with you; we will credit reporters who
  wish to be named.

## Scope

This project runs **locally** and, by design, has no remote telemetry and no
authentication layer: it is a single-user tool, not a hardened multi-user
service. Reports we care most about include, but are not limited to:

- path traversal or arbitrary-write issues in the dataset / hub / model
  downloaders, which fetch into a local cache,
- untrusted deserialization in the ONNX, NIR, `nirtorch`, or checkpoint import
  paths,
- dependency confusion or a vulnerable pinned dependency,
- secret or credential exposure in the repository.

The deliberate boundaries — no authentication, no remote telemetry, estimated
(not measured) energy, SDK-gated hardware backends — are documented in the
[Implications and boundaries](documentation/implications-and-boundaries.md)
and are **not** vulnerabilities in themselves.
