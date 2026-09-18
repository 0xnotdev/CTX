# Changelog

All notable changes are documented here. This project follows semantic versioning.

## 1.0.0 — 2026-09-18

### Changed

- Production-hardening audit over the complete CP-00 through CP-12 implementation.
- Migrated the stdio adapter to the official MCP Python SDK 2 stable line.
- Added opaque document identity, SQLite V2 migrations, atomic index generations, bounded
  multi-chunk retrieval, exact source excerpts, strict serialized response budgeting,
  document-aware ambiguity handling, explicit offline model lifecycle, concurrency hardening,
  CLI/MCP parity, release automation, and adversarial regression coverage.

### Security

- Normal operations are offline-only and model download is an explicit command.
- Hardened private config writes, source containment/no-follow reads, checksums, limits, and
  stale-generation protections.

## 0.1.0 — 2025-09-17

- Initial CP-00 through CP-12 implementation.
