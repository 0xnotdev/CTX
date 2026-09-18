# Changelog

All notable changes are documented here. This project follows semantic versioning.

## Unreleased

### Fixed

- Added strict agent-pack discovery of materially relevant normative architecture, decisions,
  current state/progress, security, testing, and other cross-document evidence. Required budget or
  filter omissions now prevent `COMPLETE` with exact actionable provenance.
- Added fail-closed `require_semantic` parity across config, service, CLI, and MCP. Strict agent
  packs validate the local provider and current compatible vector generation; explicit non-strict
  fallback is labeled `LEXICAL_ONLY`.

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
