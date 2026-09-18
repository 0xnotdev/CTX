# Security policy

## Supported release

Security fixes are applied to the current V1 release line.

## Local trust boundary

`ctx` is local, stdio-only infrastructure. It opens no listening socket and has no SaaS,
telemetry, cloud database, hosted inference, or remote embedding path. Normal `index`, `sync`,
`search`, `pack`, checkpoint, doctor, and MCP operations must not access the network. The only
network-capable operation is the explicit `ctx model download` command.

A workspace and its configured Markdown are trusted as **inert input**, not executable code.
Returned Markdown, HTML, links, and code fences must never be executed merely because they were
indexed or returned by a tool. SQLite indexes, embeddings, graph edges, checkpoint metadata, and
context packs are disposable navigation data; the exact Markdown source remains authoritative.

The stdio peer can read configured source through ctx and request index updates. Run ctx only for
workspaces and MCP clients you trust. Do not expose the stdio process through a network bridge.
The `.ctx` directory contains private local metadata and should not be shared.

Source containment rejects absolute paths, lexical escapes, and symlink escapes. Reads use
no-follow opening where the platform supports it, then validate file identity and size. No
portable userspace API can make an attacker-controlled directory tree perfectly race-proof;
do not index a workspace concurrently writable by an untrusted local user.

Model installation verifies a local manifest and artifact checksums before use. Treat models as
software supply-chain artifacts and review their licenses. A missing or invalid model causes an
explicit structural/lexical fallback; it never triggers an automatic download.

## Reporting

Report a suspected vulnerability privately to the repository owner. Include the affected
version, platform, reproduction, impact, and whether a hostile workspace or local race is needed.
Do not include private indexed content in a public report.
