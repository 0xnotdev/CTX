# V0 evaluation and hardening report

Recorded 2025-09-17 on the development host with Python 3.12.3 and NumPy 2.5.3. Fixtures are
programmatically generated technical material; no copyrighted realistic corpus is included.
Commands are reproducible and observations are not performance promises.

## Retrieval quality

`pytest tests/test_evaluation.py` evaluates seven ground-truth sections: exact heading, exact
symbol, paraphrase, cross-reference/checkpoint task, named error, and security policy. The
required pack set spans the checkpoint, dependency, interface model, security policy, and
acceptance/verify section.

| Metric | V0 regression threshold | Observed |
|---|---:|---:|
| Recall@1 | >= 0.85 | 1.000 |
| Recall@3 | 1.00 | 1.000 |
| MRR | >= 0.90 | 1.000 |
| exact-section accuracy | >= 0.85 | 1.000 |
| context-pack required-section recall | 1.00 | 1.000 |
| estimated tokens returned | <= 7,000 | 1,403 |
| experimental required-section recall / 1,000 returned tokens | > 0 | 0.713 |

The last ratio is experimental and corpus-dependent. It operationalizes the central goal—more
required authoritative information per returned token—without claiming universal tokenizer
accuracy (the stable default token estimator is documented in architecture).

## Scale and latency observations

`python benchmarks/run_benchmarks.py` generated a 3,000-line technical spec and 20,000-line
research file, indexed both with deterministic fixture embeddings, then reported medians:

| Operation | Observed |
|---|---:|
| parse 3,000 lines | 64.9 ms |
| parse 20,000 lines | 496.7 ms |
| complete fixture index | 1,269.6 ms |
| exact lookup (100 runs) | 3.96 ms |
| FTS query (50 runs) | 5.77 ms |
| warm hybrid query (20 runs) | 44.36 ms |
| context pack construction (10 runs) | 154.66 ms |
| 100 MiB default-bound check | 0.21 ms, rejected |

The V0 default intentionally bounds each source at 25 MiB. A sparse 100 MiB source was measured
and predictably rejected by that configured bound; full 100 MiB parsing/indexing was not run and
is not falsely claimed usable. Users may raise the limit, but should benchmark their host first.
The separate 20,000×384 cosine measurement is in `benchmarks/embedding_selection.md`.

`tests/test_large_corpus.py` continuously covers a 10-line document plus generated 3,000- and
20,000-line documents with front matter, nested/duplicate headings, tables, large fences, lists,
blockquotes, Unicode, a 20,000-character paragraph, technical identifiers, and exact source
reconstruction. Parser tests separately cover malformed/unclosed Markdown and identical
sections; incremental tests cover renamed/deleted sections.

## Security and correctness coverage

The quality suite verifies workspace traversal and absolute-path rejection, symlink escape,
Markdown-only and file-size bounds, read-only source access, SQL/FTS parameterization, query,
result, line-range and MCP response bounds, stale normative-source refusal, explicit auto-sync,
and inert hostile front matter/HTML/links/scripts/code fences. MCP is exercised through an
official stdio client. Malformed Markdown cannot crash indexing or retrieval.

`tests/test_final_success.py` builds a realistic three-document workspace, indexes it, creates a
CP-14 pack under 7,000 estimated tokens, and checks exact checkpoint/dependency/model/security/
acceptance/verify/out-of-scope source with hashes and line ranges without returning the whole
corpus. It then changes one section and proves one changed section/vector, one retained section
vector in that document, and zero document work for the other two sources.
