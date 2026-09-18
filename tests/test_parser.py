from ctx.parser import parse_markdown, sha256_text


def test_nested_hierarchy_duplicate_ids_and_exact_reconstruction() -> None:
    source = """---
title: Δ spec
---
intro
# API
body
## Retry
one
## Retry
two
### Child
child
# End
fin
"""
    parsed = parse_markdown(source, "docs/spec.md")
    assert parsed.front_matter == "---\ntitle: Δ spec\n---\n"
    assert "".join(section.text for section in parsed.sections) == source
    assert [section.heading for section in parsed.sections] == [
        "[preamble]",
        "API",
        "Retry",
        "Retry",
        "Child",
        "End",
    ]
    first, second = parsed.sections[2:4]
    assert first.id != second.id
    assert "retry-2" in second.id
    assert parsed.sections[4].parent_id == second.id
    assert parsed.sections[4].heading_path == ("API", "Retry", "Child")
    assert parsed.sections[-1].start_line == 13
    assert all(section.sha256 == sha256_text(section.text) for section in parsed.sections)


def test_fences_tables_and_heading_content_stay_whole() -> None:
    source = """# Data
| key | value |
| --- | ----- |
| `# x` | ok |

```python
# not a heading
print('§25')
```
# Next
text
"""
    parsed = parse_markdown(source)
    assert len(parsed.sections) == 2
    assert "# not a heading" in parsed.sections[0].text
    assert "| --- | ----- |" in parsed.sections[0].text
    assert parsed.sections[0].end_line == 9
    assert parsed.sections[1].text == "# Next\ntext\n"
    assert parsed.chunks[0].text == parsed.sections[0].text


def test_identical_sections_are_distinct_and_deterministic() -> None:
    source = "# Same\nbody\n# Same\nbody\n"
    one = parse_markdown(source, "a.md")
    two = parse_markdown(source, "a.md")
    assert one == two
    assert one.sections[0].sha256 == one.sections[1].sha256
    assert one.sections[0].id != one.sections[1].id


def test_long_paragraph_is_split_into_bounded_exact_chunks() -> None:
    paragraph = "word " * 20_000
    source = f"# Long\n{paragraph}\n"
    parsed = parse_markdown(source)
    assert len(parsed.sections) == 1
    assert len(parsed.chunks) > 50
    assert parsed.chunks[0].start_offset == 0
    assert parsed.chunks[-1].end_offset == len(source)
    assert all(
        right.start_offset < left.end_offset
        for left, right in zip(parsed.chunks, parsed.chunks[1:], strict=False)
    )
    assert all(chunk.token_estimate <= 448 for chunk in parsed.chunks)
    assert all(
        source[chunk.start_offset : chunk.end_offset] == chunk.source_text
        for chunk in parsed.chunks
    )
    assert all(chunk.source_sha256 for chunk in parsed.chunks)


def test_malformed_markdown_and_unclosed_fence_do_not_crash() -> None:
    source = "---\nbroken: [\n# Still text\n```\n# inside\n\ud7ff unicode\n"
    parsed = parse_markdown(source)
    assert "".join(section.text for section in parsed.sections) == source
    assert len(parsed.sections) == 2
    assert parsed.front_matter is None
    assert "# inside" in parsed.sections[-1].text


def test_setext_headings_come_from_ast() -> None:
    source = "Title\n=====\ntext\nSub\n---\nmore\n"
    parsed = parse_markdown(source)
    # CommonMark treats the contiguous paragraph plus underline as one setext heading.
    assert [section.heading for section in parsed.sections] == ["Title", "text\nSub"]
    assert parsed.sections[1].parent_id == parsed.sections[0].id
