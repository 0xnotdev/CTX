from pathlib import Path


def write_exact_source(path: Path, text: str) -> None:
    """Write UTF-8 fixture bytes without platform newline translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def read_exact_source(path: Path) -> str:
    """Decode fixture bytes without universal-newline normalization."""
    return path.read_bytes().decode("utf-8")
