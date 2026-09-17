from pathlib import Path

import numpy as np

from ctx.config import add_document_config, initialize_workspace
from ctx.embeddings import HashEmbedding, cosine_scores
from ctx.models import Authority
from ctx.service import ContextEngine


def test_hash_fixture_is_deterministic_and_cosine_is_stable() -> None:
    backend = HashEmbedding(32)
    first = backend.embed_documents(["network ingress", "retry timeout"])
    second = backend.embed_documents(["network ingress", "retry timeout"])
    np.testing.assert_array_equal(first, second)
    scores = cosine_scores(backend.embed_query("network ingress"), first)
    assert scores.shape == (2,)
    assert int(np.argmax(scores)) == 0
    assert scores[0] > scores[1]
    zeros = cosine_scores(np.zeros(32, dtype=np.float32), first)
    np.testing.assert_array_equal(zeros, np.zeros(2, dtype=np.float32))


def test_embeddings_persist_and_only_changed_chunk_is_reembedded(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    source = tmp_path / "spec.md"
    source.write_text("# One\nnetwork ingress\n# Two\nretry timeout\n", encoding="utf-8")
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    backend = HashEmbedding(32)
    with ContextEngine(tmp_path, embedder=backend) as engine:
        first = engine.index_workspace()
        assert first.embeddings_created == 2
        ids, matrix = engine.store.load_embeddings(backend.identity)
        assert len(ids) == 2
        assert matrix.shape == (2, 32)
        assert engine.store.get_metadata("embedding_model") == backend.identity
        assert engine.store.get_metadata("embedding_dimensions") == "32"

        unchanged = engine.sync_workspace()
        assert unchanged.embeddings_created == 0
        source.write_text(
            "# One\nnetwork ingress\n# Two\nretry timeout changed\n", encoding="utf-8"
        )
        changed = engine.sync_workspace()
        assert changed.sections_changed == 1
        assert changed.embeddings_retained == 1
        assert changed.embeddings_created == 1
        assert len(engine.store.load_embeddings(backend.identity)[0]) == 2


def test_model_identity_change_reembeds_all_chunks(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "a.md").write_text("# A\ntext\n", encoding="utf-8")
    add_document_config(tmp_path, "a.md", Authority.REFERENCE)
    with ContextEngine(tmp_path, embedder=HashEmbedding(16)) as engine:
        engine.index_workspace()
    replacement = HashEmbedding(24)
    with ContextEngine(tmp_path, embedder=replacement) as engine:
        result = engine.sync_workspace()
        assert result.documents_unchanged == 1
        assert result.embeddings_created == 1
        _, matrix = engine.store.load_embeddings(replacement.identity)
        assert matrix.shape == (1, 24)
