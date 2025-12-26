from services.search import hybrid_search, reciprocal_rank_fusion
from vectorstore.chroma import add_chunks

def test_reciprocal_rank_fusion():
    results = [(0, 0.1), (1, 0.3), (0, 0.2), (2, 0.05)]
    fused = reciprocal_rank_fusion(results)
    assert fused[0] == 0  # Most frequent
    assert fused[1] == 1

def test_hybrid_search():
    # Add test data
    docs = ["apple pie recipe", "banana smoothie", "apple orchard"]
    metas = [{"file_id": "f1", "filename": "recipes.pdf", "page_start": i+1, "page_end": i+1, "chunk_index": i} for i in range(3)]
    ids = ["a1", "a2", "a3"]
    add_chunks("f1", "recipes.pdf", docs, metas, ids)

    docs_ret, metas_ret, distances, fused = hybrid_search("apple", ["f1"])

    assert len(fused) > 0
    top_doc = docs_ret[fused[0]]
    assert "apple" in top_doc.lower()