from rank_bm25 import BM25Okapi
from vectorstore.chroma import query_vector

def reciprocal_rank_fusion(results: list[tuple[int, float]], k: int = 60) -> list[int]:
    # same as before
    ...

def hybrid_search(question: str, file_ids: list[str]):
    vector_results = query_vector(question, file_ids, n_results=20)

    # Extract data
    docs = vector_results["documents"][0]
    metas = vector_results["metadatas"][0]
    distances = vector_results["distances"][0]

    # BM25 on retrieved docs
    bm25 = BM25Okapi([doc.lower().split() for doc in docs])
    bm25_scores = bm25.get_scores(question.lower().split())

    vector_ranked = [(i, dist) for i, dist in enumerate(distances)]
    bm25_ranked = [(i, -score) for i, score in enumerate(bm25_scores) if score > 0]

    fused_indices = reciprocal_rank_fusion(vector_ranked + bm25_ranked)[:20]

    return docs, metas, distances, fused_indices