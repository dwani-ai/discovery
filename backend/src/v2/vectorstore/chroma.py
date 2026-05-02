import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import settings

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

class EmbeddingGemmaFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.encode_document(documents).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode_query(query).tolist()

embedding_function = EmbeddingGemmaFunction(settings.DWANI_EMBEDDING_MODEL)

def add_chunks(file_id: str, filename: str, documents: list[str], metadatas: list[dict], ids: list[str]):
    collection.delete(where={"file_id": file_id})
    collection.add(
        embeddings=embedding_function.embed_documents(documents),
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

def delete_by_file_id(file_id: str):
    collection.delete(where={"file_id": file_id})

def query_vector(question: str, file_ids: list[str], n_results: int = 20):
    return collection.query(
        query_embeddings=[embedding_function.embed_query(question)],
        n_results=n_results,
        where={"file_id": {"$in": file_ids}},
        include=["documents", "metadatas", "distances"],
    )