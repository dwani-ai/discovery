import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)

def add_chunks(file_id: str, filename: str, documents: list[str], metadatas: list[dict], ids: list[str]):
    collection.delete(where={"file_id": file_id})
    collection.add(
        embeddings=embedding_function(documents),
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

def delete_by_file_id(file_id: str):
    collection.delete(where={"file_id": file_id})

def query_vector(question: str, file_ids: list[str], n_results: int = 20):
    return collection.query(
        query_embeddings=embedding_function([question]),
        n_results=n_results,
        where={"file_id": {"$in": file_ids}},
        include=["documents", "metadatas", "distances"],
    )