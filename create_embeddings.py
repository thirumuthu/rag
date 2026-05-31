from sentence_transformers import SentenceTransformer
from load_documents import documents
from db_setup import collection

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def createEmbeddings():
    embeddings = embedding_model.encode(documents).tolist();

    collection.add(ids=[f"doc_{i}" for i in range(len(documents))],
               documents=documents,
               embeddings=embeddings)
    
if __name__ == "__main__":
    createEmbeddings()    