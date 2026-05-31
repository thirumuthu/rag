from sentence_transformers import SentenceTransformer
from load_documents import documents
from db_setup import collection
from wikipedia_docs import load_page, chunk_text

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def createEmbeddings(topic):

    page = load_page(topic)
    chunks = chunk_text(page)
    embeddings = embedding_model.encode(chunks).tolist();

    collection.add(ids=[f"{topic}_{i}" for i in range(len(chunks))],
               documents=chunks,
               embeddings=embeddings,
               metadatas=[
                   {"source": topic, "chunk_id": i}
                   for i, _ in enumerate(chunks)
               ])
    
    print(f"Created embeddings for topic:{topic}")
    
if __name__ == "__main__":
    topic = input("Which topic you want to discuss about? \n")
    createEmbeddings(topic)    