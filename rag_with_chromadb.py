from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import ollama

client = PersistentClient('./chroma_db')

collection = client.get_or_create_collection(name='Knowledge_base')

embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

documents = [
    "Python is a high level programming language",
    "RAG stands for Retrieval Augumentad Generation",
    "Chromadb is open source vector db",
    "Embeddings coinvert text into numerical vectors"
]

def createEmbeddings():


    embeddings = embedding_model.encode(documents).tolist();

    collection.add(ids=[f"doc_{i}" for i in range(len(documents))],
               documents=documents,
               embeddings=embeddings)

def retrive(query,top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    result = collection.query(query_embeddings=[query_embedding],
                     n_results=top_k)
    return result["documents"][0]

def ask(question):
    context_docs = retrive(question)
    context = "\n".join(context_docs)

    prompt = f"""

        Context: {context}

        Question: {question}

        Answer using only provided context.
        """
    
    response = ollama.chat(
        model="llama3",
        messages=[
           {
               "role": "user",
               "content": prompt
           }
        ]
    )

    print("Response from LLM==>")
    print(response)
    return response["message"]["content"]

print(ask("What is RAG?"))
