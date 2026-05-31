import ollama
from db_setup import collection
from create_embeddings import embedding_model



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
        Do not add any extra sentence.
        If question is outside the context then say it is outside given context.

        """
    
    response = ollama.chat(
        model="qwen3:0.6b",
        messages=[
           {
               "role": "user",
               "content": prompt
           }
        ]
    )

 
    return response["message"]["content"]

if __name__ == "__main__":
    question = input("How can i help you?")
    print(ask(question=question))
