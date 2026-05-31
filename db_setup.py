from chromadb import PersistentClient

client = PersistentClient('./chroma_db')

collection = client.get_or_create_collection(name='Knowledge_base')
