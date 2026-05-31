import wikipediaapi

def load_page(title):
    wiki = wikipediaapi.Wikipedia(
        user_agent="my-rag/1.0",
        language="en"
    )

    page = wiki.page(title)

    if not page.exists():
       raise ValueError(f"Page {title} not found in Wiki")

    return page.text

def chunk_text(text, chunk_size=500, overlap=100):
  chunk = []

  start = 0

  while start < len(text):
     end = start + chunk_size

     chunk.append(text[start:end])

     start += chunk_size - overlap

  return chunk   


