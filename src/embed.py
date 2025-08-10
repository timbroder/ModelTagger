import json
import chromadb
import tiktoken

def chunk_text(text, max_tokens=500, model="gpt-5"):
    enc = tiktoken.get_encoding("cl100k_base")
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = len(enc.encode(para))
        # If adding this paragraph would exceed the max, start new chunk
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
            current_tokens = para_tokens
        else:
            current_chunk += "\n" + para
            current_tokens += para_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def run_embedding(input_path, vector_db_path, model="gpt-5"):
    client = chromadb.PersistentClient(path=vector_db_path)
    collection = client.get_or_create_collection(name="lore")

    with open(input_path) as f:
        documents = json.load(f)

    for doc in documents:
        text = doc['text']
        url = doc['url']
        if not text.strip():
            continue
        for idx, chunk in enumerate(chunk_text(text, max_tokens=500, model=model)):
            chunk_id = f"{url}#chunk{idx}"
            collection.add(
                documents=[chunk],
                metadatas=[{'source': url, 'chunk': idx}],
                ids=[chunk_id]
            )

