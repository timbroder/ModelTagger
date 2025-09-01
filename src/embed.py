import json
import chromadb
import tiktoken
import spacy
import re
from urllib.parse import urlparse
from tqdm import tqdm

from utils import slugify

def semantic_chunk_text(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.2,
    model: str = "gpt-5",
):
    """Split ``text`` into semantically meaningful chunks.

    The chunker produces segments between ``min_tokens`` and ``max_tokens`` using
    sentence boundaries. Chunks are generated with a sliding window so that each
    chunk shares an overlapping portion of the previous chunk. The overlap is
    "semantic" in that it respects sentence boundaries and avoids returning
    duplicate chunks.
    """

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    # Lightweight sentence boundary detection without requiring a large model
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    token_counts = [len(enc.encode(sent)) for sent in sentences]

    chunks = []
    seen = set()
    start = 0
    while start < len(sentences):
        current_tokens = 0
        end = start
        # Grow the chunk until reaching max_tokens
        while end < len(sentences) and current_tokens + token_counts[end] <= max_tokens:
            current_tokens += token_counts[end]
            end += 1

        # Ensure minimum token length when possible
        if current_tokens < min_tokens and end < len(sentences):
            while end < len(sentences) and current_tokens < min_tokens:
                current_tokens += token_counts[end]
                end += 1

        chunk = " ".join(sentences[start:end]).strip()
        chunk = re.sub(r"\s+", " ", chunk)  # Normalize whitespace

        if chunk and chunk not in seen:
            chunks.append(chunk)
            seen.add(chunk)

        if end >= len(sentences):
            break

        # Determine new start index based on overlap_ratio
        target_overlap = int(overlap_ratio * current_tokens)
        overlap = 0
        new_start = end
        while new_start > start and overlap < target_overlap:
            new_start -= 1
            overlap += token_counts[new_start]

        # Ensure progress to avoid infinite loop
        if new_start >= end or new_start == start:
            start = end
        else:
            start = new_start

    return chunks


def run_embedding(input_path, vector_db_path, model: str = "gpt-5"):
    client = chromadb.PersistentClient(path=vector_db_path)
    collection = client.get_or_create_collection(
        name="lore",
        metadata={"hnsw:space": "cosine"},
    )

    with open(input_path) as f:
        documents = json.load(f)

    prepared = []
    total_chunks = 0
    with tqdm(total=len(documents), desc="Pre Embedding") as pbar:
        for doc in documents:
            text = doc["text"]
            url = doc["url"]
            if not text.strip():
                continue
            chunks = semantic_chunk_text(text, min_tokens=300, max_tokens=800, model=model)
            slug = slugify(urlparse(url).path.rstrip("/").split("/")[-1])
            prepared.append((url, slug, chunks))
            total_chunks += len(chunks)
            pbar.update(1)

    with tqdm(total=total_chunks, desc="Embedding") as pbar:
        for url, slug, chunks in prepared:
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{url}#chunk{idx}"
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": url, "slug": slug, "chunk": idx}],
                    ids=[chunk_id],
                )
                pbar.update(1)

