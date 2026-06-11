from __future__ import annotations

import json
import os
import chromadb
import frontmatter
import tiktoken
import spacy
import re
from urllib.parse import urlparse, unquote
from tqdm import tqdm

from utils import slugify, is_article_url

# Chroma's default embedding function (MiniLM-L6) truncates input at 256
# wordpiece tokens. cl100k tokens run denser than wordpieces and chunks get a
# title prefix, so cap well below 256 to keep the whole chunk visible.
_DEFAULT_EF_MAX_TOKENS = 200
_DEFAULT_EF_MIN_TOKENS = 80
_LOCAL_EF_MAX_TOKENS = 800
_LOCAL_EF_MIN_TOKENS = 300

# Page titles matching this are the site-wide <title> (title extraction failed
# on some Wayback snapshot layouts) — derive a title from the URL instead
_SITE_TITLE_RE = re.compile(r"lexicanum|fandom|wikia|just a moment", re.IGNORECASE)

# Wiki sections that carry no lore signal for tagging
_BOILERPLATE_SECTIONS = {
    "contents", "sources", "see also", "related articles", "notes",
    "references", "external links", "gallery", "uncited", "navigation",
    "footnotes",
}

# Phrases identifying maintenance/portal/navigation banner tables
_BANNER_RE = re.compile(
    r"citation issue|this article|please help|portal\b|\bstub\b|cleanup|disambiguation",
    re.IGNORECASE,
)

# Footnote references like [[1]](#fn_1) / [[25b]](#fn_25b) — the nested
# brackets defeat the generic inline-link regex, so strip them first
_FOOTNOTE_LINK_RE = re.compile(r"\[\[[^\]]{1,10}\]\]\(#[^)]*\)")

# Inline markdown links — keep only the display text. One level of nested
# parens is allowed for title attributes inside the link destination.
_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^()]*(?:\([^()]*\)[^()]*)*\)")

_HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$", re.MULTILINE)


def clean_markdown(text: str) -> str:
    """Strip wiki noise that survives scraping.

    Removes footnote references, inline link targets, and pipe-table blocks
    that are maintenance banners or pure layout (empty first column) rather
    than data tables.
    """
    text = _FOOTNOTE_LINK_RE.sub("", text)
    text = _INLINE_LINK_RE.sub(r"\1", text)

    out_lines: list[str] = []
    block: list[str] = []

    def flush_block():
        if not block:
            return
        block_text = "\n".join(block)
        first_cells = [line.strip().strip("|").split("|")[0].strip() for line in block]
        is_layout = all(c in ("", "---") for c in first_cells)
        if not (_BANNER_RE.search(block_text) or is_layout):
            out_lines.extend(block)
        block.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            block.append(line)
        else:
            flush_block()
            out_lines.append(line)
    flush_block()
    return "\n".join(out_lines)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, body) pairs by ATX headings.

    The preamble before the first heading has an empty section title.
    Boilerplate sections (Sources, See also, the Contents TOC, ...) and empty
    bodies are dropped.
    """
    sections: list[tuple[str, str]] = []
    last_title = ""
    last_end = 0
    for m in _HEADING_RE.finditer(text):
        sections.append((last_title, text[last_end:m.start()]))
        last_title = m.group(1)
        last_end = m.end()
    sections.append((last_title, text[last_end:]))

    result = []
    for title, body in sections:
        key = re.sub(r"[^a-z ]", "", title.lower()).strip()
        if key in _BOILERPLATE_SECTIONS:
            continue
        if body.strip():
            result.append((title, body))
    return result


def repair_title(title: str | None, url: str) -> str:
    """Return a usable page title, deriving one from the URL when the scraped
    title is missing or is the generic site title."""
    if title and not _SITE_TITLE_RE.search(title):
        return title
    segment = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    return segment.replace("_", " ").replace("-", " ")


def dedupe_documents(documents: list[dict]) -> list[dict]:
    """Drop redirect duplicates: pages whose (repaired) titles match are the
    same article scraped under different URLs. Keeps the longest text."""
    by_title: dict[str, dict] = {}
    untitled: list[dict] = []
    for doc in documents:
        key = repair_title(doc.get("title"), doc.get("url", "")).strip().lower()
        if not key:
            untitled.append(doc)
            continue
        cur = by_title.get(key)
        if cur is None or len(doc.get("text", "")) > len(cur.get("text", "")):
            by_title[key] = doc
    dropped = len(documents) - len(by_title) - len(untitled)
    if dropped:
        print(f"Deduped {dropped} redirect-duplicate documents")
    return list(by_title.values()) + untitled


def build_summary_chunk(title: str, categories: list, infobox: dict) -> str | None:
    """Condense page metadata (categories + infobox) into one dense chunk —
    it's the most tag-like text on the page."""
    parts = [title]
    if categories:
        parts.append("Categories: " + ", ".join(str(c) for c in categories))
    if infobox:
        parts.append("; ".join(f"{k}: {v}" for k, v in infobox.items()))
    if len(parts) == 1:
        return None
    return ". ".join(parts)


def semantic_chunk_text(
    text: str,
    min_tokens: int = 300,
    max_tokens: int = 800,
    overlap_ratio: float = 0.2,
    model: str = "gpt-4o",
) -> list[str]:
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

    # Hard-split "sentences" that exceed max_tokens on their own (typically
    # long bullet lists with no sentence punctuation) so no chunk can ever
    # blow past the embedder's window
    bounded: list[str] = []
    for sent in sentences:
        toks = enc.encode(sent)
        if len(toks) <= max_tokens:
            bounded.append(sent)
        else:
            for i in range(0, len(toks), max_tokens):
                bounded.append(enc.decode(toks[i:i + max_tokens]).strip())
    sentences = bounded
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


def load_documents(input_path: str) -> list[dict]:
    if os.path.isdir(input_path):
        documents = []
        for fname in sorted(os.listdir(input_path)):
            if fname.endswith(".md") and not fname.startswith("_"):
                with open(os.path.join(input_path, fname), encoding="utf-8") as f:
                    post = frontmatter.load(f)
                doc = dict(post.metadata)
                doc["text"] = post.content
                documents.append(doc)
        return documents
    with open(input_path) as f:
        return json.load(f)


def run_embedding(
    input_path: str,
    vector_db_path: str,
    use_local: bool = False,
    embed_model: str = "BAAI/bge-m3",
    model: str = "gpt-4o",
    min_chunk_tokens: int | None = None,
    max_chunk_tokens: int | None = None,
) -> None:
    if max_chunk_tokens is None:
        max_chunk_tokens = _LOCAL_EF_MAX_TOKENS if use_local else _DEFAULT_EF_MAX_TOKENS
    if min_chunk_tokens is None:
        min_chunk_tokens = _LOCAL_EF_MIN_TOKENS if use_local else _DEFAULT_EF_MIN_TOKENS

    client = chromadb.PersistentClient(path=vector_db_path)
    if use_local:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=embed_model)
        collection = client.get_or_create_collection(
            name="lore",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        collection = client.get_or_create_collection(
            name="lore",
            metadata={"hnsw:space": "cosine"},
        )

    documents = load_documents(input_path)
    articles = [d for d in documents if is_article_url(d.get("url", ""))]
    if len(articles) < len(documents):
        print(f"Skipped {len(documents) - len(articles)} non-article namespace pages (File:, Category:, ...)")
    documents = dedupe_documents(articles)

    prepared = []
    with tqdm(total=len(documents), desc="Processing Documents") as doc_pbar:
        for doc in documents:
            text = doc["text"]
            url = doc["url"]
            # Skip redirect pages and empty documents
            if not text.strip() or text.lstrip().lower().startswith(("redirect to:", "#redirect")):
                doc_pbar.update(1)
                continue

            text = clean_markdown(text)
            title = repair_title(doc.get("title"), url)

            # (section, chunk_text) pairs; summary chunk first
            chunks: list[tuple[str, str]] = []
            summary = build_summary_chunk(
                title, doc.get("categories") or [], doc.get("infobox") or {}
            )
            if summary:
                chunks.append(("_summary", summary))

            for section, body in split_sections(text):
                prefix = f"{title} — {section}" if section else title
                for raw in semantic_chunk_text(
                    body, min_tokens=min_chunk_tokens, max_tokens=max_chunk_tokens, model=model
                ):
                    chunks.append((section, f"{prefix} - {raw}"))

            if not chunks:
                doc_pbar.update(1)
                continue

            slug = slugify(unquote(urlparse(url).path.rstrip("/").split("/")[-1]))
            categories = ", ".join(str(c) for c in doc.get("categories") or [])
            prepared.append((url, slug, title, categories, chunks))
            doc_pbar.update(1)

    # Flatten all chunks into batches for efficient embedding
    all_docs = []
    all_metas = []
    all_ids = []
    for url, slug, title, categories, chunks in prepared:
        for idx, (section, chunk) in enumerate(chunks):
            all_docs.append(chunk)
            all_metas.append({
                "source": url,
                "slug": slug,
                "chunk": idx,
                "title": title,
                "section": section,
                "categories": categories,
            })
            all_ids.append(f"{url}#chunk{idx}")

    batch_size = 100
    with tqdm(total=len(all_docs), desc="Embedding") as pbar:
        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            batch_metas = all_metas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            # upsert keeps re-runs idempotent: existing chunk IDs are updated
            # in place instead of erroring or duplicating
            collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
            pbar.update(len(batch_docs))
