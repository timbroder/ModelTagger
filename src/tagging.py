from __future__ import annotations

import os
import tempfile
import shutil
import json
from pathlib import Path
import patoolib
from chromadb import PersistentClient
from openai import OpenAI
import tiktoken  # Optional token estimation
import csv
import time
import logging
import re
import requests
from utils import slugify

# Resolve config relative to the repo root so the CLI works from any cwd
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Default models per provider when --model is not given
_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

# Lazy-initialized API clients
_openai_client = None
_anthropic_client = None


def get_openai_client():
    """Get or create the OpenAI client (lazy initialization)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_anthropic_client():
    """Get or create the Anthropic client (lazy initialization)."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client

def extract_to_temp(file_path: Path) -> Path | None:
    """Extract archive contents to a temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    ext = file_path.suffix.lower()

    try:
        if ext in [".zip", ".rar", ".7z"]:
            patoolib.extract_archive(str(file_path), outdir=str(temp_dir))
        elif ext in [".stl", ".obj", ".png"]:
            shutil.copy(file_path, temp_dir / file_path.name)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    except Exception as e:
        print(f"Extraction failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    return temp_dir

def is_valid_archive_content(folder: Path) -> bool:
    """Check if folder contains valid 3D model files and no dangerous executables."""
    allowed_exts = {".stl", ".obj", ".png"}
    bad_exts = {".exe", ".bat", ".js", ".dll"}
    files = list(folder.rglob("*"))
    if not files:
        return False
    for f in files:
        if f.suffix.lower() in bad_exts:
            return False
    return any(f.suffix.lower() in allowed_exts for f in files)


def clean_file_name(name: str) -> str:
    """Remove dates, timestamps, and symbols from a filename stem."""
    name = re.sub(r"\d{4}[-_\.]\d{2}[-_\.]\d{2}", " ", name)  # YYYY-MM-DD or similar
    name = re.sub(r"\d{2}[-_\.]\d{2}[-_\.]\d{4}", " ", name)  # DD-MM-YYYY or MM-DD-YYYY
    name = re.sub(r"\d{8,14}", " ", name)  # Compact dates or timestamps
    name = re.sub(r"\b(19|20)\d{2}\b", " ", name)  # Year alone
    name = re.sub(r"[^0-9a-zA-Z]+", " ", name)  # Replace symbols with spaces
    return re.sub(r"\s+", " ", name).strip()

# Tokens that describe print files rather than the miniature subject
_JUNK_TOKENS = {
    "supported", "presupported", "unsupported", "presup", "sup", "supports",
    "support", "base", "bases", "body", "bodies", "head", "heads", "arm",
    "arms", "leg", "legs", "left", "right", "part", "parts", "bits", "bit",
    "stl", "obj", "lys", "chitubox", "lychee", "final", "fixed", "repaired",
    "hollow", "hollowed", "solid", "raw", "merged", "split", "cut", "uncut",
    "version", "copy", "new", "old", "test", "print", "prints", "file", "files",
}


def filter_query_tokens(words: list[str]) -> list[str]:
    """Drop filename tokens that carry no lore signal: print-prep words,
    anything with digits (versions, dates, print-farm suffixes), and
    fragments. Deduplicates case-insensitively, preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        lw = w.lower()
        if lw in _JUNK_TOKENS:
            continue
        if any(ch.isdigit() for ch in w):
            continue
        if len(w) <= 2:
            continue
        if lw not in seen:
            seen.add(lw)
            out.append(w)
    return out


def candidate_slugs(words: list[str]) -> list[str]:
    """Page slugs worth looking up for a cleaned file name: the full name,
    each adjacent bigram, and each individual word (4+ chars).

    A file like "Wolfspear Techmarine" rarely matches a page slug as a whole,
    but "wolfspear" and "techmarine" each do.
    """
    slugs: list[str] = []
    if words:
        slugs.append(slugify(" ".join(words)))
    for i in range(len(words) - 1):
        slugs.append(slugify(f"{words[i]} {words[i + 1]}"))
    for w in words:
        if len(w) >= 4:
            slugs.append(slugify(w))
    seen: set[str] = set()
    out: list[str] = []
    for s in slugs:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def select_context_docs(
    documents: list[str],
    distances: list[float],
    metadatas: list[dict | None],
    max_docs: int = 8,
    delta: float = 0.15,
    per_page: int = 2,
    min_docs: int = 3,
) -> list[tuple[str, dict | None]]:
    """Pick context (doc, metadata) pairs by rank: everything within ``delta``
    of the best distance (always at least ``min_docs``), capped at
    ``per_page`` chunks per source page so one article can't crowd out the
    rest."""
    picked: list[tuple[str, dict | None]] = []
    counts: dict[str, int] = {}
    best = distances[0]
    for doc, dist, meta in zip(documents, distances, metadatas):
        src = (meta or {}).get("source")
        if src is not None and counts.get(src, 0) >= per_page:
            continue
        if dist > best + delta and len(picked) >= min_docs:
            break
        picked.append((doc, meta))
        if src is not None:
            counts[src] = counts.get(src, 0) + 1
        if len(picked) >= max_docs:
            break
    return picked


def related_pages_block(metadatas: list[dict | None]) -> str:
    """Summarize the retrieved pages' titles and wiki categories for the LLM.

    The categories are pre-made faction/role/allegiance labels — the most
    direct taxonomy signal retrieval produces.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for meta in metadatas:
        title = (meta or {}).get("title")
        if not title or title in seen:
            continue
        seen.add(title)
        cats = (meta or {}).get("categories") or ""
        lines.append(f"- {title}" + (f" [Categories: {cats}]" if cats else ""))
    if not lines:
        return ""
    return "Related wiki pages:\n" + "\n".join(lines) + "\n\n"


# Above this best-match cosine distance the retrieved context is probably
# unrelated (third-party minis with no wiki page) — warn the LLM rather than
# let it weave noise into the tags
_WEAK_CONTEXT_DISTANCE = 0.55
_WEAK_CONTEXT_NOTE = (
    "Note: the lore context below was only a weak match for this file name. "
    "Ignore it unless it is clearly about this miniature; otherwise rely on "
    "the file names and your own knowledge.\n\n"
)


# Tag values that are print/file metadata or trivially universal — never lore.
# (_JUNK_TOKENS already covers single print-prep words; these are phrases.)
_TAG_STOPWORDS = {
    "warhammer 40k", "warhammer 40000", "warhammer", "40k", "miniature", "mini",
    "miniatures", "3d print", "3d printable", "3d model", "stl", "obj",
    "custom sculpt", "sculpt", "fan made", "fan-made", "proxy", "tabletop",
    "wargaming", "wargame", "model", "figure", "figurine", "resin", "pre-supported",
    "multi-part model", "multi-part", "multipart", "multi part", "single piece",
    "single-piece", "highly detailed", "high detail",
}

_SMALL_WORDS = {"of", "the", "and", "a", "an", "in", "on", "for", "to", "with", "or"}


def titlecase_tag(t: str) -> str:
    """Capitalize each word's first letter (leaving the rest as-is so
    'Mechanicus' / 'McGuffin' survive), lowercasing small connector words."""
    words = t.split()
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _SMALL_WORDS:
            out.append(w.lower())
        elif w[:1].isalpha():
            out.append(w[:1].upper() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def _dedupe_substrings(items: list[str]) -> list[str]:
    """Drop an item whose words appear as a contiguous run inside another item
    (e.g. drop 'Claws' when 'Oversized Claws' is present)."""
    result = []
    for i, it in enumerate(items):
        pad = f" {it.lower()} "
        if any(j != i and pad in f" {items[j].lower()} " for j in range(len(items))):
            continue
        result.append(it)
    return result


def clean_equipment(value: str) -> str:
    """Title-case, case-insensitively dedupe, and drop redundant substrings
    from a comma-joined equipment string."""
    items, seen = [], set()
    for part in value.split(","):
        p = titlecase_tag(part.strip())
        if p and p.lower() not in seen:
            seen.add(p.lower())
            items.append(p)
    return ", ".join(_dedupe_substrings(items))


def clean_tags(tags: list[str], field_values: list[str]) -> list[str]:
    """Drop tags that duplicate a structured field or are print/file metadata,
    normalize casing, and dedupe — so the tag cloud stays clean and lore-only."""
    taken: set[str] = set()
    for v in field_values:
        for part in str(v).split(","):
            p = part.strip().lower()
            if p:
                taken.add(p)
    out, seen = [], set()
    for t in tags:
        t = " ".join(str(t).split())  # collapse whitespace
        low = t.lower()
        if not t or low in taken or low in _TAG_STOPWORDS or low in _JUNK_TOKENS:
            continue
        title = titlecase_tag(t)
        if title.lower() not in seen:
            seen.add(title.lower())
            out.append(title)
    return out


def normalize_record(data: dict, fields: list[str]) -> dict:
    """Coerce a tag dict into {field: str, "tags": [str]} for CSV output.

    List-valued fields become comma strings; "unknown"-like values become
    empty; equipment is deduped/title-cased; tags are stripped of field
    duplicates and print/file metadata and normalized.
    """
    out: dict = {}
    for f in fields:
        v = data.get(f, "")
        if isinstance(v, list):
            v = ", ".join(str(x).strip() for x in v if str(x).strip())
        v = str(v).strip()
        v = "" if v.lower() in ("unknown", "none", "n/a", "null") else v
        if f == "equipment" and v:
            v = clean_equipment(v)
        out[f] = v
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = parse_tags(tags)
    tags = [str(t).strip() for t in tags if str(t).strip()]
    out["tags"] = clean_tags(tags, [out[f] for f in fields])
    return out


def parse_structured(raw: str, fields: list[str]) -> dict:
    """Parse an LLM's free-text JSON tag object into {field: str, "tags": [str]}.

    Used for providers without schema-enforced output (OpenAI/Ollama). Falls
    back to treating the response as a flat tag list when no valid JSON object
    is found.
    """
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return normalize_record(data, fields)
    return normalize_record({"tags": parse_tags(raw)}, fields)


def build_schema(fields: list[str]) -> dict:
    """A permissive JSON schema (all string fields) for presets that don't
    define their own. Real presets ship an enum-constrained schema."""
    props = {f: {"type": "string"} for f in fields}
    props["tags"] = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": props,
        "required": fields + ["tags"],
        "additionalProperties": False,
    }


def ask_anthropic(prompt: str, schema: dict, model: str = "claude-sonnet-4-6",
                  retries: int = 3) -> tuple[dict | None, int]:
    """Tag via Anthropic with schema-enforced structured output.

    Returns (record, total_tokens) where record is guaranteed to match
    ``schema`` (enum fields can only hold allowed values), or (None, 0) on
    repeated failure so the caller can fall back.
    """
    for attempt in range(retries):
        try:
            response = get_anthropic_client().messages.create(
                model=model,
                max_tokens=512,
                system="You are a miniature tagging assistant.",
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return json.loads(text), tokens
        except Exception as e:
            print(f"[Anthropic Retry {attempt + 1}] Error: {e}")
            if attempt == retries - 1:
                return None, 0
            time.sleep(2 ** attempt)


def get_tokenizer(model: str):
    """Get the appropriate tiktoken encoder for the given model."""
    import tiktoken
    if model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4", "gpt-3.5-turbo"):
        return tiktoken.get_encoding("cl100k_base")
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count the number of tokens in text for the given model."""
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def ensure_local_model(model: str) -> None:
    """Ensure the specified Ollama model is available, pulling if necessary."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("model") or m.get("name") for m in resp.json().get("models", [])]
        if model not in models:
            pull = requests.post("http://localhost:11434/api/pull", json={"name": model}, timeout=60)
            pull.raise_for_status()
    except Exception as e:
        print(f"[Ollama Pull Warning] {e}")


def ask_local_model(prompt: str, model: str) -> str:
    try:
        ensure_local_model(model)
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "temperature": 0.2, "stream": False,
                  "format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "unknown")
    except Exception as e:
        print(f"[Local Model Error] {e}")
        return "unknown"

def ask_openai(prompt: str, model: str = "gpt-4.1", retries: int = 3) -> tuple[str, int]:
    """Send a prompt to OpenAI and return (response_text, total_tokens)."""
    for attempt in range(retries):
        try:
            response = get_openai_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a miniature tagging assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            completion_text = response.choices[0].message.content
            if response.usage is not None:
                total_tokens = response.usage.total_tokens
            else:
                total_tokens = count_tokens(prompt, model) + count_tokens(completion_text, model)
            return completion_text, total_tokens
        except Exception as e:
            print(f"[OpenAI Retry {attempt + 1}] Error: {e}")
            if attempt == retries - 1:
                return "unknown", count_tokens(prompt, model)
            time.sleep(2 ** attempt)


def parse_tags(raw: str) -> list:
    """Extract a clean list of tags from an LLM response."""
    # Try to capture enumerated or bulleted list items first
    items = re.findall(r"\d+\.\s*([^;\n]+)", raw)
    if not items:
        items = re.findall(r"[-*]\s*([^;\n]+)", raw)
    if not items:
        items = re.split(r"[\n;,]+", raw)

    cleaned = []
    for item in items:
        item = re.sub(r"^\s*\d+\.?\s*", "", item)
        item = item.strip(" \"'")
        if not item:
            continue
        if re.search(r"suggested tags|tags reflect|based on", item, re.IGNORECASE):
            continue
        cleaned.append(item)
    return cleaned


def run_tagging(
    zips_dir: str,
    output_csv: str,
    vector_db_path: str | None,
    prompt_override: str | None,
    mode: str,
    use_local: bool = False,
    local_model: str = "llama3.1:8b-instruct",
    model: str | None = None,
    token_budget: int = 3000,
    rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-base",
    provider: str = "anthropic",
) -> None:
    """Tag 3D model files using RAG with lore from the vector database.

    provider: "anthropic" (schema-enforced structured output), "openai", or
    "local" (Ollama). use_local=True forces "local" for backward compatibility.
    """
    if use_local:
        provider = "local"
    logging.basicConfig(filename='tagging.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    with open(_CONFIG_DIR / "tagging_presets.json") as f:
        presets = json.load(f)[mode]

    vector_db_path = vector_db_path or presets["vector_db"]
    prompt = prompt_override or presets["prompt"]

    # The schema is the source of truth for the output fields when present.
    if presets.get("schema"):
        schema = presets["schema"]
        fields = [k for k in schema["properties"] if k != "tags"]
    else:
        fields = presets.get("fields", [])
        schema = build_schema(fields)

    if provider == "local":
        chosen_model = local_model
    else:
        chosen_model = model or _DEFAULT_MODELS[provider]

    header = ["filename"] + fields + ["tags"]

    def make_row(filename: str, parsed: dict | None = None, tags: list | None = None) -> list:
        parsed = parsed or {}
        tag_list = tags if tags is not None else parsed.get("tags", [])
        return [filename] + [parsed.get(f, "") for f in fields] + [", ".join(tag_list)]

    chroma_client = PersistentClient(path=vector_db_path)
    collection = chroma_client.get_or_create_collection(name="lore")

    cross_encoder = None
    if rerank:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(rerank_model)


    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    processed = set()
    if file_exists:
        with open(output_csv, newline="") as rf:
            reader = csv.reader(rf)
            existing_header = next(reader, None)
            if existing_header != header:
                raise SystemExit(
                    f"{output_csv} has columns {existing_header} but this mode produces "
                    f"{header}. Point --tag-output at a fresh file."
                )
            # Keep only rows with real tag content, deduped by filename (last
            # wins). Blank rows from a failed attempt are dropped so the file
            # is retried; dedup prevents a re-run from doubling a filename.
            kept: dict[str, list] = {}
            for row in reader:
                if row and any(cell.strip() for cell in row[1:]):
                    kept[row[0]] = row
        processed = set(kept)
        # Rewrite the cleaned CSV atomically before appending new rows.
        tmp = f"{output_csv}.tmp"
        with open(tmp, "w", newline="") as wf:
            w = csv.writer(wf)
            w.writerow(header)
            w.writerows(kept.values())
        os.replace(tmp, output_csv)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for path in Path(zips_dir).iterdir():
            if path.suffix.lower() not in [".zip", ".rar", ".7z", ".stl", ".obj", ".png"]:
                continue
            if path.name in processed:
                continue

            temp_dir = None
            try:
                temp_dir = extract_to_temp(path)
                if not temp_dir or not is_valid_archive_content(temp_dir):
                    # Not written to the CSV, so a re-run retries it (e.g. an
                    # extractor was missing or the archive was re-downloaded).
                    print(f"Skipping {path.name} — could not extract / no valid content (will retry on re-run)")
                    logging.warning(f"No valid content for {path.name} (not written; retry on re-run)")
                    continue

                # Put the base name (stem) of the file at the front, then add all contained names
                base_name = clean_file_name(path.stem)
                contained_names = " ".join(
                    clean_file_name(f.stem)
                    for f in temp_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() != ".txt"
                )
                joined_names = f"{base_name} {contained_names}".strip()

                # Use a concise query for vector DB lookup based only on file names.

                # Normalize base name to improve substring matching
                normalized_base_name = base_name.replace("_", " ").replace("-", " ")
                # Deduplicate words in normalized_base_name
                words = normalized_base_name.split()
                deduped_words = []
                seen = set()
                for word in words:
                    if word not in seen:
                        deduped_words.append(word)
                        seen.add(word)
                normalized_base_name = " ".join(deduped_words)

                # Build the retrieval query from cleaned filename tokens —
                # print-prep words and digit suffixes only dilute the embedding
                base_words = filter_query_tokens(normalized_base_name.split())
                query_text = " ".join(filter_query_tokens(joined_names.split())) \
                    or normalized_base_name or path.stem

                # Retrieve candidate chunks: prefer chunks from pages whose
                # slug matches any n-gram of the file name (a strong on-topic
                # signal), falling back to an unfiltered semantic query.
                slugs = candidate_slugs(base_words)
                results = None
                if slugs:
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=20,
                        where={"slug": {"$in": slugs}},
                    )
                    if not results["documents"][0]:
                        results = None
                if results is None:
                    results = collection.query(query_texts=[query_text], n_results=20)

                documents = results["documents"][0]
                distances = results["distances"][0]
                metadatas = (results.get("metadatas") or [[]])[0]
                if len(metadatas) != len(documents):
                    metadatas = [None] * len(documents)

                if not documents:
                    print(f"Skipping {path.name} — no lore found in vector DB (will retry on re-run)")
                    logging.warning(f"No lore retrieved for {path.name} (not written; retry on re-run)")
                    continue

                if rerank:
                    scores = cross_encoder.predict([(query_text, d) for d in documents])
                    ranked = sorted(zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True)
                    confident_docs = [(d, m) for d, m, _ in ranked[:8]]
                else:
                    confident_docs = select_context_docs(documents, distances, metadatas)

                context_note = _WEAK_CONTEXT_NOTE if distances[0] > _WEAK_CONTEXT_DISTANCE else ""
                related = related_pages_block([m for _, m in confident_docs])

                # cl100k token counts are approximate for non-OpenAI models, but
                # this is only the context-budget gate, not billing — close enough.
                prompt_tokens = count_tokens(prompt + context_note + related)
                context_budget = token_budget - prompt_tokens - 300
                current = 0
                context_chunks = []
                for doc, _ in confident_docs:
                    t = count_tokens(doc)
                    if current + t > context_budget:
                        break
                    context_chunks.append(doc)
                    current += t

                context = "\n".join(context_chunks)
                full_prompt = (
                    f"{prompt}\n\nThe primary subject in question is \"{normalized_base_name}\".\n\n"
                    f"Secondary subjects could include \"{joined_names}\"\n\n"
                    f"{context_note}{related}"
                    f"Lore context follows until the end of this message:\n{context}\n\n"
                )

                def fallback():
                    # Generation failed (API error / "unknown"). Don't write a
                    # row so the file is retried on the next run.
                    print(f"[Retry on re-run] Generation failed for {path.name}")
                    logging.warning(f"Generation failed for {path.name} (not written; retry on re-run)")

                if provider == "anthropic":
                    # Schema-enforced: the record is guaranteed to match `schema`
                    record, token_count = ask_anthropic(full_prompt, schema, model=chosen_model)
                    if record is None:
                        fallback()
                        continue
                    parsed = normalize_record(record, fields)
                elif provider == "local":
                    raw = ask_local_model(full_prompt, chosen_model)
                    token_count = prompt_tokens + count_tokens(raw)
                    if raw.strip().lower() == "unknown":
                        fallback()
                        continue
                    parsed = parse_structured(raw, fields)
                else:  # openai
                    raw, token_count = ask_openai(full_prompt, model=chosen_model)
                    if raw.strip().lower() == "unknown":
                        fallback()
                        continue
                    parsed = parse_structured(raw, fields)

                writer.writerow(make_row(path.name, parsed))
                processed.add(path.name)
                print(f"Tagged {path.name} [{provider}] using {token_count} tokens")
                logging.info(f"Tagged {path.name} | Tokens: {token_count} | {provider}")
            except Exception as e:
                # Unexpected failure — leave the file out of the CSV so a
                # re-run retries it rather than stranding a blank row.
                print(f"Error processing {path.name} (will retry on re-run): {e}")
                logging.error(f"Tagging failed for {path.name}: {e}")
            finally:
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
