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
from urllib.parse import urlparse
import time
import logging
import re
import requests
from utils import slugify

logging.basicConfig(filename='tagging.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Initialize the OpenAI client using environment variables for configuration
client = OpenAI()

def extract_to_temp(file_path):
    temp_dir = Path(tempfile.mkdtemp())
    ext = file_path.suffix.lower()

    try:
        if ext in [".zip", ".rar", ".7z"]:
            patoolib.extract_archive(str(file_path), outdir=str(temp_dir))
        elif ext in [".stl", ".obj", ".png"]:
            shutil.copy(file_path, temp_dir / file_path.name)
        else:
            return None
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

    return temp_dir

def is_valid_archive_content(folder):
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

def get_tokenizer(model):
    import tiktoken
    if model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4", "gpt-3.5-turbo"):
        return tiktoken.get_encoding("cl100k_base")
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text, model="gpt-4.1"):
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def ensure_local_model(model: str):
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
            json={"model": model, "prompt": prompt, "temperature": 0.2, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "unknown")
    except Exception as e:
        print(f"[Local Model Error] {e}")
        return "unknown"

def ask_openai(prompt, model="gpt-4.1", retries=3):
    enc = get_tokenizer(model)
    prompt_tokens = len(enc.encode(prompt))
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a miniature tagging assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            completion_text = response.choices[0].message.content
            completion_tokens = len(enc.encode(completion_text))
            return completion_text, prompt_tokens + completion_tokens
        except Exception as e:
            print(f"[OpenAI Retry {attempt + 1}] Error: {e}")
            if attempt == retries - 1:
                return "unknown", prompt_tokens
            time.sleep(2 ** attempt)


def run_tagging(
    zips_dir,
    output_csv,
    vector_db_path,
    prompt_override,
    mode,
    use_local: bool = False,
    local_model: str = "llama3.1:8b-instruct",
    model: str = "gpt-4o",
    token_budget: int = 3000,
    rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-base",
):
    with open("config/tagging_presets.json") as f:
        presets = json.load(f)[mode]

    vector_db_path = vector_db_path or presets["vector_db"]
    prompt = prompt_override or presets["prompt"]

    client = PersistentClient(path=vector_db_path)
    collection = client.get_or_create_collection(name="lore")

    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "tags"])

        for path in Path(zips_dir).iterdir():
            if path.suffix.lower() not in [".zip", ".rar", ".7z", ".stl", ".obj", ".png"]:
                continue

            temp_dir = extract_to_temp(path)
            if not temp_dir or not is_valid_archive_content(temp_dir):
                print(f"Skipping {path.name} — invalid content")
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

            slugified_term = slugify(normalized_base_name)
            superset = {}

            # Retrieve candidate chunks from the vector DB that mention the base name.
            # Fall back to the unfiltered query if none are found.
            results = collection.query(
                query_texts=[joined_names],
                n_results=10,
                where_document={"$contains": normalized_base_name},
                where={"slug": {"$eq": slugified_term}},
            )

            documents = results["documents"][0]
            distances = results["distances"][0]

            if documents:
                superset[distances[0]] = results

            results = collection.query(
                query_texts=[joined_names],
                n_results=10,
                where_document={"$contains": normalized_base_name},
            )

            documents = results["documents"][0]
            distances = results["distances"][0]

            if documents:
                superset[distances[0]] = results

            if not documents:
                results = collection.query(
                    query_texts=[joined_names],
                    n_results=10,
                )

                documents = results["documents"][0]
                distances = results["distances"][0]

                superset[distances[0]] = results

            min_results = superset[min(superset.keys())]
            documents = results["documents"][0]
            distances = results["distances"][0]

            # Choose the best result set based on the closest distance
            if rerank:
                from sentence_transformers import CrossEncoder
                ce = CrossEncoder(rerank_model)
                scores = ce.predict([(joined_names, d) for d in documents])
                ranked = [d for d, _ in sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)]
                confident_docs = ranked[:8]
            else:
                filtered = bool(documents)
                confidence_threshold = 0.3 if filtered else 0.1
                confident_docs = [
                    doc for doc, dist in zip(documents, distances) if dist <= confidence_threshold
                ]
                if not confident_docs:
                    confident_docs = [documents[0]]

            chosen_model = local_model if use_local else model
            prompt_tokens = count_tokens(prompt, model=chosen_model)
            context_budget = token_budget - prompt_tokens - 300
            current = 0
            context_chunks = []
            for doc in confident_docs:
                t = count_tokens(doc, model=chosen_model)
                if current + t > context_budget:
                    break
                context_chunks.append(doc)
                current += t

            context = "\n".join(context_chunks)
            full_prompt = (
                f"{prompt}\n\nThe primary subject in question is \"{normalized_base_name}\".\n\n"
                f"Secondary subjects could include \"{joined_names}\"\n\n"
                f"Lore context follows until the end of this message:\n{context}\n\n"
            )

            if use_local:
                tags = ask_local_model(full_prompt, local_model)
                token_count = prompt_tokens + count_tokens(tags, model=chosen_model)
            else:
                tags, token_count = ask_openai(full_prompt, model=model)

            if tags.strip().lower() == "unknown":
                print(f"[Fallback] Generation failed for {path.name}. Using Chroma document snippets.")
                logging.warning(f"Fallback to Chroma for {path.name}")
                tag_result = [doc[:50] for doc in documents[:5]]
                writer.writerow([path.name, "; ".join(tag_result)])
                shutil.rmtree(temp_dir)
                continue

            tag_result = tags.strip().split(",") if "," in tags else tags.strip().split("\n")
            writer.writerow([path.name, "; ".join(tag_result)])
            if use_local:
                print(f"Tagged {path.name} [local] using ~{token_count} tokens")
                logging.info(f"Tagged {path.name} | Tokens: {token_count} | local mode")
            else:
                cost_estimate = token_count / 1000 * 0.01
                print(f"Tagged {path.name} using ~{token_count} tokens (${cost_estimate:.4f})")
                logging.info(
                    f"Tagged {path.name} | Tokens: {token_count} | Cost: ${cost_estimate:.4f}"
                )
            shutil.rmtree(temp_dir)
