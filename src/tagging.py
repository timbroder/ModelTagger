import os
import tempfile
import shutil
import json
from pathlib import Path
import patoolib
from chromadb import PersistentClient
import openai
import tiktoken  # Optional token estimation
import csv
from urllib.parse import urlparse
import time
import logging

logging.basicConfig(filename='tagging.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

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

def get_tokenizer(model):
    # GPT-4.1 and friends use cl100k_base; update as needed for other future models
    if model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4", "gpt-3.5-turbo"):
        return tiktoken.get_encoding("cl100k_base")
    else:
        return tiktoken.encoding_for_model(model)


def count_tokens(text, model="gpt-4.1"):
    enc = get_tokenizer(model)
    return len(enc.encode(text))

def ask_openai(prompt, model="gpt-4.1", retries=3):
    enc = get_tokenizer(model)
    prompt_tokens = len(enc.encode(prompt))
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a miniature tagging assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            completion_tokens = len(enc.encode(response.choices[0].message["content"]))
            return response.choices[0].message["content"], prompt_tokens + completion_tokens
        except Exception as e:
            print(f"[OpenAI Retry {attempt + 1}] Error: {e}")
            if attempt == retries - 1:
                return "unknown", prompt_tokens
            time.sleep(2 ** attempt)


def run_tagging(zips_dir, output_csv, vector_db_path, prompt_override, mode):
    with open("config/tagging_presets.json") as f:
        presets = json.load(f)[mode]

    vector_db_path = vector_db_path or presets["vector_db"]
    prompt = prompt_override or presets["prompt"]

    client = PersistentClient(path=vector_db_path)
    collection = client.get_or_create_collection(name="lore")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "tags"])

        for path in Path(zips_dir).iterdir():
            if path.suffix.lower() not in [".zip", ".rar", ".7z", ".stl", ".obj", ".png"]:
                continue

            temp_dir = extract_to_temp(path)
            if not temp_dir or not is_valid_archive_content(temp_dir):
                print(f"Skipping {path.name} — invalid content")
                continue

            # Put the base name (stem) of the file at the front, then add all contained names
            base_name = path.stem  # File name without extension
            contained_names = " ".join(
                f.name
                for f in temp_dir.rglob("*")
                if f.is_file() and f.suffix.lower() != ".txt"
            )
            joined_names = f"{base_name} {contained_names}".strip()

            query = f"{prompt}\n{joined_names}"

            results = collection.query(query_texts=[query], n_results=50)
            documents = results["documents"][0]
            distances = results["distances"][0]

            # Only keep docs above confidence (adjust threshold as needed)
            confidence_threshold = 0.15
            confident_docs = [doc for doc, dist in zip(documents, distances) if dist <= confidence_threshold]

            context_chunks = []
            token_budget = 3000
            for doc in confident_docs:
                if count_tokens("\n".join(context_chunks + [doc])) > token_budget:
                    break
                context_chunks.append(doc)

            context = "\n".join(context_chunks)
            full_prompt = f"{prompt}\n\nLore context:\n{context}\n\n"
            tags, token_count = ask_openai(full_prompt)

            if tags.strip().lower() == "unknown":
                print(f"[Fallback] OpenAI failed for {path.name}. Using Chroma document snippets.")
                logging.warning(f"Fallback to Chroma for {path.name}")
                tag_result = [doc[:50] for doc in documents]
                writer.writerow([path.name, "; ".join(tag_result)])
                shutil.rmtree(temp_dir)
                continue

            tag_result = tags.strip().split(",") if "," in tags else tags.strip().split("\n")
            writer.writerow([path.name, "; ".join(tag_result)])
            print(f"Tagged {path.name} using ~{token_count} tokens")
            logging.info(f"Tagged {path.name} | Tokens: {token_count}")
            shutil.rmtree(temp_dir)
