# 🧠 Warhammer & D&D Miniature Tagger

[![Tests](https://github.com/timbroder/ModelTagger/actions/workflows/tests.yml/badge.svg)](https://github.com/timbroder/ModelTagger/actions/workflows/tests.yml)

A Retrieval-Augmented Generation (RAG) pipeline that auto-tags tabletop
miniature files (STL/OBJ/archives) with lore-aligned tags:

1. **Scrape** lore from MediaWiki sites (Fandom, Lexicanum) into markdown files
2. **Embed** it into a Chroma vector database
3. **Tag** miniature files using retrieved lore + an LLM (OpenAI or local Ollama)
4. **Upload** tags to Manyfold *(currently a dry run — see below)*

---

## 📁 Project Structure

```
ModelTagger/
├── src/                   # Pipeline steps (main.py dispatches each one)
│   ├── main.py            # CLI entry point
│   ├── scrape.py          # Wiki crawler (API → HTML → Wayback fallbacks)
│   ├── embed.py           # Chunking + Chroma embedding
│   ├── tagging.py         # RAG retrieval + LLM tagging
│   ├── manyfold_ingest.py # Manyfold upload (dry-run stub)
│   └── backfill_slugs.py  # One-off: add slug metadata to old Chroma DBs
├── config/                # Prompt presets and default vector DB paths
├── tools/                 # Shell scripts to harvest seed URL lists
├── tests/                 # pytest suite (runs offline, all network mocked)
├── requirements.txt       # Core dependencies
└── requirements-local.txt # Extra deps for local mode (torch, etc.)
```

You provide two inputs that are not checked in:

- a **seeds file** — one wiki URL per line (generate one with the
  `tools/` scripts, see below)
- a folder of **miniature files** to tag (`.zip`, `.rar`, `.7z` archives or
  loose `.stl`, `.obj`, `.png` files)

## 🛠 Setup

```bash
pip install -r requirements.txt        # OpenAI mode
pip install -r requirements-local.txt  # + local mode (SentenceTransformers, reranker)
```

OpenAI mode requires `OPENAI_API_KEY` in your environment (used for tag
*generation*; embeddings always run locally — Chroma's built-in model by
default, or a SentenceTransformer with `--use-local`).

---

## ⚙️ Usage

### 1. Scrape Lore

```bash
python src/main.py --step scrape --seeds seeds/warhammer_seeds.txt --output lore/warhammer \
    --max-pages 1000 --max-depth 2
```

Writes one markdown file per page (YAML frontmatter with title, URL,
categories, headings, and parsed infobox) into the output directory.

Worth knowing for long scrapes:

- **Resumable.** Progress is saved every 10 pages to `_visited.txt` in the
  output dir; re-running the same command skips already-scraped URLs. If
  `_visited.txt` is deleted, it is rebuilt from the markdown frontmatter.
- **Fallback chain.** Each page is fetched via the MediaWiki API first, then
  raw HTML, then the most recent Wayback Machine snapshot. Login walls and
  Cloudflare challenges are detected and rejected.
- **Lexicanum goes through Wayback.** `wh40k.lexicanum.com` is hardcoded to
  skip the live site (login wall) and search Wayback's CDX API for a usable
  snapshot, newest first.
- **Rate limiting.** Wayback requests are globally limited to ~1/sec with
  shared exponential backoff on 429/503; live sites get a 1 req/sec per-domain
  politeness delay. Expect roughly one page per second — plan accordingly.
- The crawler follows same-domain `/wiki/` links up to `--max-depth` and stops
  at `--max-pages` total pages.

#### Generating a seeds file

```bash
tools/get_urls.sh         # every page on warhammer40k.fandom.com (allpages API)
tools/scrape_sitemap.sh   # every page on wh40k.lexicanum.com (sitemap)
```

### 2. Embed Lore into Vector DB

```bash
python src/main.py --step embed --output lore/warhammer --vector-db-path .chroma/warhammer
```

`--output` here is the scrape output: either the markdown directory, or a
legacy `lore.json` from older versions. Text is chunked into 300–800-token
windows on sentence boundaries with ~20% overlap, prefixed with the page
title, and stored in a Chroma collection named `lore` (cosine distance).
Re-running is safe — chunks are upserted by ID.

> **Updating old embeddings:** if you embedded lore before the `slug`
> metadata field existed, backfill it without re-embedding:
>
> ```bash
> python src/backfill_slugs.py --vector-db-path .chroma/warhammer --collection lore
> ```

### 3. Tag Miniature Files

```bash
python src/main.py --step tag --zips data/zips --tag-output tags.csv \
    --vector-db-path .chroma/warhammer --mode warhammer --model gpt-4o
```

For each archive/file: extracts it, validates the contents (must contain a
3D model or image; archives with executables are skipped), cleans dates and
symbols out of the filenames, retrieves the most relevant lore chunks from
Chroma (exact slug match → substring match → unfiltered, keeping whichever
set matches closest), and asks the LLM for tags within `--token-budget`.

- Results append to the `--tag-output` CSV (`filename, tags`); already-listed
  files are skipped, so re-running resumes where it left off.
- `--mode warhammer|dnd` selects a prompt preset from
  `config/tagging_presets.json`; `--prompt-override` replaces it entirely.
- If generation fails, raw lore snippets are written as fallback tags and the
  failure is logged to `tagging.log`.

### 🏠 Local Mode (Ollama + bge-m3)

```bash
pip install -r requirements-local.txt

# install and start Ollama
brew install ollama          # macOS
curl -fsSL https://ollama.com/install.sh | sh   # Linux
ollama serve
```

```bash
python src/main.py --step embed --output lore/warhammer --vector-db-path .chroma/local \
    --use-local --embed-model BAAI/bge-m3

python src/main.py --step tag --zips data/zips --tag-output tags.csv \
    --vector-db-path .chroma/local --mode warhammer \
    --use-local --local-model llama3.1:8b-instruct --rerank \
    --rerank-model BAAI/bge-reranker-base --token-budget 3000
```

The tagging step pulls `--local-model` from Ollama automatically if missing.
`--rerank` re-orders retrieved chunks with a cross-encoder before building
the prompt (works in OpenAI mode too).

### 4. Upload to Manyfold

```bash
python src/main.py --step upload --csv tags.csv
```

Set `MANYFOLD_API_URL` and `MANYFOLD_API_TOKEN` in your environment.

> ⚠️ **Dry run only.** This step currently checks whether each model already
> exists in Manyfold and prints what it *would* upload. The actual file
> upload / tag POST is not implemented yet.

---

## 🔁 Supported Inputs

- `.zip`, `.rar`, `.7z` archives
- `.stl`, `.obj`, `.png` loose files

---

## 🧪 Tests

```bash
pip install pytest -r requirements.txt
pytest -q
```

All network and model calls are mocked; no API keys, GPU, or torch install
needed.

---

## 📄 License

[MIT](LICENSE)

---

## ✅ Coming Next

- Complete the Manyfold upload step
- Docker support
- Folder watching for auto-trigger
- Vision tagging with CLIP embeddings
