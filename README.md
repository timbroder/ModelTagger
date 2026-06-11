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

Every command takes `--mode warhammer|dnd` (default `warhammer`), which
supplies the default seeds file, lore directory, vector DB path, tag CSV, and
prompt from `config/tagging_presets.json`. Any of those can be overridden
with flags. Run `python src/main.py <command> --help` to see each command's
options.

Run the whole pipeline in one go:

```bash
python src/main.py all --mode warhammer --zips data/zips --max-pages 1000
```

Or step by step:

### 1. Scrape Lore

```bash
python src/main.py scrape --mode warhammer --max-pages 1000 --max-depth 2
```

Writes one markdown file per page (YAML frontmatter with title, URL,
categories, headings, and parsed infobox) into `--lore-dir` (default:
the preset's `lore_dir`, e.g. `lore/warhammer`). `--seeds` overrides the
preset's seeds file.

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
python src/main.py embed --mode warhammer
```

Reads the scrape output (`--lore-dir`: either the markdown directory, or a
legacy `lore.json` from older versions) and writes to `--vector-db-path`
(default: the preset's `vector_db`). The scrape directory is treated as
read-only — all cleanup happens here, at embed time:

- non-article namespace pages (`File:`, `Category:`, `Template:`, ...) are
  skipped — their bodies are listings that pollute retrieval (`Codex:` page
  titles are kept; that's not a namespace)
- wiki noise is stripped (maintenance/portal banner tables, footnote refs,
  link targets, TOC and boilerplate sections like Sources/See also)
- generic site titles from failed extractions are repaired from the URL
- redirect duplicates (same article under different URLs) are deduped,
  keeping the longest copy
- each page also gets a dense summary chunk built from its categories and
  infobox — the most tag-like text on the page

Text is chunked on sentence boundaries with ~20% overlap and a
`Title — Section` prefix, sized to the embedder: 80–200 tokens for Chroma's
default model (which truncates input at 256 wordpieces) or 300–800 with
`--use-local` bge-m3. Override with `--min-chunk-tokens`/`--max-chunk-tokens`.

Re-running is safe — chunks are upserted by ID — but after changing chunk
sizes, embed into a **fresh** `--vector-db-path` (stale chunks from the old
layout are not deleted).

> **Updating old embeddings:** if you embedded lore before the `slug`
> metadata field existed, backfill it without re-embedding:
>
> ```bash
> python src/backfill_slugs.py --vector-db-path .chroma/warhammer --collection lore
> ```

### 3. Tag Miniature Files

```bash
python src/main.py tag --zips data/zips --mode warhammer --model gpt-4o
```

For each archive/file: extracts it, validates the contents (must contain a
3D model or image; archives with executables are skipped), cleans the
filenames (dates, symbols, and print-prep junk like "presupported"/"v2" are
dropped), retrieves the most relevant lore chunks from Chroma — preferring
pages whose name matches any word or bigram of the file name, falling back
to a pure semantic query — and asks the LLM for tags within
`--token-budget`. Context keeps everything close to the best match, capped
at 2 chunks per page so one article can't crowd out the rest.

The LLM returns a **structured JSON record** so models can be organized
consistently — for warhammer mode: `faction` (normalized to a canonical
~25-faction vocabulary, so "Sisters of Battle" always lands as
"Adepta Sororitas"), `subfaction`, `unit`, `model_type`, `role`
(battlefield role), `allegiance` (Imperium/Chaos/Xenos), `equipment`, plus
free-form `tags`. D&D mode uses creature/creature_type/size/class/alignment.
Fields the model can't determine stay empty rather than guessed.

- Results append to the `--tag-output` CSV with one column per field
  (default: the preset's `tag_output`, e.g. `tags-warhammer.csv`);
  already-listed files are skipped, so re-running resumes where it left off.
  Changing modes against an existing CSV with different columns aborts with
  a clear error.
- The prompt includes the retrieved pages' titles and wiki categories (a
  direct faction/role taxonomy signal), and when even the best lore match is
  distant — typical for third-party minis with no wiki page — the model is
  told to ignore the context rather than weave noise into the tags.
- `--mode warhammer|dnd` selects the prompt and field schema from
  `config/tagging_presets.json`; `--prompt-override` replaces the prompt.
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
python src/main.py embed --vector-db-path .chroma/local \
    --use-local --embed-model BAAI/bge-m3

python src/main.py tag --zips data/zips --vector-db-path .chroma/local \
    --use-local --local-model llama3.1:8b-instruct --rerank \
    --rerank-model BAAI/bge-reranker-base --token-budget 3000
```

The tagging step pulls `--local-model` from Ollama automatically if missing.
`--rerank` re-orders retrieved chunks with a cross-encoder before building
the prompt (works in OpenAI mode too).

### 4. Upload to Manyfold

```bash
python src/main.py upload --mode warhammer   # or --csv path/to/tags.csv
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
