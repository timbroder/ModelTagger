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
- a folder of **miniature files** to tag — archives (`.zip`, `.rar`, `.7z`, …) or
  loose model/slicer/image files (see [Supported Inputs](#-supported-inputs))

## 🛠 Setup

```bash
pip install -r requirements.txt        # core (Anthropic + OpenAI tagging)
pip install -r requirements-local.txt  # + local mode (SentenceTransformers, reranker)
```

Tag *generation* needs an API key for the chosen `--provider`:
`ANTHROPIC_API_KEY` (default) or `OPENAI_API_KEY`. Embeddings always run
locally — Chroma's built-in model by default, or a SentenceTransformer with
`--use-local` — so no key is needed for scrape/embed.

Keys can live in a **`.env`** file (loaded automatically): copy
`.env.example` to `.env` and fill it in. Exported environment variables
always take precedence over the file, and `.env` is gitignored.

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
  skip the live site (Cloudflare-walled) and use Wayback snapshots, newest
  first. The snapshot list for the whole domain is bulk-fetched from the CDX
  API once at startup and cached as `_snapshots_<domain>.json` in the output
  dir (delete to refresh), so each page costs exactly one snapshot fetch —
  no per-page CDX queries.
- **Transient failures requeue.** A network-level failure puts the page back
  in the queue (up to 3 tries per run) instead of skipping it; only "no
  snapshot exists" is a permanent skip. Skipped pages are never written to
  `_visited.txt`, so a later resume retries them too.
- **Rate limiting.** Wayback requests are globally limited to ~1/sec with
  shared, jittered exponential backoff on 429/503; live sites get a 1 req/sec
  per-domain politeness delay. Expect roughly one page per second.
- Non-article namespace URLs (`File:`, `Category:`, ...) are skipped at
  scrape time — both in seeds and discovered links.
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
python src/main.py tag --zips data/zips --mode warhammer   # Anthropic (default)
```

The tagging backend is chosen with `--provider`:

| `--provider` | Model | Env | Notes |
|---|---|---|---|
| `anthropic` (default) | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | **Schema-enforced** structured output — `faction`/`role`/etc. are JSON enums the model can't deviate from |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | JSON requested in the prompt, parsed defensively |
| `local` | Ollama (`--local-model`) | — | Same as `--use-local` |

Override the model with `--model` (e.g. `--model claude-opus-4-8` for the
strongest subfaction accuracy). The cost across a whole library is dominated
by the number of minis, not the model — a few tens of dollars even on Opus —
so pick for quality.

For each archive/file: extracts it, validates the contents (must contain a
3D model or image; archives with executables are skipped), cleans the
filenames (dates, symbols, and print-prep junk like "presupported"/"v2" are
dropped), retrieves the most relevant lore chunks from Chroma — preferring
pages whose name matches any word or bigram of the file name, falling back
to a pure semantic query — and asks the LLM for tags within
`--token-budget`. Context keeps everything close to the best match, capped
at 2 chunks per page so one article can't crowd out the rest.

The LLM returns a **structured JSON record** so models can be organized
consistently — for warhammer mode: `faction` (a canonical ~25-faction
vocabulary, so "Sisters of Battle" always lands as "Adepta Sororitas"),
`subfaction`, `unit`, `model_type`, `role` (battlefield role), `allegiance`
(Imperium/Chaos/Xenos), `equipment`, plus free-form `tags`. D&D mode uses
creature/creature_type/size/class/alignment. Fields the model can't
determine stay empty rather than guessed.

The field schema (including the enum vocabularies) lives in each preset in
`config/tagging_presets.json` and is the source of truth for the CSV
columns. With `--provider anthropic`, that schema is sent to the API as a
**hard constraint** — `faction`/`model_type`/`role`/`allegiance` can only
ever be one of their allowed values, so the taxonomy is guaranteed at the
API level rather than hoped for in a prompt.

- Results append to the `--tag-output` CSV with one column per field plus a
  trailing `tagged_at` timestamp (ISO-8601 UTC) recording when each row was
  written (default: the preset's `tag_output`, e.g. `tags-warhammer.csv`);
  already-listed files are skipped, so re-running resumes where it left off.
  A CSV written before `tagged_at` existed is migrated in place on the next
  run. Changing modes against an existing CSV with different columns aborts
  with a clear error.
- The prompt includes the retrieved pages' titles and wiki categories (a
  direct faction/role taxonomy signal), and when even the best lore match is
  distant — typical for third-party minis with no wiki page — the model is
  told to ignore the context rather than weave noise into the tags.
- `--mode warhammer|dnd` selects the prompt and field schema from
  `config/tagging_presets.json`; `--prompt-override` replaces the prompt.
- If generation fails, raw lore snippets are written as fallback tags and the
  failure is logged to `tagging.log`.
- A **progress bar** (tqdm) shows overall tagging progress; per-file messages
  print above it.
- Skips are logged distinctly so corrupt vs empty are greppable in
  `tagging.log`: `Extraction failed for …` (corrupt/incomplete archive),
  `No model content for …` (extracted but nothing taggable), and
  `Extraction error for … : <cause>` (the raw extractor message).

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
# verify what your instance's API supports first
python src/main.py upload --check

# preview the sync without writing anything
python src/main.py upload --mode warhammer --zips data/zips \
    --library-path /path/to/manyfold/library --dry-run

# then for real
python src/main.py upload --mode warhammer --zips data/zips \
    --library-path /path/to/manyfold/library
```

Environment: `MANYFOLD_API_URL` plus either `MANYFOLD_API_TOKEN` or
`MANYFOLD_CLIENT_ID`/`MANYFOLD_CLIENT_SECRET` (OAuth client credentials).
`MANYFOLD_LIBRARY_PATH` can replace `--library-path`.

How the sync works:

- **Models already in Manyfold** (matched by normalized name, with a
  conservative fuzzy fallback) get their tags updated with namespaced tags
  from the CSV (`faction: Adepta Sororitas`, `unit: Sister Superior`, ...).
  The pipeline only ever replaces tags in namespaces it owns — anything you
  added by hand in the Manyfold UI survives re-runs. Models also get
  assigned to a collection named after their faction (created on demand)
  unless you already put them in one.
- **Models not in Manyfold yet** are staged into the library folder:
  archives are unpacked into a folder per model with a `datapackage.json`
  carrying the tags (Manyfold imports it at scan time), then a library scan
  is triggered via the API (or you're told to trigger one in the UI).
  Re-run `upload` after the scan completes to apply collections to the
  newly scanned models.
- Built for large libraries: rate-limited, retrying, resumable (`--limit`
  for incremental runs, re-staging and re-tagging are no-ops when nothing
  changed), and `--dry-run` reports the full plan without writing.
- `--delete-source` removes each source archive *after* it's successfully
  staged into the library, so you don't keep both the archive and the
  unpacked copy on disk. Off by default; never deletes on `--dry-run` or on a
  failed stage.

> Note: Manyfold's HTTP API doesn't support file uploads (as of v0.118), so
> staging requires filesystem access to the library folder. `--check` probes
> your instance's OpenAPI spec to confirm what it supports.

---

## 🔁 Supported Inputs

Mirrors [Manyfold's supported formats](https://manyfold.app/manual/supported_formats.html)
(centralized in `utils.py`):

- **Archives**: `.zip`, `.rar`, `.7z`, `.gz`, `.gzip`, `.bz2`
- **3D meshes**: `.stl`, `.obj`, `.3mf`, `.ply`, `.step`/`.stp`, `.gltf`/`.glb`,
  `.fbx`, `.dae`, `.3ds`, `.blend`, `.scad`, `.off`, `.wrl`, `.x3d`, … (full mesh list)
- **Slicer/print projects**: `.chitubox`, `.ctb`, `.lys`, `.lyt`, `.voxl`, `.gcode`
- **Images**: `.png`, `.jpg`/`.jpeg`, `.gif`, `.bmp`, `.tiff`/`.tif`, `.webp`, `.svg`

An archive is tagged if it contains at least one of these (and no executables).
Loose documents/video/PCB files aren't treated as standalone models, but ride
along inside archives.

**Multi-volume archives** — RAR sets (`name.part1.rar … name.partN.rar`) and
split sets (`name.7z.001 … name.7z.NNN`, also `.zip`/`.rar`) — are treated as one
model: only the first volume is processed (the archiver pulls in the rest), and
the volume marker is stripped from the model name.

**Nested archives** — a bundle whose contents are *inner* archives (e.g. an outer
`.rar` holding one `.zip` per sub-model, a common Patreon repack) is unpacked
recursively (bounded depth) so the real model files are found and staged as one
model; corrupt inner archives are skipped, and an executable found at any level
rejects the bundle.

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
