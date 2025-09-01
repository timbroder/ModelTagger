# 🧠 Warhammer & D&D Miniature Tagger

This project uses a Retrieval-Augmented Generation (RAG) pipeline to:
1. Scrape lore from Fandom / Lexicanum
2. Embed it in a vector database (Chroma)
3. Analyze STL/OBJ/image ZIPs and generate intelligent tag suggestions
4. Upload tags to Manyfold (optional)

---

## 📁 Project Structure

```
warhammer_tagger/
├── src/                    # Core logic (scrape, embed, tag, upload)
├── seeds/                 # Lore seed URLs (D&D, Warhammer)
├── config/                # Prompt presets and vector paths
├── data/zips/             # STL, OBJ, PNG, or archive files to tag
├── outputs/               # Generated tag CSV
├── requirements.txt       # Python dependencies
├── README.md              # You're here!
```

---

## ⚙️ Usage

### 1. Scrape Lore

```bash
python src/main.py --step scrape --seeds seeds/warhammer_seeds.txt --output outputs/lore.json
```

### 2. Embed Lore into Vector DB

```bash
python src/main.py --step embed --output outputs/lore.json --vector-db-path .chroma/warhammer
```

> **Updating old embeddings:** If you embedded lore before the `slug` field was
> introduced, you can add slugs to existing records without re-running the
> entire embedding step. Iterate over your Chroma collection, derive a slug from
> each `source` URL, and update the metadata in place:

```python
from urllib.parse import urlparse
from chromadb import PersistentClient
from utils import slugify

client = PersistentClient(path=".chroma/warhammer")
col = client.get_collection("lore")
items = col.get(include=["metadatas"])

for item_id, meta in zip(items["ids"], items["metadatas"]):
    url = meta["source"]
    slug = slugify(urlparse(url).path.rstrip("/").split("/")[-1])
    col.update(ids=[item_id], metadatas=[{**meta, "slug": slug}])
```

Or run the provided utility to backfill slugs automatically:

```bash
python src/backfill_slugs.py --vector-db-path .chroma/warhammer --collection lore
```

### 3. Tag Miniature Files

```bash
python src/main.py --step tag --zips data/zips --output outputs/tags.csv --vector-db-path .chroma/warhammer --mode warhammer
```

### 4. Upload to Manyfold

```bash
python src/main.py --step upload --csv outputs/tags.csv
```

Set `MANYFOLD_API_URL` and `MANYFOLD_API_TOKEN` in your environment.

---

## 🔁 Supported Inputs

- `.zip`, `.rar`, `.7z` archives
- `.stl`, `.obj`, `.png` loose files

---

## 🔄 RAG Flow Diagram

- `main.py` dispatches each step
- `scrape.py` crawls lore pages
- `embed.py` stores paragraphs in Chroma
- `tagging.py` uses GPT + Chroma to suggest tags
- `manyfold_ingest.py` checks if model exists, then uploads

---

## 🧠 Tag Prompt Logic

You can override prompts or use presets in `config/tagging_presets.json`.

--- 

## ✅ Coming Next

- Docker support
- Folder watching for auto-trigger
- Vision tagging with clip embeddings
