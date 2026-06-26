from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import patoolib
from tqdm import tqdm

from manyfold import ManyfoldClient, ManyfoldError, model_tags
from utils import (
    slugify, clean_file_name, filter_query_tokens,
    ARCHIVE_EXTS as _ARCHIVE_EXTS, LOOSE_EXTS as _LOOSE_EXTS, BAD_EXTS,
    multipart_volume_number, strip_multipart_suffix, extract_nested_archives,
    multipart_volume_siblings,
)

# Structured fields owned by this pipeline across all modes — namespaced tags
# with these prefixes are replaced on re-runs; everything else (manual tags
# added in the Manyfold UI) is preserved.
OWNED_NAMESPACES = [
    "faction", "subfaction", "unit", "model_type", "role", "allegiance",
    "equipment", "creature", "creature_type", "size", "class", "alignment",
]

# Fuzzy-match floor for mapping CSV filenames onto scanned Manyfold model
# names. High on purpose: at thousands of models, tagging the wrong model is
# worse than leaving one untagged.
_MATCH_RATIO = 0.92

# _ARCHIVE_EXTS / _LOOSE_EXTS are imported from utils (Manyfold's supported set).


def build_tags(row: dict) -> list[str]:
    """Flatten a structured CSV row into namespaced + free-form tags."""
    tags: list[str] = []
    for ns in OWNED_NAMESPACES:
        val = (row.get(ns) or "").strip()
        for part in (p.strip() for p in val.split(",")):
            if part:
                tags.append(f"{ns}: {part}")
    for t in (p.strip() for p in (row.get("tags") or "").split(",")):
        if t:
            tags.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def merge_tags(existing: list[str], new: list[str]) -> list[str]:
    """Apply the update-owned-keep-manual policy.

    Tags in pipeline-owned namespaces are replaced by the new set; all other
    existing tags (manual edits) are preserved. Free-form pipeline tags are
    added without removing anything.
    """
    owned_prefixes = tuple(f"{ns}:" for ns in OWNED_NAMESPACES)
    kept = [t for t in existing if not t.lower().startswith(owned_prefixes)]
    seen = {t.lower() for t in kept}
    merged = list(kept)
    for t in new:
        if t.lower() not in seen:
            seen.add(t.lower())
            merged.append(t)
    return merged


def normalize_name(name: str) -> str:
    """Normalize a filename or Manyfold model name for matching.

    Applies the same cleaning the tagging step uses (strip dates, print-farm
    junk, digit suffixes) and dedupes repeated words — so a vendor filename
    like 'space-mongol-blade-master20210330-8016-1n0lirn' matches the model
    Manyfold derives ('Space Mongol Blade Master'), and 'Name_Name+variant'
    matches 'Name'."""
    cleaned = clean_file_name(Path(name).stem.replace("+", " ").replace("_", " "))
    return "-".join(t.lower() for t in filter_query_tokens(cleaned.split()))


def match_model(filename: str, models_by_slug: dict[str, dict]) -> dict | None:
    """Find the Manyfold model for a CSV filename: exact slug, then fuzzy."""
    slug = normalize_name(filename)
    if slug in models_by_slug:
        return models_by_slug[slug]
    best, best_ratio = None, 0.0
    for s, m in models_by_slug.items():
        r = SequenceMatcher(None, slug, s).ratio()
        if r > best_ratio:
            best, best_ratio = m, r
    return best if best_ratio >= _MATCH_RATIO else None


def _model_dir_name(filename: str) -> str:
    """A filesystem-safe, human-readable folder name for the model."""
    stem = strip_multipart_suffix(Path(filename).stem)  # 'Krieg.part01' -> 'Krieg'
    stem = stem.replace("+", " ").replace("_", " ")
    stem = re.sub(r"[^0-9A-Za-z' \-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", stem).strip()
    # Dedupe repeated words case-insensitively (preserving first occurrence and
    # original casing), mirroring normalize_name. Vendor "Name_Name+variant"
    # filenames otherwise double the title, e.g. "Sister Superior Sister
    # Superior". Digits/apostrophes the title legitimately needs are kept.
    seen: set[str] = set()
    words: list[str] = []
    for w in cleaned.split():
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            words.append(w)
    return " ".join(words) or slugify(filename)


def _staged_dir_names(filenames: list[str]) -> dict[str, str]:
    """Map each CSV filename to a unique staged-folder name.

    Since --zips is recursive, two different sources can share a basename
    (e.g. 'General/Foo.zip' and 'Finished scans/Foo.zip'), which both reduce to
    the same _model_dir_name and would collide on one staged folder — silently
    dropping all but the first. Colliding names are disambiguated with their
    relative parent path ('Foo (General)' vs 'Foo (Finished scans)'); the full
    relative path is unique, so distinct sources always get distinct folders.
    Non-colliding names stay clean.
    """
    base = {f: _model_dir_name(Path(f).name) for f in filenames if f}
    counts = Counter(base.values())
    out: dict[str, str] = {}
    for f, b in base.items():
        parent = " ".join(Path(f).parent.parts)
        out[f] = f"{b} ({parent})" if counts[b] > 1 and parent else b
    return out


# Stay well under the common 255-byte per-file filesystem limit so flattened
# names from deeply nested archives don't raise ENAMETOOLONG.
_MAX_NAME_BYTES = 200


def _flat_name(rel: Path) -> str:
    """A single filename for a nested path, joining its parts with '_'.

    Deeply nested archives can produce a join longer than the filesystem's
    255-byte per-file limit; such names are truncated and given a short hash of
    the full relative path so they stay unique and within the limit, preserving
    the extension.
    """
    name = "_".join(rel.parts)
    if len(name.encode("utf-8")) <= _MAX_NAME_BYTES:
        return name
    suffix = rel.suffix
    h = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    budget = _MAX_NAME_BYTES - len(suffix.encode("utf-8")) - len(h) - 1  # 1 for '_'
    stem = name.encode("utf-8")[:max(0, budget)].decode("utf-8", "ignore")
    return f"{stem}_{h}{suffix}"


def _flatten_into_root(root: Path) -> None:
    """Move every file under ``root`` directly into ``root`` (no subfolders).

    Manyfold treats each folder containing 3D files as one model and every
    SUBFOLDER as a separate model, so an archive whose parts live in subfolders
    (Supported/, Large Printers/, ...) fans out into many models. Flattening so
    all files sit at the model folder's root makes Manyfold see exactly one
    model per zip. Flattened names come from each file's path relative to root
    (separators -> '_') so parts from different subfolders can't collide.
    """
    for p in [f for f in root.rglob("*") if f.is_file()]:
        rel = p.relative_to(root)
        if rel.parent == Path("."):
            continue  # already at the root
        flat = root / _flat_name(rel)
        i = 1
        while flat.exists():
            flat = root / f"{i}_{_flat_name(rel)}"
            i += 1
        shutil.move(str(p), str(flat))
    # Drop the now-empty subdirectories (deepest first).
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        d.rmdir()


def stage_into_library(archive: Path, library_path: Path, tags: list[str],
                       dest_name: str | None = None) -> Path:
    """Extract/copy a model's files into the Manyfold library folder.

    Archives are unpacked (a library scan won't look inside zips); loose
    files are copied. Extracted contents are flattened into a single folder so
    Manyfold registers one model per zip (see ``_flatten_into_root``). A
    datapackage.json is written alongside so Manyfold imports the tags at scan
    time. ``dest_name`` overrides the staged folder name so callers can avoid
    basename collisions across --zips subfolders (see ``_staged_dir_names``).
    """
    dest = library_path / (dest_name or _model_dir_name(archive.name))
    if dest.exists():
        return dest  # already staged on a previous run

    ext = archive.suffix.lower()
    tmp = Path(tempfile.mkdtemp())
    try:
        staging = tmp / "model"
        staging.mkdir()
        # Archives, incl. a split set's first volume (e.g. .7z.001) whose ext
        # isn't a recognized archive ext — patoolib detects it by content.
        if ext in _ARCHIVE_EXTS or multipart_volume_number(archive.name) is not None:
            patoolib.extract_archive(str(archive), outdir=str(staging))
            extract_nested_archives(staging)  # unpack inner .zip/.rar bundles
            if any(p.suffix.lower() in BAD_EXTS for p in staging.rglob("*") if p.is_file()):
                raise ManyfoldError(f"Archive contains an executable: {archive.name}")
            _flatten_into_root(staging)
        elif ext in _LOOSE_EXTS:
            shutil.copy(archive, staging / archive.name)
        else:
            raise ManyfoldError(f"Unsupported file type: {archive.name}")

        with open(staging / "datapackage.json", "w", encoding="utf-8") as f:
            json.dump({
                "name": slugify(_model_dir_name(archive.name)),
                "title": _model_dir_name(archive.name),
                "keywords": tags,
            }, f, indent=2)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(dest))
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _collection_name(c: dict) -> str:
    return (c.get("name") or c.get("title") or "").strip()


def run_upload(
    csv_path: str,
    zips_dir: str | None = None,
    library_path: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    check: bool = False,
    delete_source: bool = False,
) -> None:
    """Sync a structured tag CSV into Manyfold.

    Existing models (matched by name) get namespaced tags applied with the
    update-owned-keep-manual policy plus a faction collection. Missing models
    are staged into the library folder (with a datapackage.json carrying
    their tags) and a scan is triggered; run upload again after the scan
    completes to apply collections to the newly scanned models.
    """
    api_url = os.getenv("MANYFOLD_API_URL")
    if not api_url:
        print("Error: MANYFOLD_API_URL environment variable required.")
        return
    client = ManyfoldClient(
        api_url,
        token=os.getenv("MANYFOLD_API_TOKEN"),
        client_id=os.getenv("MANYFOLD_CLIENT_ID"),
        client_secret=os.getenv("MANYFOLD_CLIENT_SECRET"),
        scopes=os.getenv("MANYFOLD_SCOPES", "public read write"),
    )

    if check:
        caps = client.capabilities()
        print("Manyfold API capabilities:")
        for k, v in caps.items():
            print(f"  {k}: {v if v is not None else 'unknown (no OpenAPI spec found)'}")
        return

    library = Path(library_path or os.getenv("MANYFOLD_LIBRARY_PATH") or "") if (
        library_path or os.getenv("MANYFOLD_LIBRARY_PATH")
    ) else None

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Unique staged-folder name per source so same-basename files in different
    # --zips subfolders don't collide on one folder (dropping all but the first).
    staged_names = _staged_dir_names([r.get("filename", "") for r in rows])

    print(f"Loaded {len(rows)} rows from {csv_path}; fetching Manyfold state...")
    models = client.list_models()
    models_by_slug: dict[str, dict] = {}
    for m in models:
        name = m.get("name") or m.get("title") or ""
        if name:
            models_by_slug[normalize_name(name)] = m
    collections = {_collection_name(c).lower(): c for c in client.list_collections()}
    print(f"Manyfold has {len(models)} models, {len(collections)} collections")

    def ensure_collection(name: str) -> dict | None:
        key = name.lower()
        if key in collections:
            return collections[key]
        if dry_run:
            collections[key] = {"name": name, "_planned": True}
            return collections[key]
        created = client.create_collection(name)
        collections[key] = created
        return created

    stats = {"tagged": 0, "unchanged": 0, "staged": 0, "deleted_source": 0,
             "missing_source": 0, "errors": 0}
    actions = 0
    staged_any = False

    for row in tqdm(rows, desc="Syncing"):
        if limit is not None and actions >= limit:
            break
        filename = row.get("filename", "")
        if not filename:
            continue
        # A multi-part RAR set is staged via its first volume, which pulls in
        # the siblings; skip continuation-volume rows (incl. stale ones from a
        # pre-fix CSV) so we don't stage partial duplicates.
        vol = multipart_volume_number(filename)
        if vol is not None and vol != 1:
            continue
        tags = build_tags(row)
        try:
            model = match_model(filename, models_by_slug)
            if model is not None:
                # The list item is minimal; fetch the detail view for the
                # current keywords + collection (isPartOf), so we honor the
                # update-owned-keep-manual policy and don't re-assign a
                # collection the user set by hand.
                detail = client.get_model(model) if not dry_run else model
                existing = model_tags(detail)
                merged = merge_tags(existing, tags)
                attributes: dict = {}
                if sorted(merged) != sorted(existing):
                    attributes["keywords"] = merged
                faction = (row.get("faction") or "").strip()
                if faction and not detail.get("isPartOf"):
                    coll = ensure_collection(faction)
                    cid = (coll or {}).get("@id") or (coll or {}).get("id")
                    if cid is not None:
                        attributes["isPartOf"] = {"@id": cid, "@type": "Collection"}
                if not attributes:
                    stats["unchanged"] += 1
                    continue
                actions += 1
                if dry_run:
                    print(f"[DRY RUN] would update {filename}: {sorted(attributes)}")
                else:
                    client.update_model(model, attributes)
                stats["tagged"] += 1
            else:
                if library is None:
                    print(f"Skipping {filename} — not in Manyfold and no --library-path to stage into")
                    stats["missing_source"] += 1
                    continue
                source = Path(zips_dir) / filename if zips_dir else None
                if source is None or not source.exists():
                    print(f"Skipping {filename} — source file not found (need --zips)")
                    stats["missing_source"] += 1
                    continue
                actions += 1
                if dry_run:
                    extra = " then delete source" if delete_source else ""
                    print(f"[DRY RUN] would stage {filename} into {library} with {len(tags)} tags{extra}")
                else:
                    stage_into_library(source, library, tags,
                                       dest_name=staged_names.get(filename))
                    staged_any = True
                    # Only remove the source AFTER a successful stage into B —
                    # and remove every volume of a multi-part set, not just the
                    # first (which would orphan its siblings on disk).
                    if delete_source:
                        for v in multipart_volume_siblings(source):
                            v.unlink(missing_ok=True)
                            stats["deleted_source"] += 1
                stats["staged"] += 1
        except Exception as e:
            # One bad archive or API hiccup must not kill a thousands-row sync
            print(f"Error syncing {filename}: {e}")
            stats["errors"] += 1

    if staged_any and not dry_run:
        if client.trigger_scan():
            print("Library scan triggered. Re-run upload after it completes to apply collections to new models.")
        else:
            print("No scan endpoint available — trigger a library scan in the Manyfold UI, then re-run upload.")

    deleted = f", {stats['deleted_source']} sources deleted" if delete_source else ""
    print(
        f"Done: {stats['tagged']} updated, {stats['unchanged']} unchanged, "
        f"{stats['staged']} staged for scan{deleted}, {stats['missing_source']} missing source, "
        f"{stats['errors']} errors" + (" [dry run]" if dry_run else "")
    )
