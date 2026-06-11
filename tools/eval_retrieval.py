#!/usr/bin/env python3
"""Repeatable retrieval probe for eyeball evaluation.

For each model archive in a folder, runs the exact retrieval path used by the
tagging step (filename cleaning, junk filtering, n-gram slug filter with
unfiltered fallback, rank-delta selection) against a vector DB and prints
what came back: filter used, distances, page titles/sections, and wiki
categories.

Usage:
    python tools/eval_retrieval.py --files /path/to/archives \
        --vector-db-path .chroma/warhammer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chromadb import PersistentClient

from tagging import (
    clean_file_name,
    filter_query_tokens,
    candidate_slugs,
    select_context_docs,
    _WEAK_CONTEXT_DISTANCE,
)

_EXTS = {".zip", ".rar", ".7z", ".stl", ".obj", ".png"}


def probe_file(collection, stem: str, n_results: int = 20, show: int = 6) -> None:
    base = clean_file_name(stem)
    norm = " ".join(dict.fromkeys(base.replace("_", " ").replace("-", " ").split()))
    words = filter_query_tokens(norm.split())
    slugs = candidate_slugs(words)
    query = " ".join(words) or norm or stem

    used = "slug-filtered"
    results = None
    if slugs:
        results = collection.query(query_texts=[query], n_results=n_results,
                                   where={"slug": {"$in": slugs}})
        if not results["documents"][0]:
            results = None
    if results is None:
        used = "UNFILTERED"
        results = collection.query(query_texts=[query], n_results=n_results)

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    print(f"\n### {stem!r}")
    print(f"    query={query!r}")
    print(f"    slugs={slugs} -> {used}")
    if not docs:
        print("    NO RESULTS")
        return
    if dists[0] > _WEAK_CONTEXT_DISTANCE:
        print(f"    WEAK CONTEXT (best {dists[0]:.3f} > {_WEAK_CONTEXT_DISTANCE}) — LLM told to ignore")
    picked = {id(d) for d, _ in select_context_docs(docs, dists, metas)}
    for d, dist, m in list(zip(docs, dists, metas))[:show]:
        mark = "*" if id(d) in picked else " "
        m = m or {}
        print(f"  {mark} {dist:.3f} [{m.get('title')} / {m.get('section')}] "
              f"cats={str(m.get('categories', ''))[:70]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", required=True, help="Folder of model archives to probe")
    parser.add_argument("--vector-db-path", required=True, help="Chroma DB to query")
    parser.add_argument("--n-results", type=int, default=20)
    parser.add_argument("--show", type=int, default=6, help="Result rows to print per file")
    args = parser.parse_args()

    collection = PersistentClient(path=args.vector_db_path).get_or_create_collection("lore")
    print(f"Collection: {collection.count()} chunks")

    files = sorted(
        p for p in Path(args.files).iterdir()
        if p.suffix.lower() in _EXTS
    )
    if not files:
        raise SystemExit(f"No model files found in {args.files}")
    for p in files:
        probe_file(collection, p.stem, n_results=args.n_results, show=args.show)

    print(f"\nProbed {len(files)} files. '*' marks chunks the tagger would actually use.")


if __name__ == "__main__":
    main()
