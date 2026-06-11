import argparse
import json
from pathlib import Path

from tagging import run_tagging
from scrape import run_scraping
from embed import run_embedding
from manyfold_ingest import run_upload

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tagging_presets.json"


def load_preset(mode: str) -> dict:
    with open(CONFIG_PATH) as f:
        presets = json.load(f)
    if mode not in presets:
        raise SystemExit(f"Unknown mode '{mode}' — available: {', '.join(presets)}")
    return presets[mode]


def _add_mode(p):
    p.add_argument('--mode', default='warhammer',
                   help='Preset supplying default paths and prompt (warhammer or dnd)')


def _add_lore_dir(p):
    p.add_argument('--lore-dir', help='Directory of scraped lore markdown (default: preset lore_dir)')


def _add_vector_db(p):
    p.add_argument('--vector-db-path', help='Path to vector DB (default: preset vector_db)')


def _add_use_local(p):
    p.add_argument('--use-local', action='store_true', help='Use local models instead of OpenAI')


def _add_scrape_opts(p):
    p.add_argument('--seeds', help='Path to seed URLs file (default: preset seeds)')
    p.add_argument('--max-pages', type=int, default=100)
    p.add_argument('--max-depth', type=int, default=2)


def _add_embed_opts(p):
    p.add_argument('--embed-model', default='BAAI/bge-m3',
                   help='SentenceTransformer embedding model (local mode only)')
    p.add_argument('--min-chunk-tokens', type=int,
                   help='Min tokens per chunk (default: 300 local, 80 otherwise)')
    p.add_argument('--max-chunk-tokens', type=int,
                   help='Max tokens per chunk (default: 800 local, 200 otherwise — '
                        "the default embedder truncates input at 256 wordpieces)")


def _add_tag_opts(p):
    p.add_argument('--zips', required=True, help='Path to folder of ZIPs/STLs to tag')
    p.add_argument('--tag-output', help='Output CSV for tag results (default: preset tag_output)')
    p.add_argument('--prompt-override', help='Custom tag prompt (replaces the preset prompt)')
    p.add_argument('--model', default='gpt-4o', help='OpenAI model to use when not local')
    p.add_argument('--local-model', default='llama3.1:8b-instruct', help='Ollama model name')
    p.add_argument('--rerank', action='store_true', help='Rerank search results with cross-encoder')
    p.add_argument('--rerank-model', default='BAAI/bge-reranker-base', help='Cross-encoder model')
    p.add_argument('--token-budget', type=int, default=3000, help='Context token budget for RAG')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='RAG pipeline that tags miniature files with wiki lore. '
                    'Paths default to the --mode preset in config/tagging_presets.json.'
    )
    sub = parser.add_subparsers(dest='step', required=True)

    p = sub.add_parser('scrape', help='Crawl wiki lore into markdown files')
    _add_mode(p)
    _add_lore_dir(p)
    _add_scrape_opts(p)

    p = sub.add_parser('embed', help='Chunk and embed scraped lore into the vector DB')
    _add_mode(p)
    _add_lore_dir(p)
    _add_vector_db(p)
    _add_use_local(p)
    _add_embed_opts(p)

    p = sub.add_parser('tag', help='Tag miniature files using retrieved lore + an LLM')
    _add_mode(p)
    _add_vector_db(p)
    _add_use_local(p)
    _add_tag_opts(p)

    p = sub.add_parser('upload', help='Upload a tag CSV to Manyfold (dry run)')
    _add_mode(p)
    p.add_argument('--csv', help='Path to tag CSV (default: preset tag_output)')

    p = sub.add_parser('all', help='Run scrape → embed → tag in one go')
    _add_mode(p)
    _add_lore_dir(p)
    _add_vector_db(p)
    _add_use_local(p)
    _add_scrape_opts(p)
    _add_embed_opts(p)
    _add_tag_opts(p)
    p.add_argument('--upload', action='store_true', help='Also run the upload step at the end')

    return parser


def main():
    args = build_parser().parse_args()
    preset = load_preset(args.mode)

    if args.step in ('scrape', 'all'):
        if args.step == 'all':
            print("=== Step 1/3: scrape ===")
        run_scraping(
            args.seeds or preset['seeds'],
            args.lore_dir or preset['lore_dir'],
            args.max_pages,
            args.max_depth,
        )

    if args.step in ('embed', 'all'):
        if args.step == 'all':
            print("=== Step 2/3: embed ===")
        run_embedding(
            args.lore_dir or preset['lore_dir'],
            args.vector_db_path or preset['vector_db'],
            use_local=args.use_local,
            embed_model=args.embed_model,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
        )

    if args.step in ('tag', 'all'):
        if args.step == 'all':
            print("=== Step 3/3: tag ===")
        run_tagging(
            args.zips,
            args.tag_output or preset['tag_output'],
            args.vector_db_path or preset['vector_db'],
            args.prompt_override,
            args.mode,
            use_local=args.use_local,
            local_model=args.local_model,
            model=args.model,
            token_budget=args.token_budget,
            rerank=args.rerank,
            rerank_model=args.rerank_model,
        )

    if args.step == 'upload':
        run_upload(args.csv or preset['tag_output'])
    elif args.step == 'all' and args.upload:
        print("=== Upload ===")
        run_upload(args.tag_output or preset['tag_output'])


if __name__ == "__main__":
    main()
