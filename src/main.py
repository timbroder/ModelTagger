import argparse

from tagging import run_tagging
from scrape import run_scraping
from embed import run_embedding
from manyfold_ingest import run_upload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True, choices=['scrape', 'embed', 'tag', 'upload'])
    parser.add_argument('--seeds', help='Path to seed URLs file')
    parser.add_argument('--output', help='Directory for scraped lore markdown files')
    parser.add_argument('--tag-output', help='Output tag results')
    parser.add_argument('--max-pages', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--zips', help='Path to folder of ZIPs/STLs')
    parser.add_argument('--vector-db-path', help='Path to vector DB')
    parser.add_argument('--csv', help='Path to CSV for upload step')
    parser.add_argument('--prompt-override', help='Custom tag prompt')
    parser.add_argument('--mode', help='Preset mode (warhammer or dnd)', default='warhammer')
    parser.add_argument('--use-local', action='store_true', help='Use local models instead of OpenAI')
    parser.add_argument('--local-model', default='llama3.1:8b-instruct', help='Ollama model name')
    parser.add_argument('--embed-model', default='BAAI/bge-m3', help='SentenceTransformer embedding model')
    parser.add_argument('--rerank', action='store_true', help='Rerank search results with cross-encoder')
    parser.add_argument('--rerank-model', default='BAAI/bge-reranker-base', help='Cross-encoder model')
    parser.add_argument('--token-budget', type=int, default=3000, help='Context token budget for RAG')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use when not local')

    args = parser.parse_args()

    if args.step == 'scrape':
        run_scraping(args.seeds, args.output, args.max_pages, args.max_depth)
    elif args.step == 'embed':
        run_embedding(
            args.output,
            args.vector_db_path,
            use_local=args.use_local,
            embed_model=args.embed_model,
        )
    elif args.step == 'tag':
        run_tagging(
            args.zips,
            args.tag_output,
            args.vector_db_path,
            args.prompt_override,
            args.mode,
            use_local=args.use_local,
            local_model=args.local_model,
            model=args.model,
            token_budget=args.token_budget,
            rerank=args.rerank,
            rerank_model=args.rerank_model,
        )
    elif args.step == 'upload':
        run_upload(args.csv)

if __name__ == "__main__":
    main()
