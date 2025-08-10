import argparse

from tagging import run_tagging
from scrape import run_scraping
from embed import run_embedding
from manyfold_ingest import run_upload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True, choices=['scrape', 'embed', 'tag', 'upload'])
    parser.add_argument('--seeds', help='Path to seed URLs file')
    parser.add_argument('--output', help='Output file for lore ')
    parser.add_argument('--tag-output', help='Output tag results')
    parser.add_argument('--max-pages', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--zips', help='Path to folder of ZIPs/STLs')
    parser.add_argument('--vector-db-path', help='Path to vector DB')
    parser.add_argument('--csv', help='Path to CSV for upload step')
    parser.add_argument('--prompt-override', help='Custom tag prompt')
    parser.add_argument('--mode', help='Preset mode (warhammer or dnd)', default='warhammer')

    args = parser.parse_args()

    if args.step == 'scrape':
        run_scraping(args.seeds, args.output, args.max_pages, args.max_depth)
    elif args.step == 'embed':
        run_embedding(args.output, args.vector_db_path)
    elif args.step == 'tag':
        run_tagging(args.zips, args.tag_output, args.vector_db_path, None, args.mode)
    elif args.step == 'upload':
        run_upload(args.csv)

if __name__ == "__main__":
    main()
