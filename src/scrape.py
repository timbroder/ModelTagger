import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def scrape_url(args):
    """Helper function to scrape a single URL."""
    url, depth, domain, max_depth, webpage_extensions, image_extensions, visited, results = args
    if url in visited or depth > max_depth:
        return None
    visited.add(url)
    print(f"Scraping ({len(visited)}): {url}")

    new_links = []
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        results.append({'url': url, 'text': '\n'.join(paragraphs)})

        if depth < max_depth:
            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])
                parsed = urlparse(link)
                # Get the file extension (if any)
                ext = parsed.path.lower().split('.')[-1] if '.' in parsed.path else ''
                ext = f'.{ext}' if ext else ''

                # Check if it's a webpage starting with `/wiki` and not an image, and no '#' in URL
                if (parsed.netloc == domain
                        and parsed.path.startswith('/wiki')
                        and ext not in image_extensions
                        and '#' not in link
                        and '?' not in link
                        and (ext in webpage_extensions or parsed.path.endswith('/'))):
                    new_links.append((link, depth + 1))
    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return new_links


def save_progress(output_path, results):
    """Append new results to the output file."""
    try:
        # If the file exists, read its current content and append new results
        existing_results = []
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_results = json.load(f)

        # Combine existing results with the new ones
        all_results = existing_results + results

        # Dump back to the file
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Progress saved to {output_path}")
    except Exception as e:
        print(f"Error saving progress: {e}")


def load_progress(output_path):
    """Helper function to load progress from output file."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
                visited = {entry['url'] for entry in results}
                print(f"Loaded progress from {output_path}, {len(visited)} URLs already processed.")
                return results, visited
        except Exception as e:
            print(f"Error loading progress: {e}")
    return [], set()


def run_scraping(seed_file, output_path, max_pages=100, max_depth=2, max_threads=5, save_interval=10):
    with open(seed_file) as f:
        seeds = [line.strip() for line in f if line.strip()]

    # Load progress from the file if it exists
    results, visited = load_progress(output_path)

    # Initialize the queue with unvisited seed URLs
    queue = [(url, 0) for url in seeds if url not in visited]
    domain = urlparse(seeds[0]).netloc

    # Common web page extensions
    webpage_extensions = {'', '.html', '.htm', '.php', '.asp', '.aspx', '.jsp'}
    # Common image extensions to skip
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}

    if not queue and len(visited) < max_pages:
        print("All seed URLs have already been visited. Attempting to find new links dynamically.")
        for entry in results:
            # Add links from results into the queue that have not been visited
            soup = BeautifulSoup(entry['text'], 'html.parser')
            for a in soup.find_all('a', href=True):
                link = urljoin(entry['url'], a['href'])
                parsed = urlparse(link)
                ext = parsed.path.lower().split('.')[-1] if '.' in parsed.path else ''
                ext = f'.{ext}' if ext else ''

                if (parsed.netloc == domain
                        and parsed.path.startswith('/wiki')
                        and link not in visited
                        and ext not in image_extensions):
                    queue.append((link, 0))  # Reset depth for new links

    with ThreadPoolExecutor(max_threads) as executor, tqdm(
            total=max_pages, initial=len(visited), desc="Scraping"
    ) as pbar:
        future_to_url = {}
        processed_count = 0
        while queue and len(visited) < max_pages:
            while queue and len(future_to_url) < max_threads:
                url, depth = queue.pop(0)
                args = (url, depth, domain, max_depth, webpage_extensions, image_extensions, visited, results)
                future_to_url[executor.submit(scrape_url, args)] = url

            for future in as_completed(future_to_url):
                new_links = future.result()
                if new_links:
                    queue.extend(new_links)
                del future_to_url[future]

                # Increment processed count, update progress bar, and save progress at intervals
                processed_count += 1
                pbar.update(1)
                if processed_count % save_interval == 0:
                    save_progress(output_path, results)
                    results.clear()  # Clear the results list as they are now saved

    # Final save after completion
    save_progress(output_path, results)
    results.clear()