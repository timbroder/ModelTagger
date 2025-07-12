import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                        and (ext in webpage_extensions or parsed.path.endswith('/'))):
                    new_links.append((link, depth + 1))
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    
    return new_links

def run_scraping(seed_file, output_path, max_pages=100, max_depth=2, max_threads=5):
    with open(seed_file) as f:
        seeds = [line.strip() for line in f if line.strip()]

    visited = set()
    results = []
    queue = [(url, 0) for url in seeds]
    domain = urlparse(seeds[0]).netloc

    # Common web page extensions
    webpage_extensions = {'', '.html', '.htm', '.php', '.asp', '.aspx', '.jsp'}
    # Common image extensions to skip
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_url = {}
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

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)