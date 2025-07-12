import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json

def run_scraping(seed_file, output_path, max_pages=100, max_depth=2):
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

    while queue and len(visited) < max_pages:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        print(f"Scraping ({len(visited)}/{max_pages}): {url}")

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

                    # Check if it's a webpage and not an image, and no '#' in URL
                    if (parsed.netloc == domain
                            and parsed.netloc == urlparse(url).netloc
                            and ext not in image_extensions
                            and '#' not in link
                            and (ext in webpage_extensions or parsed.path.endswith('/'))):
                        queue.append((link, depth + 1))
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)