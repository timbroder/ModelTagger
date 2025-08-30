#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scrape_sitemap.sh [output_file]
BASE="https://wh40k.lexicanum.com/sitemap"
OUT="${1:-unique_urls.txt}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Extracts <loc>...</loc> values from stdin
extract_locs() {
  sed -n 's:.*<loc>\([^<]*\)</loc>.*:\1:p'
}

echo "Fetching sitemap index..." >&2
curl -fsSL "$BASE/sitemap-index-wh40k.xml" | extract_locs > "$TMP/sitemaps.txt"

: > "$TMP/all.txt"
i=0
while IFS= read -r sm; do
  [ -n "$sm" ] || continue
  i=$((i+1))
  echo "[$i] $sm" >&2
  if [[ "$sm" = *.gz ]]; then
    curl -fsSL "$sm" | gunzip -c | extract_locs >> "$TMP/all.txt"
  else
    curl -fsSL "$sm" | extract_locs >> "$TMP/all.txt"
  fi
done < "$TMP/sitemaps.txt"

LC_ALL=C sort -u "$TMP/all.txt" > "$OUT"
echo "Unique URLs: $(wc -l < "$OUT")" >&2
echo "Wrote $OUT" >&2
