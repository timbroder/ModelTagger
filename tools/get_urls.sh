#!/bin/bash

# Config
WIKI_DOMAIN="warhammer40k.fandom.com"
BASE_PATH="/wiki"
API_URL="https://${WIKI_DOMAIN}/api.php"
LIMIT=500  # Max per API request

# Output file
OUTPUT_FILE="all_pages.txt"
> "$OUTPUT_FILE" # Empty the file

# Initialize
CONTINUE=""
TOTAL=0
BATCH=1

echo "Fetching pages from $WIKI_DOMAIN..."

while :; do
    if [ -n "$CONTINUE" ]; then
        URL="${API_URL}?action=query&list=allpages&aplimit=${LIMIT}&format=json&apcontinue=$(printf '%s' "$CONTINUE" | jq -sRr @uri)"
    else
        URL="${API_URL}?action=query&list=allpages&aplimit=${LIMIT}&format=json"
    fi

    echo "Batch $BATCH: Fetching..."
    RESP=$(curl -s "$URL")

    # Extract page titles
    COUNT=$(echo "$RESP" | jq '.query.allpages | length')
    echo "  Found $COUNT pages in this batch."

    echo "$RESP" | jq -r '.query.allpages[].title' | while read -r title; do
        SAFE_TITLE=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1].replace(' ', '_')))" "$title")
        FULL_URL="https://${WIKI_DOMAIN}${BASE_PATH}/${SAFE_TITLE}"
        echo "$FULL_URL" >> "$OUTPUT_FILE"
        echo "    Added: $title"
        TOTAL=$((TOTAL + 1))
    done

    CONTINUE=$(echo "$RESP" | jq -r '.continue.apcontinue // empty')
    echo "  Total so far: $(wc -l < "$OUTPUT_FILE")"

    [ -z "$CONTINUE" ] && break
    BATCH=$((BATCH + 1))
done

echo "Done! All unique page URLs saved to $OUTPUT_FILE"
echo "Total pages found: $(wc -l < "$OUTPUT_FILE")"
