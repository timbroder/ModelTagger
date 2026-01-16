# ModelTagger Manual Test Plan

## Test Environment Setup

### Required Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export MANYFOLD_API_URL="https://your-manyfold-instance.com/api"
export MANYFOLD_API_TOKEN="your-manyfold-api-token"
```

### Required Dependencies
- Python 3.x with all packages from requirements.txt installed
- Ollama installed and running on localhost:11434 (for local model tests)
- Git repository cloned and on branch `claude/manual-test-plan-MBFAe`

### Test Data Requirements
User will provide:
- Seed URLs file for scraping
- Sample ZIP/STL/OBJ/PNG files for tagging
- Expected tag results for validation

---

## Test Plan Structure

Each step will be tested separately with the following approach:
1. **Pre-validation**: Test inputs before they enter the tool
2. **Tool execution**: Run the actual tool
3. **Post-validation**: Verify outputs after the tool completes

---

## STEP 1: SCRAPE

### Test Case 1.1: Happy Path - Valid Seed URLs
**Input:**
- Valid seed file with 2-3 accessible wiki URLs
- Output path: `test_output/scraped_lore.json`
- Max pages: 10
- Max depth: 1

**Command:**
```bash
python src/main.py --step scrape --seeds test_data/seeds.txt --output test_output/scraped_lore.json --max-pages 10 --max-depth 1
```

**Expected:**
- JSON file created with scraped content
- Each entry has `url` and `text` fields
- Progress messages during scraping
- Final save confirmation

**Validation:**
- File exists and is valid JSON
- Contains expected number of entries
- Text content is non-empty

---

### Test Case 1.2: Resume Functionality
**Input:**
- Existing scraped_lore.json from previous run
- Same seed file
- More pages to scrape

**Command:**
```bash
python src/main.py --step scrape --seeds test_data/seeds.txt --output test_output/scraped_lore.json --max-pages 20 --max-depth 1
```

**Expected:**
- Loads existing progress
- Skips already visited URLs
- Appends new results

**Validation:**
- No duplicate URLs in output
- Total URLs > previous run

---

### Test Case 1.3: Edge Case - Empty Seed File
**Input:**
- Empty seed file

**Command:**
```bash
python src/main.py --step scrape --seeds test_data/empty_seeds.txt --output test_output/scraped_lore.json --max-pages 10 --max-depth 1
```

**Expected:**
- Graceful error handling
- Error message about empty seeds
- No crash

---

### Test Case 1.4: Edge Case - Non-existent Seed File
**Input:**
- Path to file that doesn't exist

**Command:**
```bash
python src/main.py --step scrape --seeds test_data/missing.txt --output test_output/scraped_lore.json --max-pages 10 --max-depth 1
```

**Expected:**
- FileNotFoundError with clear message
- No partial output created

---

### Test Case 1.5: Edge Case - Invalid URLs
**Input:**
- Seed file with malformed/unreachable URLs

**Command:**
```bash
python src/main.py --step scrape --seeds test_data/invalid_seeds.txt --output test_output/scraped_lore.json --max-pages 10 --max-depth 1
```

**Expected:**
- Error messages for failed URLs
- Continues with remaining URLs
- Creates output with successful scrapes only

---

### Test Case 1.6: Edge Case - Network Timeout
**Input:**
- Seed file with slow-responding URLs

**Expected:**
- 10-second timeout enforced
- Error logged but doesn't crash
- Continues with next URL

---

## STEP 2: EMBED

### Test Case 2.1: Happy Path - OpenAI Embeddings
**Input:**
- Valid scraped_lore.json from step 1
- Vector DB path: `test_output/vector_db`

**Command:**
```bash
python src/main.py --step embed --output test_output/scraped_lore.json --vector-db-path test_output/vector_db
```

**Expected:**
- Vector DB directory created
- Embeddings generated for all documents
- Progress messages
- Completion confirmation

**Validation:**
- Vector DB directory exists with content
- Can query the collection successfully
- Collection name is "lore"

---

### Test Case 2.2: Happy Path - Local Embeddings
**Input:**
- Valid scraped_lore.json
- Vector DB path: `test_output/vector_db_local`
- Local model flag enabled

**Command:**
```bash
python src/main.py --step embed --output test_output/scraped_lore.json --vector-db-path test_output/vector_db_local --use-local --embed-model BAAI/bge-m3
```

**Expected:**
- Downloads embedding model if needed
- Generates embeddings locally
- Faster than OpenAI for subsequent runs

**Validation:**
- Vector DB created successfully
- No API calls made

---

### Test Case 2.3: Edge Case - Empty JSON File
**Input:**
- Empty JSON array `[]`

**Command:**
```bash
python src/main.py --step embed --output test_data/empty_lore.json --vector-db-path test_output/vector_db_empty
```

**Expected:**
- Graceful handling
- Creates DB but with no embeddings
- Clear message about no content

---

### Test Case 2.4: Edge Case - Malformed JSON
**Input:**
- Invalid JSON file (syntax errors)

**Command:**
```bash
python src/main.py --step embed --output test_data/malformed_lore.json --vector-db-path test_output/vector_db_malformed
```

**Expected:**
- JSON decode error with clear message
- No partial DB created
- No crash

---

### Test Case 2.5: Edge Case - Missing OpenAI API Key
**Input:**
- Valid JSON but OPENAI_API_KEY not set
- Not using --use-local flag

**Command:**
```bash
unset OPENAI_API_KEY
python src/main.py --step embed --output test_output/scraped_lore.json --vector-db-path test_output/vector_db_no_key
```

**Expected:**
- Clear error about missing API key
- No partial processing

---

### Test Case 2.6: Edge Case - Ollama Not Running
**Input:**
- Valid JSON with --use-local flag
- Ollama service not running

**Command:**
```bash
python src/main.py --step embed --output test_output/scraped_lore.json --vector-db-path test_output/vector_db_local_down --use-local
```

**Expected:**
- Connection error to localhost:11434
- Clear error message
- No crash

---

## STEP 3: TAG

### Test Case 3.1: Happy Path - ZIP Files with OpenAI
**Input:**
- Directory with valid ZIP files
- Vector DB from step 2
- Output CSV: `test_output/tags.csv`
- Mode: warhammer

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/vector_db --tag-output test_output/tags.csv --mode warhammer --model gpt-4o
```

**Expected:**
- Extracts each ZIP to temp directory
- Queries vector DB for context
- Generates tags using OpenAI
- Writes to CSV with filename and tags
- Shows token count and cost estimate
- Logs to tagging.log

**Validation:**
- CSV exists with header row
- Each file has entry with tags
- Tags are relevant (user verification)
- Log file contains token/cost info

---

### Test Case 3.2: Happy Path - Local Model
**Input:**
- Same ZIP files
- Same vector DB
- Output CSV: `test_output/tags_local.csv`
- Use local Ollama model

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/vector_db_local --tag-output test_output/tags_local.csv --use-local --local-model llama3.1:8b-instruct --mode warhammer
```

**Expected:**
- Checks for model availability
- Pulls model if needed
- Generates tags locally
- No API costs
- Token count shown

**Validation:**
- CSV created with tags
- No OpenAI API calls made
- Tags quality compared to OpenAI

---

### Test Case 3.3: Happy Path - STL/OBJ/PNG Files
**Input:**
- Directory with loose STL, OBJ, PNG files (no archives)

**Command:**
```bash
python src/main.py --step tag --zips test_data/loose_files --vector-db-path test_output/vector_db --tag-output test_output/tags_loose.csv --mode warhammer
```

**Expected:**
- Processes files directly without extraction
- Tags generated successfully

**Validation:**
- CSV contains all loose files
- Tags are appropriate

---

### Test Case 3.4: Happy Path - Resume Tagging
**Input:**
- Existing tags.csv with some files already processed
- Same ZIP directory with new files added

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/vector_db --tag-output test_output/tags.csv --mode warhammer
```

**Expected:**
- Reads existing CSV
- Skips already processed files
- Appends only new files
- Shows "Skipping X - already processed" messages

**Validation:**
- No duplicate entries
- Existing entries unchanged
- Only new files added

---

### Test Case 3.5: Happy Path - With Reranking
**Input:**
- Valid ZIP files
- Rerank flag enabled

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/vector_db --tag-output test_output/tags_rerank.csv --mode warhammer --rerank --rerank-model BAAI/bge-reranker-base
```

**Expected:**
- Downloads reranker model if needed
- Reranks retrieved documents
- Better context selection
- Tags potentially more accurate

**Validation:**
- Compare tags quality vs non-reranked

---

### Test Case 3.6: Happy Path - DND Mode
**Input:**
- Valid ZIP files
- Mode set to dnd

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/vector_db_dnd --tag-output test_output/tags_dnd.csv --mode dnd
```

**Expected:**
- Uses dnd preset from config
- Different prompt and vector DB
- Tags relevant to D&D context

**Validation:**
- Tags reflect D&D themes

---

### Test Case 3.7: Edge Case - Invalid Archive Content
**Input:**
- ZIP files containing .exe, .bat, .dll files

**Command:**
```bash
python src/main.py --step tag --zips test_data/malicious_zips --vector-db-path test_output/vector_db --tag-output test_output/tags_malicious.csv --mode warhammer
```

**Expected:**
- Detects bad file extensions
- Skips with "invalid content" message
- CSV entry with empty tags
- Logged as skipped
- No execution of malicious files

**Validation:**
- File skipped, not processed
- No temp files left behind

---

### Test Case 3.8: Edge Case - Empty Archives
**Input:**
- ZIP files with no content

**Command:**
```bash
python src/main.py --step tag --zips test_data/empty_zips --vector-db-path test_output/vector_db --tag-output test_output/tags_empty.csv --mode warhammer
```

**Expected:**
- Detects empty archive
- Skips with "invalid content"
- CSV entry with empty tags

---

### Test Case 3.9: Edge Case - Corrupted Archives
**Input:**
- ZIP files that can't be extracted

**Command:**
```bash
python src/main.py --step tag --zips test_data/corrupted_zips --vector-db-path test_output/vector_db --tag-output test_output/tags_corrupted.csv --mode warhammer
```

**Expected:**
- Extraction fails gracefully
- Error logged to tagging.log
- CSV entry with empty tags
- Continues with next file

---

### Test Case 3.10: Edge Case - Missing Vector DB
**Input:**
- Valid ZIP files
- Non-existent vector DB path

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/nonexistent_db --tag-output test_output/tags_nodb.csv --mode warhammer
```

**Expected:**
- Error about missing/empty vector DB
- Clear message to run embed step first
- No processing

---

### Test Case 3.11: Edge Case - No Matching Context
**Input:**
- Valid ZIP with filename that has no matches in vector DB
- Very restrictive filters

**Expected:**
- Falls back to unfiltered query
- Still retrieves some context
- Generates tags with lower confidence
- May show warning in logs

---

### Test Case 3.12: Edge Case - LLM Returns "unknown"
**Input:**
- Files that confuse the LLM
- Very poor vector DB context

**Expected:**
- Detects "unknown" response
- Falls back to Chroma document snippets
- Logs warning about fallback
- CSV gets fallback tags

**Validation:**
- Fallback mechanism works
- CSV has some tags, not empty

---

### Test Case 3.13: Edge Case - Token Budget Exceeded
**Input:**
- Very large context documents
- Small token budget

**Command:**
```bash
python src/main.py --step tag --zips test_data/zips --vector-db-path test_output/vector_db --tag-output test_output/tags_small_budget.csv --mode warhammer --token-budget 500
```

**Expected:**
- Truncates context to fit budget
- Still generates tags
- Warning if context severely limited

---

### Test Case 3.14: Edge Case - OpenAI Rate Limit
**Input:**
- Many files to process
- Hit rate limits

**Expected:**
- Retry logic kicks in (3 retries)
- Exponential backoff (2^attempt seconds)
- Eventually succeeds or fails gracefully
- Partial CSV saved

---

### Test Case 3.15: Edge Case - Network Failure During Tagging
**Input:**
- Valid files
- Simulate network interruption

**Expected:**
- Retry logic for current file
- If all retries fail, marks as "unknown"
- Continues with next file
- Partial results saved to CSV

---

## STEP 4: UPLOAD

### Test Case 4.1: Happy Path - Valid CSV Upload
**Input:**
- Valid tags.csv from step 3
- Manyfold API credentials set

**Command:**
```bash
python src/main.py --step upload --csv test_output/tags.csv
```

**Expected:**
- Reads CSV file
- Authenticates with Manyfold API
- Uploads each entry
- Success messages for each upload
- Summary of uploaded items

**Validation:**
- User verifies files appear in Manyfold
- Tags are correctly applied

---

### Test Case 4.2: Edge Case - Empty CSV
**Input:**
- CSV with only header row

**Command:**
```bash
python src/main.py --step upload --csv test_data/empty_tags.csv
```

**Expected:**
- Detects no data rows
- Message about nothing to upload
- No API calls made

---

### Test Case 4.3: Edge Case - Malformed CSV
**Input:**
- CSV with inconsistent columns, missing data

**Command:**
```bash
python src/main.py --step upload --csv test_data/malformed_tags.csv
```

**Expected:**
- CSV parsing errors caught
- Clear error message
- No partial uploads

---

### Test Case 4.4: Edge Case - Missing API URL
**Input:**
- Valid CSV
- MANYFOLD_API_URL not set

**Command:**
```bash
unset MANYFOLD_API_URL
python src/main.py --step upload --csv test_output/tags.csv
```

**Expected:**
- Error about missing environment variable
- No upload attempted

---

### Test Case 4.5: Edge Case - Missing API Token
**Input:**
- Valid CSV
- MANYFOLD_API_TOKEN not set

**Command:**
```bash
unset MANYFOLD_API_TOKEN
python src/main.py --step upload --csv test_output/tags.csv
```

**Expected:**
- Error about missing token
- No upload attempted

---

### Test Case 4.6: Edge Case - Invalid API Endpoint
**Input:**
- Valid CSV
- MANYFOLD_API_URL points to non-existent server

**Command:**
```bash
export MANYFOLD_API_URL="https://invalid-url-that-does-not-exist.com/api"
python src/main.py --step upload --csv test_output/tags.csv
```

**Expected:**
- Connection error
- Clear error message
- No crash

---

### Test Case 4.7: Edge Case - API Authentication Failure
**Input:**
- Valid CSV
- Invalid API token

**Command:**
```bash
export MANYFOLD_API_TOKEN="invalid-token-12345"
python src/main.py --step upload --csv test_output/tags.csv
```

**Expected:**
- 401/403 authentication error
- Clear message about invalid credentials
- No partial uploads

---

### Test Case 4.8: Edge Case - Network Failure During Upload
**Input:**
- Valid CSV with multiple entries
- Simulate network interruption mid-upload

**Expected:**
- Handles network errors gracefully
- Reports which files failed
- May have partial uploads (some succeeded)

---

## Test Execution Notes

### Test Data Preparation
Before starting tests, create:
```
test_data/
├── seeds.txt (2-3 valid wiki URLs)
├── empty_seeds.txt (empty file)
├── invalid_seeds.txt (malformed URLs)
├── empty_lore.json (empty JSON array)
├── malformed_lore.json (invalid JSON)
├── zips/ (valid ZIP files with STL/OBJ files)
├── loose_files/ (STL, OBJ, PNG files)
├── malicious_zips/ (ZIPs with .exe, .bat files)
├── empty_zips/ (empty ZIP archives)
├── corrupted_zips/ (corrupted archives)
├── empty_tags.csv (header only)
└── malformed_tags.csv (bad CSV structure)

test_output/ (created during tests)
├── scraped_lore.json
├── vector_db/
├── vector_db_local/
├── tags.csv
├── tags_local.csv
└── (various other outputs)
```

### Cleanup Between Tests
- Remove test_output/ directory before major test suites
- Keep test_data/ intact
- Check tagging.log for error messages

### Success Criteria
- All happy path tests pass
- Edge cases handled gracefully with clear error messages
- No crashes or unhandled exceptions
- Partial results saved when possible
- Logs contain useful debugging information

---

## Additional Flags to Facilitate Testing

### Suggested New Flags (if needed):
1. `--dry-run` - Show what would be processed without actually doing it
2. `--skip-extraction` - For tag step, assume files are already extracted
3. `--limit N` - Process only first N files (for quick testing)
4. `--verbose` - More detailed logging output
5. `--no-cleanup` - Keep temp directories after processing (for debugging)

**Question for user:** Should we add any of these flags to make testing easier?

---

## Test Execution Order

1. **Environment Setup** - Verify all dependencies and env vars
2. **SCRAPE Tests** - 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6
3. **EMBED Tests** - 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6
4. **TAG Tests** - 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9 → 3.10 → 3.11 → 3.12 → 3.13 → 3.14 → 3.15
5. **UPLOAD Tests** - 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6 → 4.7 → 4.8

Estimated time: 3-4 hours for full test suite
