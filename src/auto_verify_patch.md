# ✅ Patch: Auto-Verify Archive Contents Before Tagging

To ensure only valid miniature files are processed, add this content filter to your tagging script.

---

## 🔍 Step 1: Add Verification Function

```python
def is_valid_archive_content(folder):
    allowed_exts = {".stl", ".obj", ".png"}
    bad_exts = {".exe", ".bat", ".js", ".dll"}

    files = list(folder.rglob("*"))
    if not files:
        return False

    for f in files:
        if f.suffix.lower() in bad_exts:
            return False

    return any(f.suffix.lower() in allowed_exts for f in files)
```

---

## ✅ Step 2: Use it in your processing loop

After extracting the archive or wrapping a loose file in a temp folder:

```python
if not is_valid_archive_content(extracted_dir):
    print(f"⚠️ Skipping {file_path.name} — no valid content")
    shutil.rmtree(extracted_dir)
    continue
```

This ensures:
- The archive isn't empty
- There's at least one `.stl`, `.obj`, or `.png`
- There are no unsafe or irrelevant file types

--- 
## ✅ Result

Reduces errors, improves tag quality, and avoids wasting GPT token usage on junk files.
