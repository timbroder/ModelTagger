import re
from urllib.parse import urlparse, unquote


def slugify(title: str) -> str:
    """Convert ``title`` into a URL-friendly slug.

    This normalizes the string to lowercase, replaces non-alphanumeric
    characters with hyphens, and strips extraneous hyphens from the ends.
    """
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


# Non-article MediaWiki namespaces: image description pages, category
# listings, templates, ... Their bodies are listings/boilerplate. Note
# "Codex:" page titles are NOT a namespace and must be kept.
_NON_ARTICLE_NAMESPACES = {
    "file", "image", "media", "category", "template", "portal", "help",
    "user", "talk", "special", "mediawiki", "module", "timedtext", "forum",
    "lexicanum", "draft", "gadget",
}


def is_article_url(url: str) -> bool:
    """True if the wiki URL points at a main-namespace article."""
    segment = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
    if ":" not in segment:
        return True
    ns = segment.split(":", 1)[0].strip().lower().replace("_", " ")
    return not (ns in _NON_ARTICLE_NAMESPACES or ns.endswith("talk"))


def clean_file_name(name: str) -> str:
    """Remove dates, timestamps, and symbols from a filename stem."""
    name = re.sub(r"\d{4}[-_\.]\d{2}[-_\.]\d{2}", " ", name)  # YYYY-MM-DD or similar
    name = re.sub(r"\d{2}[-_\.]\d{2}[-_\.]\d{4}", " ", name)  # DD-MM-YYYY or MM-DD-YYYY
    name = re.sub(r"\d{8,14}", " ", name)  # Compact dates or timestamps
    name = re.sub(r"\b(19|20)\d{2}\b", " ", name)  # Year alone
    name = re.sub(r"[^0-9a-zA-Z]+", " ", name)  # Replace symbols with spaces
    return re.sub(r"\s+", " ", name).strip()


# Tokens that describe print files rather than the miniature subject
_JUNK_TOKENS = {
    "supported", "presupported", "unsupported", "presup", "sup", "supports",
    "support", "base", "bases", "body", "bodies", "head", "heads", "arm",
    "arms", "leg", "legs", "left", "right", "part", "parts", "bits", "bit",
    "stl", "obj", "lys", "chitubox", "lychee", "final", "fixed", "repaired",
    "hollow", "hollowed", "solid", "raw", "merged", "split", "cut", "uncut",
    "version", "copy", "new", "old", "test", "print", "prints", "file", "files",
}


def filter_query_tokens(words: list[str]) -> list[str]:
    """Drop filename tokens that carry no lore signal: print-prep words,
    anything with digits (versions, dates, print-farm suffixes), and
    fragments. Deduplicates case-insensitively, preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        lw = w.lower()
        if lw in _JUNK_TOKENS:
            continue
        if any(ch.isdigit() for ch in w):
            continue
        if len(w) <= 2:
            continue
        if lw not in seen:
            seen.add(lw)
            out.append(w)
    return out
