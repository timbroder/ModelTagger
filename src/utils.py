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
