import re


def slugify(title: str) -> str:
    """Convert ``title`` into a URL-friendly slug.

    This normalizes the string to lowercase, replaces non-alphanumeric
    characters with hyphens, and strips extraneous hyphens from the ends.
    """
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")
