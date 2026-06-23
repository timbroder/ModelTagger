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


# Multi-part RAR volumes: name.part1.rar / name.part01.rar ... name.partN.rar.
# unrar pulls the whole set from the FIRST volume, so we process only part 1,
# skip the rest, and strip the .partN marker from derived model names. Note a
# space ("Foo Part 3.rar") is a name, not a volume marker, so it's left alone.
_MULTIPART_RE = re.compile(r"\.part(\d+)\.rar$", re.IGNORECASE)


def multipart_volume_number(name: str) -> int | None:
    """Volume number if ``name`` is a ``.partN.rar`` multi-volume member, else None."""
    m = _MULTIPART_RE.search(name)
    return int(m.group(1)) if m else None


def strip_multipart_suffix(stem: str) -> str:
    """Drop a trailing ``.partN`` volume marker from a filename stem
    ('Krieg.part01' -> 'Krieg')."""
    return re.sub(r"\.part\d+$", "", stem, flags=re.IGNORECASE)


def clean_file_name(name: str) -> str:
    """Remove dates, timestamps, and symbols from a filename stem."""
    name = strip_multipart_suffix(name)  # 'Krieg.part01' -> 'Krieg'
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


# --- Supported file formats ------------------------------------------------
# Mirrors Manyfold's supported set (https://manyfold.app/manual/supported_formats.html)
# so the tagger discovers and stages anything Manyfold can index.

# 3D mesh / CAD model formats.
MESH_EXTS = frozenset({
    ".splat", ".spz", ".3ds", ".max", ".3mf", ".amf", ".abc", ".dwg", ".dxf",
    ".iam", ".ipt", ".brep", ".blend", ".jas", ".gml", ".dae", ".dcm", ".x",
    ".drc", ".fbx", ".fcstd", ".f3d", ".f3z", ".gltf", ".glb", ".hfp", ".iges",
    ".igs", ".ifc", ".ldr", ".mpd", ".ma", ".mb", ".mix", ".mha", ".mhd",
    ".lxo", ".nrrd", ".nhdr", ".off", ".xbf", ".scad", ".ply", ".pts", ".mdl",
    ".step", ".stp", ".stl", ".skp", ".sldprt", ".spm", ".wrl", ".vtk", ".vtp",
    ".vtu", ".vti", ".vtr", ".vts", ".obj", ".mtl", ".x3d",
})

# Slicer / print-project formats (ChiTuBox, Lychee, gcode, ...).
SLICER_EXTS = frozenset({".chitubox", ".ctb", ".voxl", ".gcode", ".lys", ".lyt"})

# Raster/vector image formats (renders, references).
IMAGE_EXTS = frozenset({
    ".bmp", ".gif", ".jpg", ".jpeg", ".jpe", ".pjpeg", ".png", ".svg",
    ".tiff", ".tif", ".webp",
})

# Archive/container formats we can unpack.
ARCHIVE_EXTS = frozenset({".zip", ".rar", ".7z", ".gz", ".gzip", ".bz2"})

# Anything that makes a file/archive a printable model.
MODEL_EXTS = MESH_EXTS | SLICER_EXTS

# Loose files worth discovering, tagging, and staging on their own (a model or
# its render). Documents/video/PCB files aren't standalone models, but ride
# along inside archives when present.
LOOSE_EXTS = MODEL_EXTS | IMAGE_EXTS

# Loose files plus archives — everything the tag step walks --zips for.
TAGGABLE_EXTS = LOOSE_EXTS | ARCHIVE_EXTS

# Executables never belong in a model archive — reject archives containing them.
BAD_EXTS = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".msi", ".scr", ".js", ".dll", ".sh", ".ps1",
})


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
