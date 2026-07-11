import re
import shutil
from pathlib import Path
from urllib.parse import urlparse, unquote

import patoolib


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


# Multi-volume archive members, two schemes:
#   - RAR new-style:  name.part1.rar / name.part01.rar ... name.partN.rar
#   - split archives: name.7z.001 / name.zip.001 / name.rar.001 ... .NNN
# The archiver pulls the whole set from the FIRST volume (its siblings live
# alongside), so we process only volume 1, skip the rest, and strip the volume
# marker from derived model names. A space ("Foo Part 3.rar") is a name, not a
# volume marker, so it's left alone.
_MULTIPART_RE = re.compile(
    r"(?:\.part(\d+)\.rar|\.(?:7z|zip|rar)\.(\d+))$", re.IGNORECASE
)


def multipart_volume_number(name: str) -> int | None:
    """Volume number if ``name`` is a multi-volume archive member, else None.

    Handles ``name.partN.rar`` and split ``name.<ext>.NNN`` (7z/zip/rar) sets.
    """
    m = _MULTIPART_RE.search(name)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def strip_multipart_suffix(stem: str) -> str:
    """Drop a trailing volume marker from a filename stem.

    'Krieg.part01' -> 'Krieg' (RAR new-style; the .rar is already gone from the
    stem) and 'Grey Knights.7z' -> 'Grey Knights' (split ``.<ext>.NNN`` sets
    leave the archive ext exposed in the stem after the .NNN is dropped).
    """
    stem = re.sub(r"\.part\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.(?:7z|zip|rar)$", "", stem, flags=re.IGNORECASE)
    return stem


def multipart_volume_siblings(path: Path) -> list[Path]:
    """Every volume file of the multi-volume set ``path`` belongs to (including
    itself), or just ``[path]`` if it isn't a multi-volume member.

    Lets --delete-source remove the whole set, not only the first volume (which
    would orphan e.g. Krieg.part02.rar … or Grey Knights.7z.002 …).
    """
    m = _MULTIPART_RE.search(path.name)
    if not m:
        return [path]
    prefix = re.escape(path.name[:m.start()])
    if m.group(1) is not None:                       # name.partN.rar
        pat = re.compile(rf"^{prefix}\.part\d+\.rar$", re.IGNORECASE)
    else:                                            # name.<ext>.NNN
        ext = re.escape(m.group(0).lstrip(".").split(".")[0])
        pat = re.compile(rf"^{prefix}\.{ext}\.\d+$", re.IGNORECASE)
    return sorted(p for p in path.parent.iterdir() if p.is_file() and pat.match(p.name))


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

# Loose files that ride along with a model (meshes plus their render images).
# Images are NOT standalone models — they only accompany a real model inside an
# archive/folder. Documents/video/PCB files likewise ride along but are never a
# model on their own.
LOOSE_EXTS = MODEL_EXTS | IMAGE_EXTS

# Files that can be a model ON THEIR OWN when found loose under --zips: a mesh/
# slicer file, or an archive to unpack. A lone image is a render, not a model,
# so it is deliberately excluded — otherwise every stray .jpg/.png becomes its
# own Manyfold model (see ModelTagger2-2pu).
DISCOVERABLE_EXTS = MODEL_EXTS | ARCHIVE_EXTS

# Loose files plus archives — the widest set (used where images-inside count).
TAGGABLE_EXTS = LOOSE_EXTS | ARCHIVE_EXTS

# Executables never belong in a model archive — reject archives containing them.
BAD_EXTS = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".msi", ".scr", ".js", ".dll", ".sh", ".ps1",
})


def _is_archive_member(name: str) -> bool:
    """True if ``name`` is an unpackable archive — a normal archive ext or any
    volume of a multi-volume set (.partN.rar / .<ext>.NNN)."""
    return Path(name).suffix.lower() in ARCHIVE_EXTS or multipart_volume_number(name) is not None


def extract_nested_archives(root: Path, max_depth: int = 3) -> None:
    """Recursively unpack archives found INSIDE ``root`` (in place).

    Patreon repacks often bundle inner .zip/.rar/.7z (one per sub-model) with
    the real model files a level deeper, so a single extract leaves only inner
    archives and the content looks empty. This unpacks them (bounded depth),
    removing each archive once extracted. Corrupt/incomplete inner archives are
    skipped (siblings still get unpacked); only a multi-volume set's first
    volume is extracted, and any leftover archive shells / continuation volumes
    are dropped so they aren't flattened into the model.
    """
    for _ in range(max_depth):
        firsts = [p for p in root.rglob("*")
                  if p.is_file() and _is_archive_member(p.name)
                  and (multipart_volume_number(p.name) or 1) == 1]
        if not firsts:
            break
        progressed = False
        for arc in firsts:
            if not arc.exists():
                continue
            out = arc.parent / f"{arc.name}__unpacked"
            try:
                out.mkdir(exist_ok=True)
                patoolib.extract_archive(str(arc), outdir=str(out))
            except Exception:
                shutil.rmtree(out, ignore_errors=True)
                continue  # corrupt/incomplete inner archive — skip, keep the rest
            arc.unlink()
            progressed = True
        if not progressed:
            break
    for p in root.rglob("*"):
        if p.is_file() and _is_archive_member(p.name):
            p.unlink(missing_ok=True)


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
