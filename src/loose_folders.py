"""Boundary detection for loose-file model folders (ModelTagger2-17z).

Loose model files in a directory tree (no archive) are otherwise each
discovered as their own model, exploding a kit into dozens of models + LLM
calls. This resolves a folder tree into the right number of *model units* so a
kit becomes the right number of models, tagged once each and staged (flattened)
as one Manyfold model.

THE RULE (per folder, evaluated in this order):
  1. direct model files  -> ONE model (flatten the whole subtree). "Direct
     files win" even if the folder also has model-bearing subfolders.
  2. drop variant/wrapper folders (OBJ/STL/Supported/...) from the child set.
  3. no real children left -> ONE model (variant-only wrapper).
  4. parts-kitbash (>=3 children, >=60% part-named) -> ONE model.
  5. exactly one real child -> wrapper-collapse: descend, keep the model.
  6. else -> SPLIT: each real child is its own model; recurse.

Each emitted ``ModelUnit`` is rooted at the *shallowest meaningful* folder of
its collapsed chain, so its rel_path's final component is the model name (a
generic wrapper like ``Patreon`` is skipped in favor of the folder below it).
Archives are NOT considered here — they stay one model each via the existing
archive discovery; this only groups loose model files.

Validated by simulation against ./files/ (Barbgants/Biovore, Necron HQ/Troops,
a Hive Tyrant kitbash); see ModelTagger2-17z. The denylist + PART_WORDS are
tunable knobs, not a final taxonomy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from utils import MODEL_EXTS, ARCHIVE_EXTS, multipart_volume_number

# GUARD 1 — variant/wrapper folders. Their files flatten into the parent rather
# than counting as separate models.
_VARIANT_EXACT = {"obj", "stl", "raw", "files", "lychee", "chitubox", "lys"}
_VARIANT_SUBSTR = (
    "supported", "unsupported", "presupported", "pre-supported", "lychee", "chitubox",
)
# Format / print-prep tokens. A folder whose name is built ENTIRELY of these
# (e.g. "stl supp", "lys supp", "stl fin", "STL Fixed") is a variant/wrapper of
# one model, not a distinct model — it carries no model identity, so it folds
# into its parent. Anything with a real word ("Support Weapon", "Genestealer 1")
# is kept. See ModelTagger2-keh.
_VARIANT_TOKENS = {
    "obj", "stl", "raw", "file", "files", "lys", "lyt", "ctb", "chitubox",
    "lychee", "gcode", "3mf",
    "supported", "unsupported", "presupported", "presup", "unsup", "sup",
    "supp", "supps", "nosupp", "supports", "support",
    "fin", "final", "fixed", "fix", "repaired", "hollow", "hollowed", "solid",
    "merged", "cut", "uncut", "split", "ready", "print", "prints", "printed",
    "version", "ver",
}

# GUARD 2 — a folder whose children are mostly part-named is one kitbash model,
# not a split. "part-like" = any token is one of these words.
PART_WORDS = {
    "body", "bodies", "head", "heads", "arm", "arms", "hand", "hands", "leg",
    "legs", "base", "bases", "bust", "torso", "wing", "wings", "gun", "guns",
    "weapon", "weapons", "sword", "swords", "saber", "blade", "blades", "claw",
    "claws", "tail", "tails", "helmet", "shoulder", "shoulders", "backpack",
    "cape", "cloak", "horn", "horns", "spine", "mandible", "mandibles",
    "thorax", "pose", "part", "parts", "rending", "gland", "glands", "hook",
    "hooks", "spike", "spikes", "plate", "plates",
}
_KITBASH_MIN_CHILDREN = 3
_KITBASH_PART_FRACTION = 0.6

# Generic wrapper names that carry no model identity — when a single-child chain
# is topped by one of these, the name (and staging root) falls through to the
# meaningful folder below it.
_GENERIC_NAMES = {
    "patreon", "kickstarter", "myminifactory", "mmf", "files", "file", "stl",
    "obj", "raw", "model", "models", "miniature", "miniatures", "mini", "minis",
    "release", "releases", "monthly", "supported", "unsupported", "presupported",
    "lychee", "chitubox", "lys", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
}

# If a single folder would split into more than this many models, it's flagged
# for a human glance rather than silently committed (it may be an over-split
# kitbash). The units are still produced — the warning rides along.
MAX_SPLIT = 8


@dataclass(frozen=True)
class ModelUnit:
    """One resolved model: ``rel_path`` (POSIX, relative to the scan root) is the
    folder (or root-level loose file) to stage/flatten; ``name`` is its model
    name (== the final component of rel_path by construction)."""
    rel_path: str
    name: str


def _subtree_counts(folder: Path) -> tuple[int, int]:
    """(archive files, loose model files) anywhere under ``folder``.

    A multi-volume set counts as one archive (only its first volume)."""
    archives = models = 0
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        vol = multipart_volume_number(p.name)
        if vol is not None:
            archives += 1 if vol == 1 else 0
        elif suf in ARCHIVE_EXTS:
            archives += 1
        elif suf in MODEL_EXTS:
            models += 1
    return archives, models


def _is_archive_dominated(folder: Path) -> bool:
    """True if ``folder``'s subtree is dominated by archives (archives present
    and at least as many as loose model files).

    Such a folder is an archive COLLECTION, not a loose-file kit: each archive
    is already its own model via the normal archive pass, and any stray loose
    files sitting among them are not a coherent kit. Grouping it would sweep all
    the archives into one merged model, so loose grouping skips it entirely
    (ModelTagger2-89r)."""
    archives, models = _subtree_counts(folder)
    return archives > 0 and archives >= models


def _has_direct_model_files(folder: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in MODEL_EXTS for p in folder.iterdir()
    )


def _is_model_bearing(folder: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in MODEL_EXTS for p in folder.rglob("*")
    )


def _subfolders(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir())


def _is_variant_folder(name: str) -> bool:
    low = name.strip().lower()
    if low in _VARIANT_EXACT or any(s in low for s in _VARIANT_SUBSTR):
        return True
    # A name made up entirely of format/print-prep tokens (ignoring bare
    # numbers) carries no model identity — e.g. "stl supp", "lys supp",
    # "stl fin". A single real word ("Support Weapon") keeps it a model.
    non_digit = [t for t in _tokens(name) if not t.isdigit()]
    return bool(non_digit) and all(t.lower() in _VARIANT_TOKENS for t in non_digit)


def _tokens(name: str) -> list[str]:
    """Split a name on whitespace / _ - . and digit boundaries."""
    parts = re.split(r"[\s_\-.]+|(?<=\D)(?=\d)|(?<=\d)(?=\D)", name)
    return [t for t in parts if t]


def _is_part_like(name: str) -> bool:
    return any(t.lower() in PART_WORDS for t in _tokens(name))


def _is_kitbash(children: list[Path]) -> bool:
    if len(children) < _KITBASH_MIN_CHILDREN:
        return False
    part_like = sum(1 for c in children if _is_part_like(c.name))
    return part_like / len(children) >= _KITBASH_PART_FRACTION


def _is_generic_name(name: str) -> bool:
    low = name.strip().lower()
    if low in _GENERIC_NAMES or _is_variant_folder(name):
        return True
    # No meaningful token (only digits / generic words / short fragments).
    meaningful = [
        t for t in re.split(r"[\s_\-.]+", low)
        if len(t) >= 3 and not t.isdigit() and t not in _GENERIC_NAMES
    ]
    return not meaningful


def _model_bearing_real_children(folder: Path) -> list[Path]:
    """Model-bearing subfolders that aren't variant/wrapper folders."""
    return [
        c for c in _subfolders(folder)
        if _is_model_bearing(c) and not _is_variant_folder(c.name)
    ]


def _resolve_folder(
    folder: Path,
    root: Path,
    top: Path,
    top_name: str,
    units: list[ModelUnit],
    warnings: list[str],
) -> None:
    """Resolve ``folder`` (the current point in a collapsed chain rooted at
    ``top``) into one or more ModelUnits, appending to ``units``."""
    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix()

    # 0. Archive-dominated: an archive COLLECTION, not a loose-file kit. Skip it
    #    (never merge the archives into one model) — the archives are modeled
    #    individually by the normal archive pass; stray loose files are left
    #    un-grouped. See ModelTagger2-89r.
    if _is_archive_dominated(folder):
        archives, models = _subtree_counts(folder)
        warnings.append(
            f"{rel(folder)}: archive-dominated ({archives} archives vs {models} loose "
            f"model files) — skipped loose grouping; archives are modeled individually"
        )
        return

    # 1. Direct model files win: this whole subtree is one model.
    if _has_direct_model_files(folder):
        units.append(ModelUnit(rel(top), top_name))
        return

    real = _model_bearing_real_children(folder)

    # 3. Variant-only (e.g. just an OBJ/ + STL/, or a files/ wrapper): one model.
    if not real:
        units.append(ModelUnit(rel(top), top_name))
        return

    # 4. Parts kitbash: mostly part-named children -> one model, flatten all.
    if _is_kitbash(real):
        units.append(ModelUnit(rel(top), top_name))
        return

    # 5. Single real child: wrapper-collapse. Keep the meaningful top; if the
    #    top is a generic wrapper (Patreon/files/...), advance the staging root
    #    and name to the meaningful child below it.
    if len(real) == 1:
        child = real[0]
        if _is_generic_name(top_name):
            _resolve_folder(child, root, child, child.name, units, warnings)
        else:
            _resolve_folder(child, root, top, top_name, units, warnings)
        return

    # 6. Split: each real child is its own model; recurse from that child.
    if len(real) > MAX_SPLIT:
        warnings.append(
            f"{rel(folder)} split into {len(real)} models "
            f"(> {MAX_SPLIT}) — review before upload (possible over-split kitbash)"
        )
    for child in real:
        _resolve_folder(child, root, child, child.name, units, warnings)


def resolve_model_units(root) -> tuple[list[ModelUnit], list[str]]:
    """Resolve loose-file folders under ``root`` into model units.

    Returns ``(units, warnings)``. Each immediate child of ``root`` starts a
    chain: a loose model file directly in the root stays its own model (as
    today); a folder is resolved via the descend/split rule. Folders with no
    loose model files (e.g. archive-only) produce no unit — archives are handled
    by the existing archive discovery.
    """
    root = Path(root)
    units: list[ModelUnit] = []
    warnings: list[str] = []
    if not root.is_dir():
        return units, warnings
    for child in sorted(root.iterdir()):
        if child.is_file():
            if child.suffix.lower() in MODEL_EXTS:
                units.append(ModelUnit(child.name, child.stem))
        elif child.is_dir() and _is_model_bearing(child):
            _resolve_folder(child, root, child, child.name, units, warnings)
    return units, warnings
