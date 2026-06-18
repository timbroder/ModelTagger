import sys

sys.path.append('src')

from tagging import titlecase_tag, clean_equipment, clean_tags, normalize_record

WH_FIELDS = ["faction", "subfaction", "unit", "model_type", "role", "allegiance", "equipment"]


def test_titlecase_preserves_inner_caps_and_small_words():
    assert titlecase_tag("white grey armour") == "White Grey Armour"
    assert titlecase_tag("master of the hunt") == "Master of the Hunt"
    assert titlecase_tag("Adeptus Mechanicus trained") == "Adeptus Mechanicus Trained"
    assert titlecase_tag("kor'sarro khan") == "Kor'sarro Khan"  # apostrophe kept


def test_clean_equipment_dedupes_substrings_and_cases():
    out = clean_equipment("Claws, Oversized Claws, power armour, Power Armour")
    items = out.split(", ")
    assert "Oversized Claws" in items
    assert "Claws" not in items            # substring of "Oversized Claws"
    assert items.count("Power Armour") == 1  # case-insensitive dedupe


def test_clean_tags_drops_field_duplicates():
    fields = ["Space Marines", "Wolfspear", "Techmarine", "", "", "", ""]
    tags = ["Space Wolves Successor", "Wolfspear", "Techmarine", "Ultima Founding"]
    out = clean_tags(tags, fields)
    assert "Wolfspear" not in out          # duplicates subfaction field
    assert "Techmarine" not in out         # duplicates unit field
    assert "Space Wolves Successor" in out
    assert "Ultima Founding" in out


def test_clean_tags_drops_print_and_meta_junk():
    out = clean_tags(
        ["Warhammer 40K", "Custom Sculpt", "Supported", "Tabletop", "Khorne", "presupported"],
        ["", "", "", "", "", "", ""],
    )
    assert out == ["Khorne"]


def test_clean_tags_normalizes_case_and_dedupes():
    out = clean_tags(["melee focused", "Melee Focused", "blood warriors"], [""] * 7)
    assert out == ["Melee Focused", "Blood Warriors"]


def test_normalize_record_end_to_end_cleanup():
    # Mirrors the real Sister Superior / space-mongol style noise
    data = {
        "faction": "Adepta Sororitas",
        "unit": "Sister Superior",
        "equipment": ["Chainsword", "chainsword", "Bolt Pistol", "Pistol"],
        "tags": ["Sisters of Battle", "Sister Superior", "Warhammer 40K",
                 "Supported", "Veteran", "veteran"],
    }
    out = normalize_record(data, WH_FIELDS)
    # equipment: case dedupe + substring dedupe ("Pistol" inside "Bolt Pistol")
    eq = out["equipment"].split(", ")
    assert "Bolt Pistol" in eq and "Pistol" not in eq
    assert eq.count("Chainsword") == 1
    # tags: field dup ("Sister Superior") + faction-synonym handled? "Sisters of
    # Battle" is NOT the faction value (that's "Adepta Sororitas") so it stays,
    # but the unit dup and the junk are gone; casing deduped.
    assert "Sister Superior" not in out["tags"]
    assert "Warhammer 40K" not in out["tags"]
    assert "Supported" not in out["tags"]
    assert out["tags"].count("Veteran") == 1
