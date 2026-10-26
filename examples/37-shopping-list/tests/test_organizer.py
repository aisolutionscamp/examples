from pathlib import Path

import pytest

from shopping_list.categories import categorize
from shopping_list.organizer import format_list, load_layout, merge, organize, singular
from shopping_list.parser import Item, parse_text

LAYOUT = Path(__file__).parents[1] / "layouts" / "example-store.json"


@pytest.mark.parametrize("name, category", [
    ("tomatoes", "produce"), ("whole milk", "dairy"), ("ice cream", "frozen"), ("frozen peas", "frozen"),
    ("sourdough bread", "bakery"), ("salmon fillet", "meat_fish"), ("toilet paper", "household"),
    ("spaghetti", "pantry"), ("tomato sauce", "pantry"), ("chocolate ice cream", "frozen"), ("orange juice", "beverages"), ("batteries", "other"),
])
def test_categorize(name, category):
    assert categorize(name) == category


def test_singular():
    assert [singular(w) for w in ["eggs", "tomatoes", "berries", "peaches", "glass"]] == [
        "egg", "tomato", "berry", "peach", "glass"]


def test_merge_sums_same_item_and_unit():
    merged = merge([Item("egg", 2), Item("eggs", 6), Item("milk", 1, "l"), Item("milk", 500, "ml")])
    assert [(i.name, i.quantity, i.unit) for i in merged] == [("egg", 8, None), ("milk", 1, "l"), ("milk", 500, "ml")]


def test_organize_follows_store_layout():
    items = parse_text("ice cream, 2 kg potatoes, milk, bread, soap, 6 eggs, bananas")
    custom = ["frozen", "household", "dairy", "bakery", "produce"]
    sections = organize(items, custom)
    assert list(sections) == ["frozen", "household", "dairy", "bakery", "produce"]
    assert [i.name for i in sections["dairy"]] == ["eggs", "milk"]


def test_layout_file_is_complete_and_validated(tmp_path):
    aisles = load_layout(LAYOUT)
    assert aisles[0] == "produce" and "other" in aisles
    bad = tmp_path / "bad.json"
    bad.write_text('{"aisles": ["produce", "candy"]}')
    with pytest.raises(ValueError):
        load_layout(bad)


def test_format_list():
    text = format_list(organize(parse_text("2 kg potatoes, salmon"), load_layout(LAYOUT)))
    assert text.splitlines() == ["PRODUCE", "  [ ] 2 kg potatoes", "MEAT & FISH", "  [ ] 1 salmon"]
