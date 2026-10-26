"""Turn free text like "2 kg potatoes, milk and 6 eggs" into structured items."""
import re
from dataclasses import dataclass

UNITS = {"kg": "kg", "g": "g", "l": "l", "ml": "ml", "pack": "pack", "packs": "pack", "can": "can",
         "cans": "can", "bottle": "bottle", "bottles": "bottle", "bunch": "bunch", "loaf": "loaf"}
_QTY = re.compile(r"^(?P<qty>\d+(?:[.,]\d+)?)\s*(?:x\s*)?(?P<unit>[a-z]+\b)?\s*(?:of\s+)?(?P<name>.+)$", re.IGNORECASE)


@dataclass
class Item:
    name: str
    quantity: float = 1.0
    unit: str | None = None
    category: str = "other"


def split_items(text: str) -> list[str]:
    parts = re.split(r"[\n,;]+|\band\b", text, flags=re.IGNORECASE)
    return [p.strip(" -*\t.") for p in parts if p.strip(" -*\t.")]


def parse_item(fragment: str) -> Item:
    match = _QTY.match(fragment.strip())
    if not match:
        return Item(name=fragment.strip().lower())
    qty = float(match["qty"].replace(",", "."))
    unit = (match["unit"] or "").lower()
    name = match["name"].strip()
    if unit and unit in UNITS:
        return Item(name=name.lower(), quantity=qty, unit=UNITS[unit])
    # the "unit" was really the first word of the name ("6 eggs")
    full = f"{match['unit']} {name}" if match["unit"] else name
    return Item(name=full.strip().lower(), quantity=qty)


def parse_text(text: str) -> list[Item]:
    return [parse_item(f) for f in split_items(text)]
