"""Categorize, merge duplicates and order the list by the aisle sequence of one store."""
import json
from collections import OrderedDict
from pathlib import Path

from .categories import CATEGORIES, categorize
from .parser import Item


def load_layout(path: str | Path) -> list[str]:
    """A layout is the order in which you walk past the categories in a specific store."""
    aisles = json.loads(Path(path).read_text(encoding="utf-8"))["aisles"]
    unknown = [a for a in aisles if a not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown categories in layout: {unknown}")
    # categories missing from the layout go to the end, in default order
    return aisles + [c for c in CATEGORIES if c not in aisles]


def singular(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith(("oes", "ches", "shes", "xes")):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def merge(items: list[Item]) -> list[Item]:
    merged: OrderedDict[tuple[str, str | None], Item] = OrderedDict()
    for item in items:
        key = (singular(item.name), item.unit)  # "egg"/"eggs" count as one item
        if key in merged:
            merged[key].quantity += item.quantity
        else:
            merged[key] = Item(item.name, item.quantity, item.unit, item.category)
    return list(merged.values())


def organize(items: list[Item], aisles: list[str] | None = None) -> OrderedDict[str, list[Item]]:
    order = aisles or list(CATEGORIES)
    for item in items:
        if item.category == "other":
            item.category = categorize(item.name)
    sections: OrderedDict[str, list[Item]] = OrderedDict((a, []) for a in order)
    for item in merge(items):
        sections[item.category].append(item)
    for section in sections.values():
        section.sort(key=lambda i: i.name)
    return OrderedDict((a, s) for a, s in sections.items() if s)


def format_list(sections: OrderedDict[str, list[Item]]) -> str:
    lines = []
    for aisle, items in sections.items():
        lines.append(aisle.replace("_", " & ").upper())
        for i in items:
            qty = f"{i.quantity:g}" + (f" {i.unit}" if i.unit else "")
            lines.append(f"  [ ] {qty} {i.name}")
    return "\n".join(lines)
