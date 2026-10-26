"""Keyword-based grocery categories. Extend KEYWORDS for your own items or languages."""
import re

CATEGORIES = ("produce", "bakery", "meat_fish", "dairy", "pantry", "beverages", "frozen", "household", "other")

KEYWORDS: dict[str, tuple[str, ...]] = {
    "produce": ("apple", "banana", "berry", "berries", "lemon", "lime", "orange", "grape", "pear", "avocado",
                "tomato", "potato", "onion", "garlic", "carrot", "cucumber", "lettuce", "spinach", "pepper",
                "broccoli", "zucchini", "mushroom", "herb", "basil", "parsley", "ginger", "salad"),
    "bakery": ("bread", "baguette", "roll", "bagel", "croissant", "tortilla", "bun", "toast"),
    "meat_fish": ("chicken", "beef", "pork", "turkey", "ham", "bacon", "sausage", "mince", "salmon", "tuna",
                  "fish", "shrimp", "prawn"),
    "dairy": ("milk", "cheese", "yogurt", "yoghurt", "butter", "cream", "egg", "eggs", "quark", "mozzarella",
              "parmesan"),
    "pantry": ("rice", "pasta", "spaghetti", "noodle", "flour", "sugar", "salt", "oil", "vinegar", "oats",
               "cereal", "beans", "lentils", "canned", "sauce", "honey", "jam", "coffee", "tea", "spice",
               "chocolate", "nuts"),
    "beverages": ("water", "juice", "soda", "beer", "wine", "lemonade"),
    "frozen": ("frozen", "ice cream", "pizza"),
    "household": ("soap", "detergent", "toilet paper", "paper towel", "sponge", "trash bag", "dish", "shampoo",
                  "toothpaste"),
}

_PAIRS = [(kw, cat) for cat, kws in KEYWORDS.items() for kw in kws]
# "frozen" is a modifier that overrides the base item's category
_PRIORITY = {"frozen": "frozen"}


def categorize(name: str) -> str:
    """Pick the keyword that ends last (English head noun: "orange juice" is a juice), longest on ties."""
    text = name.lower()
    for word, category in _PRIORITY.items():
        if re.search(rf"\b{word}\b", text):
            return category
    best: tuple[int, int, str] | None = None
    for keyword, category in _PAIRS:
        for match in re.finditer(rf"\b{re.escape(keyword)}(e?s)?\b", text):
            candidate = (match.end(), len(keyword), category)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best[2] if best else "other"
