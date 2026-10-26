"""python -m shopping_list [--layout layouts/example-store.json] [--llm] < note.txt"""
import argparse
import sys

from .organizer import format_list, load_layout, organize
from .parser import parse_text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shopping_list")
    parser.add_argument("text", nargs="?", help="shopping note; read from stdin if omitted")
    parser.add_argument("--layout", help="store layout JSON with the aisle order")
    parser.add_argument("--llm", action="store_true", help="parse free text with an LLM (needs .env)")
    args = parser.parse_args(argv)

    text = args.text or sys.stdin.read()
    if args.llm:
        from .llm_parser import parse_with_llm

        items = parse_with_llm(text)
    else:
        items = parse_text(text)
    aisles = load_layout(args.layout) if args.layout else None
    print(format_list(organize(items, aisles)))


if __name__ == "__main__":
    main()
