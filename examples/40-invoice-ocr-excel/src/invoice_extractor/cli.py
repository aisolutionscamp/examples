"""Extract data from all invoices in a folder and write one Excel file.

    python -m invoice_extractor.cli samples/ -o invoices.xlsx
    python -m invoice_extractor.cli scan.png --us-dates -o out.csv
"""
import argparse
from pathlib import Path

from .export import write_csv, write_excel
from .ocr import SUPPORTED, load_text
from .parse import InvoiceData, parse_invoice


def collect(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    return files


def run(paths: list[Path], output: Path, day_first: bool = True, lang: str = "eng") -> list[InvoiceData]:
    results = []
    for f in collect(paths):
        try:
            results.append(parse_invoice(load_text(f, lang), source=f.name, day_first=day_first))
        except (RuntimeError, ValueError, OSError) as exc:  # keep going, list the file as a problem
            results.append(InvoiceData(source=f.name, problems=[str(exc)]))
    (write_csv if output.suffix.lower() == ".csv" else write_excel)(results, output)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="files or folders")
    p.add_argument("-o", "--output", type=Path, default=Path("invoices.xlsx"))
    p.add_argument("--us-dates", action="store_true", help="read 03/10/2026 as March 10 (default: 3 October)")
    p.add_argument("--lang", default="eng", help="Tesseract language, e.g. eng, deu, eng+deu")
    args = p.parse_args()
    results = run(args.paths, args.output, day_first=not args.us_dates, lang=args.lang)
    for r in results:
        status = "check: " + "; ".join(r.problems) if r.problems else "ok"
        print(f"{r.source}: {r.invoice_number or '-'} {r.total or '-'} {r.currency or ''} ({status})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
