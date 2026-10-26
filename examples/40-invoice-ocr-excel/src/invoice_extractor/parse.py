"""Find invoice number, date, vendor and amounts in OCR text with regular expressions."""
import datetime as dt
import re
from dataclasses import asdict, dataclass, field

# "1,234.56" (US/UK) or "1.234,56" / "4,00" (EU); the last 2 digits are the cents
AMOUNT = r"(?:[$€£]\s?)?(\d{1,3}(?:[., ]\d{3})*[.,]\d{2}|\d+[.,]\d{2})(?:\s?(?:EUR|USD|GBP|CHF|€))?"
I, IM = re.IGNORECASE, re.IGNORECASE | re.MULTILINE
DATE = r"([0-9]{1,4}[./-][0-9]{1,2}[./-][0-9]{2,4})"
PATTERNS = {
    "invoice_number": re.compile(
        r"\b(?:invoice|inv|receipt|bill)[ \t]*(?:no\.?|number|nr\.?|#)?[ \t]*[:#]?[ \t]*"
        r"((?=[A-Z\-/]*\d)[A-Z0-9][A-Z0-9\-/]{2,})", I),
    "date": re.compile(r"(?<!due )\b(?:invoice\s+)?date\b\s*:?\s*" + DATE, I),
    "due_date": re.compile(r"\bdue(?:\s+date)?\b\s*:?\s*" + DATE, I),
    "subtotal": re.compile(r"^\s*(?:sub\s?total|net(?:\s+amount)?)\b[^\n\d$€£]*" + AMOUNT, IM),
    "tax": re.compile(r"^\s*(?:tax|vat|sales\s+tax)\b[^\n$€£]*?\s" + AMOUNT, IM),
    "total": re.compile(r"^\s*(?:total(?:\s+due)?|amount\s+due|balance\s+due|grand\s+total)\b[^\n\d$€£]*"
                        + AMOUNT, IM),
}
CURRENCIES = {"$": "USD", "€": "EUR", "£": "GBP", "EUR": "EUR", "USD": "USD", "GBP": "GBP", "CHF": "CHF"}


@dataclass
class InvoiceData:
    source: str = ""
    vendor: str | None = None
    invoice_number: str | None = None
    date: dt.date | None = None
    due_date: dt.date | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_amount(text: str) -> float:
    digits = re.sub(r"[^\d.,]", "", text)
    cents_sep = digits[-3]  # the separator before the last two digits
    whole = re.sub(r"[.,]", "", digits[:-3])
    return float(f"{whole}.{digits[-2:]}") if cents_sep in ".," else float(whole)


def parse_date(text: str, day_first: bool = True) -> dt.date | None:
    text = text.replace(".", "/").replace("-", "/")
    formats = ["%Y/%m/%d"] + (["%d/%m/%Y", "%d/%m/%y"] if day_first else ["%m/%d/%Y", "%m/%d/%y"])
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_invoice(text: str, source: str = "", day_first: bool = True) -> InvoiceData:
    data = InvoiceData(source=source)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data.vendor = lines[0] if lines else None  # most invoices start with the sender's name

    if m := PATTERNS["invoice_number"].search(text):
        data.invoice_number = m.group(1)
    for key in ("date", "due_date"):
        if m := PATTERNS[key].search(text):
            setattr(data, key, parse_date(m.group(1), day_first))
    for key in ("subtotal", "tax", "total"):
        matches = PATTERNS[key].findall(text)
        if matches:
            setattr(data, key, parse_amount(matches[-1]))

    found = re.findall(r"[$€£]|\b(?:EUR|USD|GBP|CHF)\b", text)
    data.currency = CURRENCIES[found[0]] if found else None

    for name in ("invoice_number", "date", "total"):
        if getattr(data, name) is None:
            data.problems.append(f"{name} not found")
    if None not in (data.subtotal, data.tax, data.total) and abs(data.subtotal + data.tax - data.total) > 0.02:
        data.problems.append("subtotal + tax does not match total")
    if data.date and data.due_date and data.due_date < data.date:
        data.problems.append("due date before invoice date")
    return data
