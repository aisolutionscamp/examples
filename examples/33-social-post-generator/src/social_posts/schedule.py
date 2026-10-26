"""Spread generated posts over the coming days and export them as CSV for a scheduling tool."""

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

# Example posting slots per platform (local time). Adjust to when your audience is online.
DEFAULT_SLOTS = {
    "instagram": time(18, 0),
    "facebook": time(12, 0),
    "linkedin": time(8, 30),
    "x": time(9, 0),
}


@dataclass
class ScheduledPost:
    platform: str
    publish_at: datetime
    text: str


def next_weekdays(start: date, count: int, skip_weekends: bool = True) -> list[date]:
    days, current = [], start
    while len(days) < count:
        if not skip_weekends or current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def build_schedule(posts: dict[str, list[str]], start: date, slots: dict[str, time] | None = None,
                   skip_weekends: bool = True) -> list[ScheduledPost]:
    """One post per platform per day, starting at `start`."""
    slots = slots or DEFAULT_SLOTS
    schedule = []
    for platform, texts in posts.items():
        for day, text in zip(next_weekdays(start, len(texts), skip_weekends), texts, strict=True):
            schedule.append(ScheduledPost(platform, datetime.combine(day, slots[platform]), text))
    return sorted(schedule, key=lambda p: (p.publish_at, p.platform))


def write_csv(schedule: list[ScheduledPost], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["platform", "publish_at", "characters", "text"])
        for p in schedule:
            writer.writerow([p.platform, p.publish_at.isoformat(timespec="minutes"), len(p.text), p.text])
