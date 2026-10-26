"""Daily reminders (medication, appointments) stored in a JSON file.

The scheduler is a plain function of "now", so it can be tested with fixed times and
driven by any loop (a while-loop with sleep, cron, a systemd timer).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path


@dataclass
class Reminder:
    text: str
    at: str  # "HH:MM", local time
    kind: str = "medication"  # medication | appointment | other
    on_date: str | None = None  # "YYYY-MM-DD" for a one-off reminder, None = every day
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    last_fired: str | None = None  # ISO date of the last day it was announced
    acknowledged: str | None = None  # ISO date the user confirmed it

    def due_at(self, day: date) -> datetime | None:
        if self.on_date and self.on_date != day.isoformat():
            return None
        return datetime.combine(day, time.fromisoformat(self.at))


class ReminderStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.items: list[Reminder] = []
        if self.path.exists():
            self.items = [Reminder(**r) for r in json.loads(self.path.read_text())]

    def save(self) -> None:
        self.path.write_text(json.dumps([asdict(r) for r in self.items], indent=2))

    def add(self, text: str, at: str, kind: str = "medication", on_date: str | None = None) -> Reminder:
        time.fromisoformat(at)  # raises ValueError for "25:00" or "8 pm"
        if on_date:
            date.fromisoformat(on_date)
        r = Reminder(text=text, at=at, kind=kind, on_date=on_date)
        self.items.append(r)
        self.save()
        return r

    def due(self, now: datetime) -> list[Reminder]:
        """Reminders whose time has passed today and that were not announced today yet.

        A reminder that is still unacknowledged is repeated every `repeat` minutes by
        `repeat_unacknowledged`; this method only handles the first announcement.
        """
        today = now.date().isoformat()
        hits = []
        for r in self.items:
            t = r.due_at(now.date())
            if t and t <= now and r.last_fired != today:
                r.last_fired = today
                hits.append(r)
        if hits:
            self.save()
        return hits

    def repeat_unacknowledged(self, now: datetime, after: timedelta = timedelta(minutes=30)) -> list[Reminder]:
        today = now.date().isoformat()
        return [r for r in self.items
                if r.last_fired == today and r.acknowledged != today
                and (t := r.due_at(now.date())) and now - t >= after]

    def acknowledge_open(self, now: datetime) -> list[Reminder]:
        """Mark every reminder announced today as done (the user said "done" / "I took them")."""
        today = now.date().isoformat()
        done = [r for r in self.items if r.last_fired == today and r.acknowledged != today]
        for r in done:
            r.acknowledged = today
        if done:
            self.save()
        return done

    def for_day(self, day: date) -> list[Reminder]:
        return sorted((r for r in self.items if r.due_at(day)), key=lambda r: r.at)
