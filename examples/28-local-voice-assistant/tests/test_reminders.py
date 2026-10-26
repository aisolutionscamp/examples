from datetime import date, datetime, timedelta

import pytest

from voice_assistant.reminders import ReminderStore


def test_daily_reminder_fires_once_per_day_and_persists(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("take the blood pressure tablet", "08:00")
    assert store.due(datetime(2026, 3, 1, 7, 59)) == []
    assert [r.text for r in store.due(datetime(2026, 3, 1, 8, 0))] == ["take the blood pressure tablet"]
    assert store.due(datetime(2026, 3, 1, 9, 0)) == []  # already announced today
    reloaded = ReminderStore(tmp_path / "r.json")
    assert reloaded.due(datetime(2026, 3, 1, 9, 0)) == []
    assert len(reloaded.due(datetime(2026, 3, 2, 8, 5))) == 1  # next day again


def test_one_off_appointment_only_on_its_date(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("doctor appointment at 10:30", "09:30", kind="appointment", on_date="2026-03-04")
    assert store.due(datetime(2026, 3, 3, 12, 0)) == []
    assert len(store.due(datetime(2026, 3, 4, 9, 31))) == 1
    assert [r.kind for r in store.for_day(date(2026, 3, 4))] == ["appointment"]
    assert store.for_day(date(2026, 3, 5)) == []


def test_repeat_until_acknowledged(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("take the tablet", "08:00")
    t0 = datetime(2026, 3, 1, 8, 0)
    store.due(t0)
    assert store.repeat_unacknowledged(t0 + timedelta(minutes=10)) == []
    assert len(store.repeat_unacknowledged(t0 + timedelta(minutes=30))) == 1
    assert len(store.acknowledge_open(t0 + timedelta(minutes=31))) == 1
    assert store.repeat_unacknowledged(t0 + timedelta(minutes=60)) == []


@pytest.mark.parametrize("bad", ["25:00", "8 pm"])
def test_rejects_invalid_times(tmp_path, bad):
    with pytest.raises(ValueError):
        ReminderStore(tmp_path / "r.json").add("x", bad)
