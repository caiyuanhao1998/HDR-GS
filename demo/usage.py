"""Persistent successful render counts for the hosted studio."""

from datetime import datetime, timedelta
import logging
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo


PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
LOG = logging.getLogger(__name__)


class UsageCounter:
    def __init__(self, storage):
        self.storage = Path(storage)
        self.lock = threading.Lock()
        self.cached = None
        self.cache_time = 0

    def record(self, job, *, now=None):
        day = (now or datetime.now(PACIFIC_TIME)).astimezone(PACIFIC_TIME).date()
        with self.lock:
            try:
                directory = self.storage / day.isoformat()
                directory.mkdir(exist_ok=True)
                # The job ID makes retries idempotent without a shared mutable total.
                (directory / f"{job}.render").write_bytes(b"1\n")
                self.cached = None
            except OSError as error:
                LOG.warning("Could not record successful usage (%s)", type(error).__name__)

    def summary(self, *, now=None):
        today = (now or datetime.now(PACIFIC_TIME)).astimezone(PACIFIC_TIME).date()
        today_key = today.isoformat()
        first_key = (today - timedelta(days=29)).isoformat()
        with self.lock:
            if (self.cached and self.cached[0] == today_key
                    and time.monotonic() - self.cache_time < 30):
                return self.cached[1]
            counts = {}
            try:
                for directory in self.storage.iterdir():
                    day = directory.name
                    if first_key <= day <= today_key and directory.is_dir():
                        counts[day] = sum(event.name.endswith(".render") for event in directory.iterdir())
            except OSError as error:
                LOG.warning("Could not read usage counts (%s)", type(error).__name__)
                return None
            summary = dict(today=counts.get(today_key, 0), last_30_days=sum(counts.values()),
                           timezone=PACIFIC_TIME.key)
            self.cached = (today_key, summary)
            self.cache_time = time.monotonic()
            return summary

    def badges(self):
        stats = self.summary()
        items = [("Last month", "last_30_days"), ("Today", "today")]
        chips = []
        for label, key in items:
            number = f"{stats[key]:,}" if stats is not None else "—"
            chips.append(f'<span class="usage-chip"><span>{label}</span><b>{number}</b></span>')
        title = ("Successful LDR and HDR render pairs. Last month includes today "
                 "and the preceding 29 dates in America/Los_Angeles.")
        return f'<div class="usage-counts" aria-label="Successful renders" title="{title}">' + "".join(chips) + "</div>"
