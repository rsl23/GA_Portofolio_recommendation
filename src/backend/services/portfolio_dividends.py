"""Cash-dividend entitlements from the local ex-date history.

The source has no payment dates. Amounts are therefore accrued receivables,
never spendable cash, trades, or adjustments to a stock's cost basis.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd


DIVIDEND_FILE = Path(__file__).resolve().parents[3] / "data" / "dividend_events_20_tahun.xlsx"
REQUIRED_COLUMNS = ("Ticker", "Ex_Dividend_Date", "Dividend_Per_Saham")


class DividendDataUnavailableError(ValueError):
    """The dividend source cannot support a reliable entitlement calculation."""


@dataclass(frozen=True)
class DividendEvent:
    ticker: str
    ex_date: date
    per_share: Decimal


@dataclass(frozen=True)
class DividendHistory:
    events: tuple[DividendEvent, ...]

    @property
    def first_event_date(self) -> date | None:
        return self.events[0].ex_date if self.events else None

    @property
    def last_event_date(self) -> date | None:
        return self.events[-1].ex_date if self.events else None


def parse_dividend_history(frame: pd.DataFrame) -> DividendHistory:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DividendDataUnavailableError(f"Kolom dividen tidak lengkap: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise DividendDataUnavailableError("File riwayat dividen kosong; kelengkapan data tidak dapat dibuktikan.")
    dates = pd.to_datetime(frame["Ex_Dividend_Date"], errors="coerce")
    unique: dict[tuple[str, date], DividendEvent] = {}
    for row_number, ((ticker, _, raw_amount), ex_date) in enumerate(
        zip(frame.loc[:, list(REQUIRED_COLUMNS)].itertuples(index=False, name=None), dates), start=2
    ):
        code = str(ticker).strip().upper().removesuffix(".JK") if pd.notna(ticker) else ""
        try:
            amount = Decimal(str(raw_amount))
        except InvalidOperation as exc:
            raise DividendDataUnavailableError(f"Nilai dividen baris {row_number} tidak valid.") from exc
        if not code or pd.isna(ex_date) or not amount.is_finite() or amount < 0:
            raise DividendDataUnavailableError(f"Ticker, tanggal, atau nilai dividen baris {row_number} tidak valid.")
        event = DividendEvent(code, ex_date.date(), amount)
        key = (code, event.ex_date)
        if key in unique and unique[key].per_share != amount:
            raise DividendDataUnavailableError(f"Dividen {code} pada {event.ex_date} memiliki nilai yang bertentangan.")
        # Exact duplicate source rows must not pay the same dividend twice.
        unique[key] = event
    return DividendHistory(tuple(sorted(unique.values(), key=lambda event: (event.ex_date, event.ticker))))


@lru_cache(maxsize=2)
def _read_dividend_history(path: str, modified_ns: int, size: int) -> DividendHistory:
    # File identity invalidates the immutable cache when the archive is updated.
    return parse_dividend_history(pd.read_excel(path, usecols=list(REQUIRED_COLUMNS)))


def load_dividend_history() -> DividendHistory:
    try:
        stat = DIVIDEND_FILE.stat()
        return _read_dividend_history(str(DIVIDEND_FILE), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, ImportError, BadZipFile) as exc:
        raise DividendDataUnavailableError(f"Riwayat dividen tidak dapat dibaca: {exc}") from exc


def _date(value) -> date:
    return value.date() if isinstance(value, datetime) else value


def dividend_entitlements(history: DividendHistory, snapshots: list[dict], end_date: date) -> list[dict]:
    """Use the last position snapshot strictly BEFORE each ex-date.

    Selling/reducing on ex-date retains the old lots' entitlement. Adding or
    buying on ex-date does not increase it. This also works across missing
    valuation sessions: entitlement uses the recorded trade date, not the day
    on which the performance chart next has a price.
    """
    result = []
    index = 0
    active = None
    for event in history.events:
        if event.ex_date > end_date:
            break
        while index < len(snapshots) and _date(snapshots[index]["date"]) < event.ex_date:
            active = snapshots[index]
            index += 1
        if active is None or event.ticker not in active["positions"] or event.per_share == 0:
            continue
        item = active["positions"][event.ticker]
        start, end = item.get("start_date"), item.get("end_date")
        # Item windows describe trade-time ownership. The old item ending ON
        # ex-date was held beforehand and remains entitled to that dividend.
        if ((start is not None and _date(start) >= event.ex_date) or
                (end is not None and _date(end) < event.ex_date)):
            continue
        lots = int(item["lots"])
        if lots <= 0:
            continue
        result.append({
            "ticker": event.ticker, "ex_date": event.ex_date,
            "entitlement_portfolio_id": active["id"], "eligible_lots": lots,
            "eligible_shares": lots * 100, "dividend_per_share": event.per_share,
            "amount": lots * 100 * event.per_share,
        })
    return result
