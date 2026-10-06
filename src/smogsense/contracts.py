"""smogsense.contracts - Shared value objects."""
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass(frozen=True)
class Issuance:
    t0: datetime

    def block_window(self, h: int) -> tuple[datetime, datetime]:
        """Returns the [start_utc, end_utc) for the 24 h block ending at T0 + h."""
        end_utc = self.t0 + timedelta(hours=h)
        start_utc = end_utc - timedelta(hours=24)
        return start_utc, end_utc

@dataclass(frozen=True)
class Horizon:
    lead_h: int

@dataclass(frozen=True)
class QuantileGrid:
    levels: tuple[float, ...] = (
        0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
        0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.96, 0.97, 0.98, 0.99
    )

    @property
    def public(self) -> tuple[float, float, float]:
        return (0.10, 0.50, 0.90)

@dataclass(frozen=True)
class DomainId:
    name: str
