"""
Time engine — temporal features for the supply chain simulation.

Tracks simulation time and provides modulation factors for:
- Time of day (rush-hour traffic)
- Day of week (weekday/weekend patterns)
- Seasonality (monsoon, winter, etc.)
- Holidays / festivals

Outputs cyclical encodings (sin/cos) suitable for neural network input,
and multiplicative factors that the environment applies to travel times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


# Indian holidays / festivals (month, day) — approximate dates
INDIAN_HOLIDAYS = [
    (1, 26),   # Republic Day
    (3, 29),   # Holi (approx)
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (10, 24),  # Dussehra (approx)
    (11, 12),  # Diwali (approx)
    (12, 25),  # Christmas
]

# Monsoon months (June-September) — increased weather disruptions
MONSOON_MONTHS = {6, 7, 8, 9}


@dataclass
class TimeState:
    """Current time state for observation building."""
    hour: float             # 0-24
    day_of_week: int        # 0-6 (Mon-Sun)
    month: int              # 1-12
    day_of_month: int       # 1-31
    is_holiday: bool
    is_monsoon: bool
    is_festival_season: bool
    # Cyclical encodings
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    month_sin: float
    month_cos: float


class TimeEngine:
    """
    Tracks simulation time and computes temporal factors.

    Time advances in fractional hours when shipments travel.
    """

    def __init__(self, start_hour: float = 8.0, start_day: int = 0,
                 start_month: int = 1, start_day_of_month: int = 15):
        self.hour = start_hour
        self.day_of_week = start_day        # 0 = Monday
        self.month = start_month
        self.day_of_month = start_day_of_month
        self.total_hours_elapsed = 0.0

    def advance(self, hours: float):
        """Advance simulation clock by the given hours."""
        self.total_hours_elapsed += hours
        self.hour += hours

        # Roll over days
        while self.hour >= 24.0:
            self.hour -= 24.0
            self.day_of_week = (self.day_of_week + 1) % 7
            self.day_of_month += 1
            if self.day_of_month > 30:  # simplified month length
                self.day_of_month = 1
                self.month = (self.month % 12) + 1

    def randomize(self, rng) -> None:
        """Set a random start time for episode variety."""
        self.hour = rng.uniform(0, 24)
        self.day_of_week = rng.randint(0, 6)
        self.month = rng.randint(1, 12)
        self.day_of_month = rng.randint(1, 28)
        self.total_hours_elapsed = 0.0

    # ── Modulation factors ─────────────────────────────────────────────

    def traffic_factor(self) -> float:
        """
        Time-of-day traffic multiplier.
        Peak hours: 8-10 AM and 5-8 PM → 1.3-1.6x
        Night:      11 PM - 5 AM      → 0.7x
        """
        h = self.hour
        if 8 <= h < 10:
            return 1.4 + 0.2 * math.sin(math.pi * (h - 8) / 2)
        elif 17 <= h < 20:
            return 1.5 + 0.1 * math.sin(math.pi * (h - 17) / 3)
        elif 23 <= h or h < 5:
            return 0.7
        else:
            return 1.0

    def weekend_factor(self) -> float:
        """Weekday vs weekend effect (less commercial traffic on weekends)."""
        if self.day_of_week >= 5:  # Saturday, Sunday
            return 0.85
        return 1.0

    def seasonal_weather_prob_boost(self) -> float:
        """
        Extra probability of weather anomalies during monsoon.
        Returns additive boost (e.g., 0.15 means +15% chance).
        """
        if self.month in MONSOON_MONTHS:
            return 0.15
        return 0.0

    def is_holiday(self) -> bool:
        """Check if current date is near a known holiday."""
        for hm, hd in INDIAN_HOLIDAYS:
            if self.month == hm and abs(self.day_of_month - hd) <= 1:
                return True
        return False

    def is_festival_season(self) -> bool:
        """Festival season: Oct-Nov (Dussehra-Diwali)."""
        return self.month in (10, 11)

    def is_monsoon(self) -> bool:
        return self.month in MONSOON_MONTHS

    def holiday_demand_factor(self) -> float:
        """Increased demand/congestion near holidays/festivals."""
        factor = 1.0
        if self.is_holiday():
            factor *= 1.3
        if self.is_festival_season():
            factor *= 1.2
        return factor

    def fuel_price_factor(self) -> float:
        """Seasonal fuel price variation (simplified)."""
        # Slightly higher in summer, lower in winter
        return 1.0 + 0.05 * math.sin(2 * math.pi * (self.month - 4) / 12)

    # ── Observation helpers ────────────────────────────────────────────

    def _cyclical(self, value: float, period: float) -> Tuple[float, float]:
        angle = 2 * math.pi * value / period
        return math.sin(angle), math.cos(angle)

    def get_state(self) -> TimeState:
        """Build the full time state for observations."""
        h_sin, h_cos = self._cyclical(self.hour, 24.0)
        d_sin, d_cos = self._cyclical(self.day_of_week, 7.0)
        m_sin, m_cos = self._cyclical(self.month - 1, 12.0)

        return TimeState(
            hour=self.hour,
            day_of_week=self.day_of_week,
            month=self.month,
            day_of_month=self.day_of_month,
            is_holiday=self.is_holiday(),
            is_monsoon=self.is_monsoon(),
            is_festival_season=self.is_festival_season(),
            hour_sin=h_sin,
            hour_cos=h_cos,
            dow_sin=d_sin,
            dow_cos=d_cos,
            month_sin=m_sin,
            month_cos=m_cos,
        )

    def get_context_vector(self) -> list:
        """
        Flat vector for the global context part of the observation.
        [hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
         is_holiday, is_festival, is_monsoon, fuel_price_factor]
        """
        ts = self.get_state()
        return [
            ts.hour_sin, ts.hour_cos,
            ts.dow_sin, ts.dow_cos,
            ts.month_sin, ts.month_cos,
            float(ts.is_holiday),
            float(ts.is_festival_season),
            float(ts.is_monsoon),
            self.fuel_price_factor(),
        ]
