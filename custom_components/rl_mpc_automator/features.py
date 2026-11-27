"""Ekstrakcja wektorów cech środowiskowych, czasowych i obecności domowników [Przykazanie #6, #9]."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import statistics
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import COND_FEATURE_DIM

RECENT_ARRIVAL_SECONDS = 900.0  # 15 minut od powrotu do domu
RECENT_MOTION_SECONDS = 180.0  # 3 minuty aktywności czujnika binarnego
DARK_LUX_THRESHOLD = 80.0

SUNNY_CONDITIONS: frozenset[str] = frozenset({"sunny", "clear-night", "partlycloudy"})
PRECIP_CONDITIONS: frozenset[str] = frozenset(
    {"rainy", "pouring", "snowy", "snowy-rainy", "lightning", "lightning-rainy", "hail"}
)


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Pełna migawka stanu domu w chwili zaobserwowania zdarzenia."""

    utc_timestamp: float
    local_hour: int
    local_minute: int
    weekday: int  # 0 = poniedziałek .. 6 = niedziela
    is_weekend: bool
    slot_30m: int  # 0..47 (półgodzinny kubełek doby dla MPC)
    present_user_ids: tuple[str, ...]
    present_person_entities: tuple[str, ...]
    home_ratio: float
    recent_arrival: bool
    illuminance_lux: float | None
    illuminance_entity_id: str | None
    is_dark: bool
    weather_condition: str | None
    weather_entity_id: str | None
    outdoor_temp: float | None
    calendar_busy: bool
    calendar_entity_id: str | None
    active_binary_sensors: tuple[str, ...]
    automation_recently_active: bool

    @property
    def minute_of_day(self) -> int:
        """Minuta doby lokalnej (0..1439)."""
        return self.local_hour * 60 + self.local_minute

    @property
    def anyone_home(self) -> bool:
        """Czy ktokolwiek jest w domu wg trackerów / encji person."""
        return self.home_ratio > 0.0 or len(self.present_user_ids) > 0

    def to_condition_vector(self) -> list[float]:
        """Koduje migawkę do znormalizowanego wektora R^10 dla klasyfikatora warunków 'Czasami'."""
        angle_day = 2.0 * math.pi * (self.minute_of_day / 1440.0)
        angle_week = 2.0 * math.pi * (self.weekday / 7.0)
        lux_norm = (
            min(1.0, math.log1p(max(0.0, self.illuminance_lux)) / math.log1p(2000.0))
            if self.illuminance_lux is not None
            else 0.5
        )
        is_precip = 1.0 if self.weather_condition in PRECIP_CONDITIONS else 0.0
        vec = [
            1.0,  # 0: bias
            math.sin(angle_day),  # 1: pora dnia (sin)
            math.cos(angle_day),  # 2: pora dnia (cos)
            math.sin(angle_week),  # 3: dzień tygodnia (sin)
            1.0 if self.is_weekend else 0.0,  # 4: weekend
            self.home_ratio,  # 5: odsetek domowników w domu
            1.0 if self.recent_arrival else 0.0,  # 6: niedawny powrót do domu
            1.0 if self.is_dark else (1.0 - lux_norm),  # 7: ciemność / niskie oświetlenie
            is_precip,  # 8: opady / zła pogoda
            1.0 if self.calendar_busy else 0.0,  # 9: aktywne wydarzenie w kalendarzu
        ]
        assert len(vec) == COND_FEATURE_DIM
        return vec


@dataclass(slots=True)
class PersonMapping:
    """Powiązanie encji `person.*` z kontem `user_id` oraz usługą `notify.mobile_app_*`."""

    person_entity_id: str
    user_id: str | None
    friendly_name: str
    device_trackers: list[str] = field(default_factory=list)


class FeatureExtractor:
    """Bezblokadowy ekstraktor cech stanu z `hass.states`."""

    def __init__(self) -> None:
        self._last_automation_ts: float = 0.0

    def mark_automation_activity(self, now_ts: float) -> None:
        """Odnotowuje czas ostatniego zadziałania dowolnej automatyzacji (jako kontekst tła)."""
        self._last_automation_ts = now_ts

    def get_person_mappings(self, hass: HomeAssistant) -> list[PersonMapping]:
        """Odczytuje powiązania `person.* -> user_id + device_trackers` bezpośrednio ze stanów HA."""
        mappings: list[PersonMapping] = []
        for st in hass.states.async_all("person"):
            uid = st.attributes.get("user_id")
            trackers = list(st.attributes.get("device_trackers") or [])
            mappings.append(
                PersonMapping(
                    person_entity_id=st.entity_id,
                    user_id=str(uid) if uid else None,
                    friendly_name=st.name or st.entity_id,
                    device_trackers=[str(t) for t in trackers],
                )
            )
        return mappings

    def sole_present_user_id(self, hass: HomeAssistant) -> str | None:
        """Zwraca `user_id`, jeśli w domu znajduje się dokładnie jeden zidentyfikowany domownik."""
        present_uids: list[str] = []
        for st in hass.states.async_all("person"):
            if st.state == "home":
                uid = st.attributes.get("user_id")
                if uid:
                    present_uids.append(str(uid))
        if len(present_uids) == 1:
            return present_uids[0]
        return None

    def capture_snapshot(
        self, hass: HomeAssistant, now_utc: datetime | None = None
    ) -> ContextSnapshot:
        """Buduje kompletną migawkę stanu domu dla silnika RL/MPC."""
        utc_dt = now_utc or dt_util.utcnow()
        now_ts = utc_dt.timestamp()
        local_dt = dt_util.as_local(utc_dt)

        # 1. Obecność domowników (`person` oraz `device_tracker`)
        person_states = hass.states.async_all("person")
        present_uids: list[str] = []
        present_persons: list[str] = []
        recent_arrival = False

        if person_states:
            total_tracked = len(person_states)
            home_count = 0
            for pst in person_states:
                if pst.state == "home":
                    home_count += 1
                    present_persons.append(pst.entity_id)
                    uid = pst.attributes.get("user_id")
                    if uid:
                        present_uids.append(str(uid))
                    if (now_ts - pst.last_changed.timestamp()) <= RECENT_ARRIVAL_SECONDS:
                        recent_arrival = True
            home_ratio = home_count / max(1, total_tracked)
        else:
            trackers = hass.states.async_all("device_tracker")
            total_tracked = len(trackers)
            home_count = 0
            for tst in trackers:
                if tst.state == "home":
                    home_count += 1
                    if (now_ts - tst.last_changed.timestamp()) <= RECENT_ARRIVAL_SECONDS:
                        recent_arrival = True
            home_ratio = (home_count / total_tracked) if total_tracked > 0 else 1.0

        # 2. Natężenie oświetlenia z czujników `sensor` (device_class == illuminance)
        lux_values: list[tuple[float, str]] = []
        for sst in hass.states.async_all("sensor"):
            if sst.attributes.get("device_class") == "illuminance":
                try:
                    lux_values.append((float(sst.state), sst.entity_id))
                except (TypeError, ValueError):
                    continue

        illuminance_lux: float | None = None
        illuminance_entity: str | None = None
        if lux_values:
            lux_values.sort(key=lambda pair: pair[0])
            mid = len(lux_values) // 2
            illuminance_lux = float(statistics.median([v for v, _ in lux_values]))
            illuminance_entity = lux_values[mid][1]

        # Ciemność oceniamy z czujnika lux lub z encji `sun.sun` jako fallback
        sun_state = hass.states.get("sun.sun")
        sun_below = sun_state is not None and sun_state.state == "below_horizon"
        is_dark = (
            (illuminance_lux < DARK_LUX_THRESHOLD)
            if illuminance_lux is not None
            else sun_below
        )

        # 3. Pogoda (`weather.*`)
        weather_cond: str | None = None
        weather_entity: str | None = None
        outdoor_temp: float | None = None
        for wst in hass.states.async_all("weather"):
            if wst.state not in ("unavailable", "unknown"):
                weather_cond = wst.state
                weather_entity = wst.entity_id
                raw_t = wst.attributes.get("temperature")
                if isinstance(raw_t, (int, float)):
                    outdoor_temp = float(raw_t)
                break

        # 4. Kalendarz (`calendar.*`)
        calendar_busy = False
        calendar_entity: str | None = None
        for cst in hass.states.async_all("calendar"):
            if cst.state == "on":
                calendar_busy = True
                calendar_entity = cst.entity_id
                break

        # 5. Aktywne czujniki binarne ruchu / obecności / otwarcia
        active_bs: list[str] = []
        for bst in hass.states.async_all("binary_sensor"):
            dev_cls = bst.attributes.get("device_class")
            if dev_cls in ("motion", "occupancy", "presence", "door", "garage_door", "opening"):
                if bst.state == "on" or (now_ts - bst.last_changed.timestamp()) <= RECENT_MOTION_SECONDS:
                    if bst.state == "on":
                        active_bs.append(bst.entity_id)

        hour = local_dt.hour
        minute = local_dt.minute
        weekday = local_dt.weekday()
        slot_30m = (hour * 60 + minute) // 30

        return ContextSnapshot(
            utc_timestamp=now_ts,
            local_hour=hour,
            local_minute=minute,
            weekday=weekday,
            is_weekend=weekday >= 5,
            slot_30m=slot_30m,
            present_user_ids=tuple(present_uids),
            present_person_entities=tuple(present_persons),
            home_ratio=round(home_ratio, 3),
            recent_arrival=recent_arrival,
            illuminance_lux=illuminance_lux,
            illuminance_entity_id=illuminance_entity,
            is_dark=is_dark,
            weather_condition=weather_cond,
            weather_entity_id=weather_entity,
            outdoor_temp=outdoor_temp,
            calendar_busy=calendar_busy,
            calendar_entity_id=calendar_entity,
            active_binary_sensors=tuple(active_bs[:8]),
            automation_recently_active=(now_ts - self._last_automation_ts) <= 60.0,
        )
