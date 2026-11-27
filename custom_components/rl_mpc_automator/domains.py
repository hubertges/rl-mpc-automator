"""Wspólna warstwa abstrakcji dla wszystkich 46 domen encji Home Assistanta.

Moduł odpowiada za:
1. Semantyczną deduplikację raportów (odporność na MQTT `force_update` i KNX) [Przykazanie #4].
2. Kwantyzację atrybutów ciągłych (jasność, pozycja rolet, temperatura), aby drobny szum
   nie rozbijał tego samego nawyku na dziesiątki osobnych akcji [Przykazanie #7].
3. Mapowanie stanów docelowych człowieka na wykonywalne wywołania usług HA [Przykazanie #7].
4. Oznaczanie domen wrażliwych bezpieczeństwa (zamki, alarmy, bramy garażowe) [Przykazanie #2].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from homeassistant.core import State

from .const import PRIMARY_EMPHASIS_DOMAINS, SUPPORTED_DOMAINS

UNAVAILABLE_STATES: frozenset[str] = frozenset({"unavailable", "unknown", ""})


class DomainRole(StrEnum):
    """Rola domeny w silniku RL+MPC."""

    ACTUATOR = "actuator"  # Może być zarówno wyzwalaczem, jak i celem automatyzacji
    CONTEXT_SENSOR = "context_sensor"  # Wyzwalacz i cecha kontekstu (nigdy cel akcji)
    SERVICE_ONLY = "service_only"  # Domena zadaniowa / powiadomieniowa


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Kanonizowany opis akcji człowieka gotowy do odtworzenia jako usługa HA."""

    action_key: str  # Stabilny klucz wzorca, np. "cover.rolety_salon::open_cover"
    entity_id: str
    domain: str
    service: str
    service_data: dict[str, Any] = field(default_factory=dict)
    desired_state: str = ""
    safety_critical: bool = False
    description_pl: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializacja do zapisu w .storage."""
        return {
            "action_key": self.action_key,
            "entity_id": self.entity_id,
            "domain": self.domain,
            "service": self.service,
            "service_data": dict(self.service_data),
            "desired_state": self.desired_state,
            "safety_critical": self.safety_critical,
            "description_pl": self.description_pl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpec:
        """Deserializacja z .storage."""
        return cls(
            action_key=str(data["action_key"]),
            entity_id=str(data["entity_id"]),
            domain=str(data["domain"]),
            service=str(data["service"]),
            service_data=dict(data.get("service_data") or {}),
            desired_state=str(data.get("desired_state", "")),
            safety_critical=bool(data.get("safety_critical", False)),
            description_pl=str(data.get("description_pl", "")),
        )


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """Specyfikacja pojedynczej domeny Home Assistanta."""

    domain: str
    role: DomainRole
    stateless_event: bool = False
    safety_sensitive: bool = False
    significant_attrs: tuple[str, ...] = ()
    priority_weight: float = 0.6


def _quantize_number(val: Any, step: float) -> float | None:
    """Kwantyzuje wartość liczbową do najbliższego kroku `step`."""
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return round(round(num / step) * step, 2)


# Rejestr wszystkich 46 domen referencyjnych [Przykazanie #7]
DOMAIN_SPECS: dict[str, DomainSpec] = {
    "ai_task": DomainSpec("ai_task", DomainRole.SERVICE_ONLY, priority_weight=0.45),
    "air_quality": DomainSpec(
        "air_quality",
        DomainRole.CONTEXT_SENSOR,
        significant_attrs=("particulate_matter_2_5", "carbon_dioxide", "air_quality_index"),
        priority_weight=0.55,
    ),
    "alarm_control_panel": DomainSpec(
        "alarm_control_panel",
        DomainRole.ACTUATOR,
        safety_sensitive=True,
        priority_weight=0.90,
    ),
    "assist_satellite": DomainSpec(
        "assist_satellite", DomainRole.SERVICE_ONLY, priority_weight=0.45
    ),
    "binary_sensor": DomainSpec(
        "binary_sensor",
        DomainRole.CONTEXT_SENSOR,
        significant_attrs=("device_class",),
        priority_weight=0.95,
    ),
    "button": DomainSpec(
        "button", DomainRole.ACTUATOR, stateless_event=True, priority_weight=0.75
    ),
    "calendar": DomainSpec(
        "calendar",
        DomainRole.CONTEXT_SENSOR,
        significant_attrs=("message", "all_day"),
        priority_weight=0.85,
    ),
    "camera": DomainSpec(
        "camera",
        DomainRole.ACTUATOR,
        significant_attrs=("motion_detection",),
        priority_weight=0.60,
    ),
    "climate": DomainSpec(
        "climate",
        DomainRole.ACTUATOR,
        significant_attrs=("temperature", "target_temp_low", "target_temp_high", "preset_mode"),
        priority_weight=0.95,
    ),
    "conversation": DomainSpec("conversation", DomainRole.SERVICE_ONLY, priority_weight=0.45),
    "cover": DomainSpec(
        "cover",
        DomainRole.ACTUATOR,
        significant_attrs=("current_position", "current_tilt_position", "device_class"),
        priority_weight=1.00,
    ),
    "date": DomainSpec("date", DomainRole.ACTUATOR, priority_weight=0.50),
    "datetime": DomainSpec("datetime", DomainRole.ACTUATOR, priority_weight=0.50),
    "device_tracker": DomainSpec(
        "device_tracker", DomainRole.CONTEXT_SENSOR, priority_weight=0.95
    ),
    "event": DomainSpec(
        "event",
        DomainRole.CONTEXT_SENSOR,
        stateless_event=True,
        significant_attrs=("event_type",),
        priority_weight=0.80,
    ),
    "fan": DomainSpec(
        "fan",
        DomainRole.ACTUATOR,
        significant_attrs=("percentage", "preset_mode", "oscillating"),
        priority_weight=0.80,
    ),
    "geo_location": DomainSpec("geo_location", DomainRole.CONTEXT_SENSOR, priority_weight=0.50),
    "humidifier": DomainSpec(
        "humidifier",
        DomainRole.ACTUATOR,
        significant_attrs=("humidity", "mode"),
        priority_weight=0.75,
    ),
    "image": DomainSpec("image", DomainRole.CONTEXT_SENSOR, priority_weight=0.40),
    "image_processing": DomainSpec(
        "image_processing",
        DomainRole.CONTEXT_SENSOR,
        significant_attrs=("total_faces",),
        priority_weight=0.55,
    ),
    "infrared": DomainSpec(
        "infrared", DomainRole.ACTUATOR, stateless_event=True, priority_weight=0.65
    ),
    "lawn_mower": DomainSpec("lawn_mower", DomainRole.ACTUATOR, priority_weight=0.65),
    "light": DomainSpec(
        "light",
        DomainRole.ACTUATOR,
        significant_attrs=("brightness", "color_temp_kelvin", "effect"),
        priority_weight=0.95,
    ),
    "lock": DomainSpec(
        "lock", DomainRole.ACTUATOR, safety_sensitive=True, priority_weight=1.00
    ),
    "media_player": DomainSpec(
        "media_player",
        DomainRole.ACTUATOR,
        significant_attrs=("source", "volume_level", "is_volume_muted"),
        priority_weight=0.90,
    ),
    "notify": DomainSpec(
        "notify", DomainRole.SERVICE_ONLY, stateless_event=True, priority_weight=0.85
    ),
    "number": DomainSpec("number", DomainRole.ACTUATOR, priority_weight=0.70),
    "radio_frequency": DomainSpec(
        "radio_frequency", DomainRole.ACTUATOR, stateless_event=True, priority_weight=0.65
    ),
    "remote": DomainSpec(
        "remote",
        DomainRole.ACTUATOR,
        significant_attrs=("current_activity",),
        priority_weight=0.75,
    ),
    "scene": DomainSpec(
        "scene", DomainRole.ACTUATOR, stateless_event=True, priority_weight=0.85
    ),
    "select": DomainSpec("select", DomainRole.ACTUATOR, priority_weight=0.75),
    "sensor": DomainSpec(
        "sensor",
        DomainRole.CONTEXT_SENSOR,
        significant_attrs=("device_class", "unit_of_measurement"),
        priority_weight=0.90,
    ),
    "siren": DomainSpec(
        "siren", DomainRole.ACTUATOR, safety_sensitive=True, priority_weight=0.75
    ),
    "stt": DomainSpec(
        "stt", DomainRole.CONTEXT_SENSOR, stateless_event=True, priority_weight=0.45
    ),
    "switch": DomainSpec(
        "switch",
        DomainRole.ACTUATOR,
        significant_attrs=("device_class",),
        priority_weight=1.00,
    ),
    "tag": DomainSpec(
        "tag", DomainRole.CONTEXT_SENSOR, stateless_event=True, priority_weight=0.85
    ),
    "text": DomainSpec("text", DomainRole.ACTUATOR, priority_weight=0.55),
    "time": DomainSpec("time", DomainRole.ACTUATOR, priority_weight=0.55),
    "todo": DomainSpec("todo", DomainRole.ACTUATOR, priority_weight=0.60),
    "tts": DomainSpec(
        "tts", DomainRole.SERVICE_ONLY, stateless_event=True, priority_weight=0.50
    ),
    "update": DomainSpec("update", DomainRole.CONTEXT_SENSOR, priority_weight=0.40),
    "vacuum": DomainSpec("vacuum", DomainRole.ACTUATOR, priority_weight=0.80),
    "valve": DomainSpec(
        "valve",
        DomainRole.ACTUATOR,
        safety_sensitive=True,
        significant_attrs=("current_position",),
        priority_weight=0.85,
    ),
    "wake_word": DomainSpec(
        "wake_word", DomainRole.CONTEXT_SENSOR, stateless_event=True, priority_weight=0.50
    ),
    "water_heater": DomainSpec(
        "water_heater",
        DomainRole.ACTUATOR,
        significant_attrs=("temperature", "operation_mode"),
        priority_weight=0.80,
    ),
    "weather": DomainSpec(
        "weather",
        DomainRole.CONTEXT_SENSOR,
        significant_attrs=("temperature", "humidity", "wind_speed"),
        priority_weight=0.90,
    ),
}

# Gwarancja zgodności ze specyfikacją referencyjną w momencie importu modułu
if frozenset(DOMAIN_SPECS.keys()) != frozenset(SUPPORTED_DOMAINS):
    missing = set(SUPPORTED_DOMAINS) - set(DOMAIN_SPECS)
    extra = set(DOMAIN_SPECS) - set(SUPPORTED_DOMAINS)
    raise RuntimeError(f"Niezgodność rejestru domen! Brakujące={missing}, nadmiarowe={extra}")


class DomainAdapter:
    """Adapter tłumaczący surowe obiekty State HA na semantyczne odciski, wyzwalacze i akcje."""

    def __init__(self) -> None:
        # Ostatnie semantyczne odciski stanów do filtrowania wymuszonych aktualizacji
        self._last_fingerprints: dict[str, tuple[Any, ...]] = {}

    def is_supported(self, domain: str) -> bool:
        """Sprawdza, czy domena należy do 46 obsługiwanych domen."""
        return domain in DOMAIN_SPECS

    def get_spec(self, domain: str) -> DomainSpec | None:
        """Zwraca specyfikację domeny."""
        return DOMAIN_SPECS.get(domain)

    def get_priority(self, domain: str) -> float:
        """Zwraca wagę priorytetu domeny z premią dla głównych domen domowych."""
        spec = DOMAIN_SPECS.get(domain)
        if spec is None:
            return 0.3
        bonus = 0.05 if domain in PRIMARY_EMPHASIS_DOMAINS else 0.0
        return min(1.0, spec.priority_weight + bonus)

    def semantic_fingerprint(self, state: State) -> tuple[Any, ...]:
        """Tworzy kanoniczny odcisk stanu odporny na szum i `force_update` [Przykazanie #4].

        Ignoruje pola czasowe (`last_reported`, `last_updated`), RSSI czy ikony,
        a ciągłe atrybuty kwantyzuje do stabilnych przedziałów.
        """
        domain = state.domain
        spec = DOMAIN_SPECS.get(domain)
        attrs = state.attributes
        raw_state = state.state

        if spec is not None and spec.stateless_event:
            # Domeny bezstanowe (button, event, tag, scene) niosą informację w samym wyzwoleniu
            event_type = attrs.get("event_type")
            return (domain, raw_state, event_type)

        if domain == "light":
            # Kwantyzacja jasności (0..255 -> krok 64) i temperatury barwowej (krok 500K)
            b_q = _quantize_number(attrs.get("brightness"), 64.0) if raw_state == "on" else None
            ct_q = (
                _quantize_number(attrs.get("color_temp_kelvin"), 500.0)
                if raw_state == "on"
                else None
            )
            return (raw_state, b_q, ct_q, attrs.get("effect"))

        if domain == "cover":
            # Pozycja rolet/bramy w krokach 25%; stany przejściowe opening/closing mapujemy docelowo
            norm_state = (
                "open"
                if raw_state == "opening"
                else ("closed" if raw_state == "closing" else raw_state)
            )
            pos_q = _quantize_number(attrs.get("current_position"), 25.0)
            return (norm_state, pos_q)

        if domain == "climate":
            temp_q = _quantize_number(attrs.get("temperature"), 0.5)
            return (raw_state, temp_q, attrs.get("preset_mode"))

        if domain == "media_player":
            vol_q = _quantize_number(attrs.get("volume_level"), 0.1)
            return (raw_state, attrs.get("source"), vol_q, attrs.get("is_volume_muted"))

        if domain == "sensor":
            # Dla czujników ciągłych kwantyzujemy wg klasy urządzenia, by nie mielić szumu 0.01°C
            dev_cls = attrs.get("device_class")
            if dev_cls == "illuminance":
                return (dev_cls, _quantize_number(raw_state, 50.0))
            if dev_cls in ("temperature", "humidity"):
                return (dev_cls, _quantize_number(raw_state, 1.0))
            return (dev_cls, _quantize_number(raw_state, 5.0) or raw_state)

        if domain == "weather":
            return (raw_state, _quantize_number(attrs.get("temperature"), 2.0))

        if spec is not None and spec.significant_attrs:
            sig_vals = tuple(attrs.get(k) for k in spec.significant_attrs)
            return (raw_state, *sig_vals)

        return (raw_state,)

    def is_meaningful_change(self, old_state: State | None, new_state: State) -> bool:
        """Weryfikuje, czy zmiana stanu jest semantycznie istotna (odrzuca `force_update`)."""
        if new_state.state in UNAVAILABLE_STATES:
            return False

        spec = DOMAIN_SPECS.get(new_state.domain)
        if spec is None:
            return False

        new_fp = self.semantic_fingerprint(new_state)
        prev_cached = self._last_fingerprints.get(new_state.entity_id)

        if old_state is not None and old_state.state not in UNAVAILABLE_STATES:
            old_fp = self.semantic_fingerprint(old_state)
            if old_fp == new_fp:
                # Typowy przypadek MQTT `force_update: true` lub cyklicznego raportu KNX
                return False

        if prev_cached is not None and prev_cached == new_fp and not spec.stateless_event:
            return False

        self._last_fingerprints[new_state.entity_id] = new_fp
        return True

    def is_safety_critical(self, state: State) -> bool:
        """Rozpoznaje encje o podwyższonym ryzyku fizycznym (zamki, bramy, alarmy, zawory)."""
        spec = DOMAIN_SPECS.get(state.domain)
        if spec is not None and spec.safety_sensitive:
            return True
        if state.domain == "cover":
            dev_class = str(state.attributes.get("device_class", "")).lower()
            if dev_class in {"garage", "gate", "door", "window"}:
                return True
        return False

    def make_trigger_key(self, state: State) -> str | None:
        """Buduje dyskretny klucz wyzwalacza dla sekwencji (np. 'switch.ekspres::on')."""
        domain = state.domain
        raw = state.state
        if raw in UNAVAILABLE_STATES:
            return None

        spec = DOMAIN_SPECS.get(domain)
        if spec is None:
            return None

        if domain == "sensor":
            # Zwykłe sensory liczbowe traktujemy jako cechy ciągłe w ContextSnapshot,
            # chyba że to dyskretny sensor tekstowy (np. stan ekspresu/pralki)
            try:
                float(raw)
                return None
            except ValueError:
                return f"{state.entity_id}::{raw}"

        if domain == "weather":
            # Pogoda jest cechą środowiskową tła, nie punktowym wyzwalaczem sekwencji
            return None

        if spec.stateless_event:
            ev_type = state.attributes.get("event_type", "pressed")
            return f"{state.entity_id}::event:{ev_type}"

        if domain == "cover":
            norm = "open" if raw in ("open", "opening") else ("closed" if raw in ("closed", "closing") else raw)
            return f"{state.entity_id}::{norm}"

        return f"{state.entity_id}::{raw}"

    def describe_trigger_pl(self, trigger_key: str, friendly_name: str | None = None) -> str:
        """Generuje zwięzły, polski opis wyzwalacza do treści powiadomienia na Androida."""
        entity_id, _, state_token = trigger_key.partition("::")
        name = friendly_name or entity_id
        domain = entity_id.split(".", 1)[0]

        if domain in ("switch", "light", "fan", "humidifier", "input_boolean"):
            verb = "włączeniu" if state_token == "on" else "wyłączeniu"
            return f"po {verb} «{name}»"
        if domain == "binary_sensor":
            verb = "aktywacji" if state_token == "on" else "zakończeniu aktywności"
            return f"po {verb} czujnika «{name}»"
        if domain == "device_tracker":
            place = "powrocie do domu" if state_token == "home" else f"opuszczeniu domu ({state_token})"
            return f"przy {place} («{name}»)"
        if domain == "cover":
            verb = "otwarciu" if state_token == "open" else "zamknięciu"
            return f"po {verb} «{name}»"
        if domain == "lock":
            verb = "odryglowaniu" if state_token == "unlocked" else "zaryglowaniu"
            return f"po {verb} «{name}»"
        if domain == "calendar":
            return f"gdy rozpoczyna się wydarzenie w «{name}»"
        if state_token.startswith("event:"):
            return f"po wyzwoleniu «{name}»"
        return f"gdy «{name}» przejdzie w stan «{state_token}»"

    def extract_action_spec(self, state: State) -> ActionSpec | None:
        """Przekształca ręczną zmianę stanu domownika w odtwarzalny ActionSpec [Przykazanie #7]."""
        domain = state.domain
        spec = DOMAIN_SPECS.get(domain)
        if spec is None or spec.role != DomainRole.ACTUATOR:
            return None

        raw = state.state
        if raw in UNAVAILABLE_STATES:
            return None

        entity_id = state.entity_id
        attrs = state.attributes
        name = state.name or entity_id
        safety = self.is_safety_critical(state)

        service: str | None = None
        service_data: dict[str, Any] = {}
        desired_state = raw
        action_suffix = raw
        desc = f"ustawić «{name}» ({raw})"

        if domain == "light":
            if raw == "on":
                service = "turn_on"
                b_val = attrs.get("brightness")
                if isinstance(b_val, (int, float)) and 0 < b_val < 245:
                    pct = int(round((float(b_val) / 255.0) * 100.0 / 25.0) * 25)
                    pct = max(10, min(100, pct))
                    service_data["brightness_pct"] = pct
                    action_suffix = f"on_b{pct}"
                    desc = f"włączyć światło «{name}» ({pct}%)"
                else:
                    desc = f"włączyć światło «{name}»"
            elif raw == "off":
                service = "turn_off"
                desc = f"wyłączyć światło «{name}»"

        elif domain == "switch":
            if raw in ("on", "off"):
                service = f"turn_{raw}"
                verb = "włączyć" if raw == "on" else "wyłączyć"
                desc = f"{verb} «{name}»"

        elif domain == "cover":
            pos = attrs.get("current_position")
            if raw in ("open", "opening"):
                if isinstance(pos, (int, float)) and 15 <= pos <= 85 and raw == "open":
                    q_pos = int(round(float(pos) / 25.0) * 25)
                    service = "set_cover_position"
                    service_data["position"] = q_pos
                    action_suffix = f"pos_{q_pos}"
                    desc = f"ustawić «{name}» na {q_pos}%"
                else:
                    service = "open_cover"
                    desired_state = "open"
                    action_suffix = "open"
                    desc = f"otworzyć «{name}»"
            elif raw in ("closed", "closing"):
                service = "close_cover"
                desired_state = "closed"
                action_suffix = "closed"
                desc = f"zamknąć «{name}»"

        elif domain == "lock":
            if raw in ("locked", "locking"):
                service = "lock"
                desired_state = "locked"
                action_suffix = "lock"
                desc = f"zaryglować zamek «{name}»"
            elif raw in ("unlocked", "unlocking"):
                service = "unlock"
                desired_state = "unlocked"
                action_suffix = "unlock"
                desc = f"odryglować zamek «{name}»"

        elif domain == "climate":
            if raw == "off":
                service = "turn_off"
                desc = f"wyłączyć klimatyzację/ogrzewanie «{name}»"
            else:
                temp = _quantize_number(attrs.get("temperature"), 0.5)
                if temp is not None:
                    service = "set_temperature"
                    service_data["hvac_mode"] = raw
                    service_data["temperature"] = temp
                    action_suffix = f"{raw}_{temp}"
                    desc = f"ustawić «{name}» na {temp}°C ({raw})"
                else:
                    service = "set_hvac_mode"
                    service_data["hvac_mode"] = raw
                    desc = f"ustawić tryb «{name}» na {raw}"

        elif domain == "media_player":
            if raw == "off":
                service = "turn_off"
                desc = f"wyłączyć «{name}»"
            elif raw == "playing":
                service = "media_play"
                src = attrs.get("source")
                desc = f"odtworzyć multimedia na «{name}»" + (f" ({src})" if src else "")
            elif raw == "paused":
                service = "media_pause"
                desc = f"wstrzymać odtwarzanie na «{name}»"
            elif raw in ("on", "idle", "standby"):
                service = "turn_on"
                desc = f"włączyć «{name}»"

        elif domain in ("fan", "humidifier", "siren", "remote", "camera"):
            if raw in ("on", "off", "streaming", "idle", "recording"):
                is_on = raw not in ("off",)
                service = "turn_on" if is_on else "turn_off"
                action_suffix = "on" if is_on else "off"
                desc = f"{'włączyć' if is_on else 'wyłączyć'} «{name}»"

        elif domain == "valve":
            if raw in ("open", "opening"):
                service = "open_valve"
                desired_state = "open"
                action_suffix = "open"
                desc = f"otworzyć zawór «{name}»"
            elif raw in ("closed", "closing"):
                service = "close_valve"
                desired_state = "closed"
                action_suffix = "closed"
                desc = f"zamknąć zawór «{name}»"

        elif domain == "vacuum":
            if raw == "cleaning":
                service = "start"
                desc = f"uruchomić odkurzacz «{name}»"
            elif raw in ("docked", "returning"):
                service = "return_to_base"
                desired_state = "docked"
                desc = f"odesłać odkurzacz «{name}» do bazy"

        elif domain == "lawn_mower":
            if raw == "mowing":
                service = "start_mowing"
                desc = f"uruchomić kosiarkę «{name}»"
            elif raw in ("docked", "returning"):
                service = "dock"
                desired_state = "docked"
                desc = f"zadokować kosiarkę «{name}»"

        elif domain == "alarm_control_panel":
            alarm_map = {
                "armed_away": ("alarm_arm_away", "uzbroić alarm (poza domem)"),
                "armed_home": ("alarm_arm_home", "uzbroić alarm (w domu)"),
                "armed_night": ("alarm_arm_night", "uzbroić alarm (noc)"),
                "disarmed": ("alarm_disarm", "rozbroić alarm"),
            }
            if raw in alarm_map:
                service, verb_phrase = alarm_map[raw]
                desc = f"{verb_phrase} «{name}»"

        elif domain == "water_heater":
            if raw == "off":
                service = "turn_off"
                desc = f"wyłączyć bojler «{name}»"
            else:
                service = "set_operation_mode"
                service_data["operation_mode"] = raw
                desc = f"ustawić bojler «{name}» w tryb {raw}"

        elif domain == "scene":
            service = "turn_on"
            action_suffix = "activate"
            desired_state = ""
            desc = f"aktywować scenę «{name}»"

        elif domain == "button":
            service = "press"
            action_suffix = "press"
            desired_state = ""
            desc = f"nacisnąć przycisk «{name}»"

        elif domain == "select":
            service = "select_option"
            service_data["option"] = raw
            desc = f"ustawić «{name}» na «{raw}»"

        elif domain == "number":
            val = _quantize_number(raw, 1.0)
            if val is not None:
                service = "set_value"
                service_data["value"] = val
                action_suffix = f"val_{val}"
                desc = f"ustawić «{name}» na {val}"

        elif domain == "text":
            service = "set_value"
            service_data["value"] = raw
            desc = f"wpisać wartość w «{name}»"

        elif domain in ("date", "datetime", "time"):
            service = "set_value"
            field_key = "datetime" if domain == "datetime" else domain
            service_data[field_key] = raw
            desc = f"ustawić czas/datę «{name}»"

        elif domain in ("infrared", "radio_frequency"):
            service = "send_command"
            action_suffix = "send_command"
            desired_state = ""
            desc = f"wysłać sygnał przez «{name}»"

        elif domain == "todo":
            # Zmiana liczby zadań sama w sobie nie tworzy jednoznacznej akcji bez treści zadania
            return None

        if service is None:
            return None

        return ActionSpec(
            action_key=f"{entity_id}::{action_suffix}",
            entity_id=entity_id,
            domain=domain,
            service=service,
            service_data=service_data,
            desired_state=desired_state,
            safety_critical=safety,
            description_pl=desc,
        )
