"""Encje diagnostyczne prezentujące stan uczenia RL+MPC w panelu Home Assistanta."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

if TYPE_CHECKING:
    from . import RLMPCCoordinator


@dataclass(frozen=True, kw_only=True)
class RLMPCSensorDescription(SensorEntityDescription):
    """Opis pojedynczego sensora diagnostycznego."""

    value_fn: Callable[[RLMPCCoordinator], int | float]
    attrs_fn: Callable[[RLMPCCoordinator], dict[str, Any]] | None = None


def _top_patterns_attrs(coord: RLMPCCoordinator) -> dict[str, Any]:
    patterns = sorted(
        coord.engine.patterns.values(),
        key=lambda p: (p.automated, p.household_occurrences),
        reverse=True,
    )[:10]
    return {
        "wzorce_top": [
            {
                "pattern_id": p.pattern_id,
                "typ": p.kind.value,
                "wyzwalacz": p.trigger_description_pl,
                "akcja": p.action.description_pl,
                "wystapienia": p.household_occurrences,
                "dni": p.distinct_days,
                "zautomatyzowany": p.automated,
                "automation_id": p.automation_id,
            }
            for p in patterns
        ]
    }


SENSOR_DESCRIPTIONS: tuple[RLMPCSensorDescription, ...] = (
    RLMPCSensorDescription(
        key="learned_patterns",
        name="Zaobserwowane wzorce",
        icon="mdi:brain",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda c: len(c.engine.patterns),
        attrs_fn=_top_patterns_attrs,
    ),
    RLMPCSensorDescription(
        key="active_automations",
        name="Utworzone automatyzacje",
        icon="mdi:robot-happy",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda c: sum(1 for p in c.engine.patterns.values() if p.automated),
    ),
    RLMPCSensorDescription(
        key="rl_cumulative_reward",
        name="Skumulowana nagroda RL",
        icon="mdi:chart-line",
        state_class=SensorStateClass.TOTAL,
        value_fn=lambda c: round(c.engine.policy.cumulative_reward, 2),
        attrs_fn=lambda c: {
            "liczba_aktualizacji_rl": c.engine.policy.total_updates,
            "wagi_theta": [round(w, 4) for w in c.engine.policy.theta()],
        },
    ),
    RLMPCSensorDescription(
        key="daily_question_budget",
        name="Pozostały budżet pytań dziś",
        icon="mdi:bell-check-outline",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda c: c.notifier.remaining_daily_budget(),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Inicjalizacja sensorów diagnostycznych dla wpisu ConfigEntry."""
    coordinator: RLMPCCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        RLMPCDiagnosticSensor(coordinator, entry.entry_id, desc)
        for desc in SENSOR_DESCRIPTIONS
    )


class RLMPCDiagnosticSensor(SensorEntity):
    """Sensor raportujący metryki silnika RL+MPC bez odpytywania (local_push)."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    entity_description: RLMPCSensorDescription

    def __init__(
        self,
        coordinator: RLMPCCoordinator,
        entry_id: str,
        description: RLMPCSensorDescription,
    ) -> None:
        self._coordinator = coordinator
        self.entity_description = description
        self._attr_unique_id = f"{entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="RL+MPC Domowy Automatyzator",
            manufacturer="Local Home ML",
            model="Hybrid LinUCB + Finite-Horizon MPC",
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Subskrypcja sygnału aktualizacji online."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                self._coordinator.signal_updated,
                self._handle_coordinator_update,
            )
        )

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> int | float:
        return self.entity_description.value_fn(self._coordinator)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        if self.entity_description.attrs_fn is not None:
            return self.entity_description.attrs_fn(self._coordinator)
        return None
