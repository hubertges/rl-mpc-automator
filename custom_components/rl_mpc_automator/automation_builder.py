"""Synteza, atomowy zapis i odwracalne wycofywanie natywnych automatyzacji HA [Przykazanie #2, #10]."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from homeassistant.core import HomeAssistant

from .const import (
    AUTOMATION_ID_PREFIX,
    EVENT_AUTOMATION_CREATED,
    EVENT_AUTOMATION_REVOKED,
)
from .model import PatternKind, PatternRecord

_LOGGER = logging.getLogger(__name__)


class AutomationBuilder:
    """Tworzy i wycofuje standardowe, czytelne dla człowieka automatyzacje w `automations.yaml`."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    def _automations_path(self) -> Path:
        return Path(self._hass.config.path("automations.yaml"))

    def build_automation_config(
        self,
        pattern: PatternRecord,
        *,
        conditional: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        """Buduje słownik konfiguracji automatyzacji zgodny ze schematem Home Assistanta."""
        auto_id = f"{AUTOMATION_ID_PREFIX}{pattern.pattern_id}"
        mode_label = "warunkowa" if conditional else "stała"
        alias = f"[RL+MPC] {pattern.action.description_pl.capitalize()} ({mode_label})"

        # 1. Wyzwalacz (trigger)
        if pattern.kind == PatternKind.SEQUENTIAL:
            trig_entity, _, trig_state = pattern.trigger_key.partition("::")
            if trig_state.startswith("event:"):
                ev_type = trig_state.split(":", 1)[1]
                trigger: dict[str, Any] = {
                    "platform": "state",
                    "entity_id": trig_entity,
                    "attribute": "event_type",
                    "to": ev_type,
                }
            else:
                trigger = {
                    "platform": "state",
                    "entity_id": trig_entity,
                    "to": trig_state,
                }
        else:
            mean_min, _ = pattern.condition_learner.circular_time_stats()
            at_str = f"{mean_min // 60:02d}:{mean_min % 60:02d}:00"
            trigger = {
                "platform": "time",
                "at": at_str,
            }

        # 2. Warunki (condition)
        conditions: list[dict[str, Any]] = []
        cond_descriptions: list[str] = []

        # Strażnik idempotentności: nie wywołuj usługi, jeśli encja już ma pożądany stan główny
        if pattern.action.desired_state and not pattern.action.service_data:
            conditions.append(
                {
                    "condition": "not",
                    "conditions": [
                        {
                            "condition": "state",
                            "entity_id": pattern.action.entity_id,
                            "state": pattern.action.desired_state,
                        }
                    ],
                }
            )

        if conditional:
            # Dla rutyn czasowych wyzwalacz sam ustala godzinę, więc nie dublujemy okna godzinowego
            learned_conds, cond_descriptions = (
                pattern.condition_learner.synthesize_ha_conditions(
                    include_time_window=(pattern.kind == PatternKind.SEQUENTIAL)
                )
            )
            conditions.extend(learned_conds)
        elif pattern.action.safety_critical:
            # Dla domen wrażliwych (zamek, brama garażowa) dołączamy warunek obecności, jeśli znany
            for person_ent, cnt in pattern.condition_learner.person_counts.items():
                if cnt >= 2:
                    conditions.append(
                        {
                            "condition": "state",
                            "entity_id": person_ent,
                            "state": "home",
                        }
                    )
                    cond_descriptions.append(f"{person_ent} w domu (ochrona bezpieczeństwa)")
                    break

        # 3. Akcja (action)
        action_step: dict[str, Any] = {
            "service": f"{pattern.action.domain}.{pattern.action.service}",
            "target": {"entity_id": pattern.action.entity_id},
        }
        if pattern.action.service_data:
            action_step["data"] = dict(pattern.action.service_data)

        actions: list[dict[str, Any]] = []
        # Jeżeli w sekwencji człowiek konsekwentnie czekał np. >15 s, zachowujemy naturalne opóźnienie
        if pattern.kind == PatternKind.SEQUENTIAL and 15.0 <= pattern.delay_mean_sec <= 120.0:
            delay_s = int(round(pattern.delay_mean_sec / 5.0) * 5)
            actions.append({"delay": {"seconds": delay_s}})

        actions.append(action_step)

        description_parts = [
            f"Utworzona za zgodą domownika przez RL+MPC Domowy Automatyzator (wzorzec {pattern.pattern_id}).",
            f"Wyzwalacz: {pattern.trigger_description_pl}.",
        ]
        if cond_descriptions:
            description_parts.append(f"Warunki: {', '.join(cond_descriptions)}.")
        description_parts.append(
            f"Możesz ją edytować w UI lub wycofać usługą rl_mpc_automator.revoke_automation (pattern_id: {pattern.pattern_id})."
        )

        config_dict: dict[str, Any] = {
            "id": auto_id,
            "alias": alias,
            "description": " ".join(description_parts),
            "mode": "single",
            "trigger": [trigger],
            "condition": conditions,
            "action": actions,
        }
        return (config_dict, cond_descriptions)

    def _upsert_yaml_sync(self, automation_cfg: dict[str, Any]) -> None:
        """Atomowy zapis do `automations.yaml` wykonywany w puli wątków executora."""
        path = self._automations_path()
        existing: list[dict[str, Any]] = []
        if path.exists():
            raw_text = path.read_text(encoding="utf-8").strip()
            if raw_text:
                loaded = yaml.safe_load(raw_text)
                if isinstance(loaded, list):
                    existing = loaded

        target_id = automation_cfg["id"]
        replaced = False
        for idx, item in enumerate(existing):
            if isinstance(item, dict) and item.get("id") == target_id:
                existing[idx] = automation_cfg
                replaced = True
                break
        if not replaced:
            existing.append(automation_cfg)

        self._atomic_write_yaml(path, existing)

    def _remove_yaml_sync(self, auto_id: str) -> bool:
        """Atomowe usunięcie reguły z `automations.yaml` po `id`."""
        path = self._automations_path()
        if not path.exists():
            return False
        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return False
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, list):
            return False

        filtered = [
            item for item in loaded if not (isinstance(item, dict) and item.get("id") == auto_id)
        ]
        if len(filtered) == len(loaded):
            return False

        self._atomic_write_yaml(path, filtered)
        return True

    @staticmethod
    def _atomic_write_yaml(path: Path, data: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".automations_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    data,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    async def async_create_or_update(
        self,
        pattern: PatternRecord,
        *,
        conditional: bool,
    ) -> tuple[str, list[str]]:
        """Zapisuje zaakceptowaną automatyzację i przeładowuje domenę `automation` w HA."""
        auto_cfg, cond_descs = self.build_automation_config(pattern, conditional=conditional)
        auto_id = str(auto_cfg["id"])

        await self._hass.async_add_executor_job(self._upsert_yaml_sync, auto_cfg)
        pattern.automation_id = auto_id
        pattern.automated = True

        if self._hass.services.has_service("automation", "reload"):
            await self._hass.services.async_call("automation", "reload", blocking=True)

        self._hass.bus.async_fire(
            EVENT_AUTOMATION_CREATED,
            {
                "pattern_id": pattern.pattern_id,
                "automation_id": auto_id,
                "conditional": conditional,
                "conditions": cond_descs,
                "entity_id": pattern.action.entity_id,
            },
        )
        _LOGGER.info(
            "Utworzono natywną automatyzację %s dla wzorca %s (warunkowa=%s)",
            auto_id,
            pattern.pattern_id,
            conditional,
        )
        return (auto_id, cond_descs)

    async def async_revoke(self, pattern_id: str) -> bool:
        """Wycofuje utworzoną automatyzację z HA i przywraca stan sprzed zgody [Przykazanie #10]."""
        auto_id = (
            pattern_id
            if pattern_id.startswith(AUTOMATION_ID_PREFIX)
            else f"{AUTOMATION_ID_PREFIX}{pattern_id}"
        )
        clean_pid = pattern_id.removeprefix(AUTOMATION_ID_PREFIX)

        removed = await self._hass.async_add_executor_job(self._remove_yaml_sync, auto_id)
        if removed and self._hass.services.has_service("automation", "reload"):
            await self._hass.services.async_call("automation", "reload", blocking=True)

        if removed:
            self._hass.bus.async_fire(
                EVENT_AUTOMATION_REVOKED,
                {"pattern_id": clean_pid, "automation_id": auto_id},
            )
            _LOGGER.info("Wycofano automatyzację %s (wzorzec %s)", auto_id, clean_pid)
        return removed
