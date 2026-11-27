"""Główny moduł integracji Home Assistant: RL+MPC Domowy Automatyzator.

Spina w jednej pętli `asyncio`:
1. Nasłuchiwanie `state_changed`, `automation_triggered`, `script_started` i `call_service`.
2. Semantyczną deduplikację (`DomainAdapter`) oraz śledzenie `context.parent_id` (`ContextTracker`).
3. Natychmiastowe uczenie online z działań ludzi i ewaluację horyzontu MPC + polityki RL.
4. Wysyłkę aktywnych powiadomień na Androida (`Tak / Nie / Zawsze / Czasami`) i obsługę odpowiedzi.
5. Tworzenie oraz wycofywanie natywnych automatyzacji HA wraz z trwałym zapisem w `.storage`.
"""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    EVENT_CALL_SERVICE,
    EVENT_HOMEASSISTANT_STOP,
    EVENT_STATE_CHANGED,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Context,
    Event,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from .attribution import ContextTracker
from .automation_builder import AutomationBuilder
from .const import (
    CONF_ALLOW_PHYSICAL_ACTIONS,
    CONF_EXCLUDED_ENTITIES,
    CONF_MAX_QUESTIONS_PER_DAY,
    CONF_MIN_COOLDOWN_MINUTES,
    CONF_MIN_OCCURRENCES,
    CONF_NOTIFY_SERVICE,
    CONF_UCB_ALPHA,
    DEFAULT_ALLOW_PHYSICAL_ACTIONS,
    DEFAULT_MAX_QUESTIONS_PER_DAY,
    DEFAULT_MIN_COOLDOWN_MINUTES,
    DEFAULT_MIN_OCCURRENCES,
    DEFAULT_NOTIFY_SERVICE,
    DEFAULT_UCB_ALPHA,
    DOMAIN,
    FEEDBACK_ALWAYS,
    FEEDBACK_NO,
    FEEDBACK_SOMETIMES,
    FEEDBACK_YES,
    MPC_STEP_SECONDS,
    PLATFORMS,
    SERVICE_EVALUATE_NOW,
    SERVICE_RESET_PATTERN,
    SERVICE_REVOKE_AUTOMATION,
)
from .domains import DOMAIN_SPECS, DomainAdapter, DomainRole
from .features import FeatureExtractor
from .model import OnlineRLMPCEngine, PatternKind, PatternRecord
from .notifier import CompanionNotifier, PendingQuestion
from .storage import ModelStorage

_LOGGER = logging.getLogger(__name__)

EVENT_AUTOMATION_TRIGGERED = "automation_triggered"
EVENT_SCRIPT_STARTED = "script_started"

SCHEMA_PATTERN_ID = vol.Schema({vol.Required("pattern_id"): cv.string})


class RLMPCCoordinator:
    """Koordynator czasu rzeczywistego łączący obserwację stanów z silnikiem RL+MPC."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.signal_updated = f"{DOMAIN}_{entry.entry_id}_updated"

        opts = {**entry.data, **entry.options}
        self.allow_physical_actions = bool(
            opts.get(CONF_ALLOW_PHYSICAL_ACTIONS, DEFAULT_ALLOW_PHYSICAL_ACTIONS)
        )
        self.excluded_entities: set[str] = set(opts.get(CONF_EXCLUDED_ENTITIES) or [])

        self.adapter = DomainAdapter()
        self.context_tracker = ContextTracker()
        self.feature_extractor = FeatureExtractor()
        self.automation_builder = AutomationBuilder(hass)

        self.engine = OnlineRLMPCEngine(
            min_occurrences=int(opts.get(CONF_MIN_OCCURRENCES, DEFAULT_MIN_OCCURRENCES)),
            ucb_alpha=float(opts.get(CONF_UCB_ALPHA, DEFAULT_UCB_ALPHA)),
        )
        self.storage = ModelStorage(hass, entry.entry_id, self.engine)

        cooldown_sec = float(
            opts.get(CONF_MIN_COOLDOWN_MINUTES, DEFAULT_MIN_COOLDOWN_MINUTES)
        ) * 60.0
        self.notifier = CompanionNotifier(
            hass,
            default_notify_service=str(opts.get(CONF_NOTIFY_SERVICE, DEFAULT_NOTIFY_SERVICE)),
            min_cooldown_seconds=cooldown_sec,
            max_per_day=int(opts.get(CONF_MAX_QUESTIONS_PER_DAY, DEFAULT_MAX_QUESTIONS_PER_DAY)),
            on_feedback=self.async_handle_user_feedback,
        )

        self._unsubs: list[CALLBACK_TYPE] = []

    async def async_start(self) -> None:
        """Wczytuje model z `.storage` i uruchamia wszystkie nasłuchiwacze zdarzeń."""
        await self.storage.async_load()
        self.notifier.start()

        # Kolejność subskrypcji gwarantuje rejestrację kontekstów automatyzacji i usług
        self._unsubs.append(
            self.hass.bus.async_listen(
                EVENT_AUTOMATION_TRIGGERED, self._on_automation_triggered
            )
        )
        self._unsubs.append(
            self.hass.bus.async_listen(
                EVENT_SCRIPT_STARTED, self._on_automation_triggered
            )
        )
        self._unsubs.append(
            self.hass.bus.async_listen(EVENT_CALL_SERVICE, self._on_service_call)
        )
        self._unsubs.append(
            self.hass.bus.async_listen(EVENT_STATE_CHANGED, self._on_state_changed)
        )
        self._unsubs.append(
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self._on_ha_stop)
        )
        # Lekki zegar horyzontu MPC (co 150s) pozwalający proaktywnie pytać przed rutyną czasową
        self._unsubs.append(
            async_track_time_interval(
                self.hass,
                self._async_mpc_horizon_tick,
                timedelta(seconds=MPC_STEP_SECONDS),
            )
        )

    async def async_stop(self) -> None:
        """Zatrzymuje nasłuchiwanie i zapisuje bieżący stan modelu na dysk."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        self.notifier.stop()
        await self.storage.async_save_now()

    @callback
    def _on_automation_triggered(self, event: Event[dict[str, Any]]) -> None:
        self.context_tracker.record_automation_triggered(event)
        self.feature_extractor.mark_automation_activity(dt_util.utcnow().timestamp())

    @callback
    def _on_service_call(self, event: Event[dict[str, Any]]) -> None:
        self.context_tracker.record_service_call(event)
        if event.data.get("domain") in ("automation", "script"):
            self.feature_extractor.mark_automation_activity(dt_util.utcnow().timestamp())

    async def _on_ha_stop(self, _event: Event[Any]) -> None:
        await self.storage.async_save_now()

    @callback
    def _on_state_changed(self, event: Event[dict[str, Any]]) -> None:
        """Szybki filtr synchroniczny przekazujący istotne zdarzenia do korutyny uczenia."""
        entity_id = event.data.get("entity_id")
        new_state: State | None = event.data.get("new_state")
        old_state: State | None = event.data.get("old_state")

        if not entity_id or new_state is None:
            return
        if entity_id in self.excluded_entities or entity_id.startswith(f"sensor.{DOMAIN}"):
            return

        # Kontekst encji automation/script rejestrujemy zawsze [Przykazanie #3]
        if new_state.domain in ("automation", "script"):
            self.context_tracker.record_automation_triggered(event)
            self.feature_extractor.mark_automation_activity(dt_util.utcnow().timestamp())
            return

        if not self.adapter.is_supported(new_state.domain):
            return

        # Semantyczna deduplikacja przeciw `force_update` (MQTT, KNX) [Przykazanie #4]
        if not self.adapter.is_meaningful_change(old_state, new_state):
            return

        self.hass.async_create_task(self._async_process_meaningful_state(new_state))

    async def _async_process_meaningful_state(self, new_state: State) -> None:
        """Przetwarza zdeduplikowaną zmianę stanu w trybie online [Przykazanie #3, #5]."""
        spec = DOMAIN_SPECS.get(new_state.domain)
        if spec is None:
            return

        is_actuator = spec.role == DomainRole.ACTUATOR
        sole_uid = self.feature_extractor.sole_present_user_id(self.hass)
        attribution = await self.context_tracker.async_classify_state_change(
            self.hass,
            new_state,
            is_actuator_or_button=is_actuator,
            allow_physical_actions=self.allow_physical_actions,
            sole_present_user_id=sole_uid,
        )

        now_utc = new_state.last_updated or dt_util.utcnow()
        snapshot = self.feature_extractor.capture_snapshot(self.hass, now_utc)
        local_date_str = dt_util.as_local(now_utc).strftime("%Y-%m-%d")

        if attribution.actor_kind.value == "automation":
            # [Przykazanie #3] Akcja automatyzacji to wyłącznie kontekst tła — nigdy cel uczenia!
            self.feature_extractor.mark_automation_activity(snapshot.utc_timestamp)
            return

        trigger_key = self.adapter.make_trigger_key(new_state)
        just_updated_patterns: list[PatternRecord] = []

        # 1. Jeżeli to intencjonalna akcja człowieka na aktuatorze -> uczymy model nawyków!
        if attribution.is_human and is_actuator:
            action_spec = self.adapter.extract_action_spec(new_state)
            if action_spec is not None:
                just_updated_patterns = self.engine.observe_human_action(
                    action=action_spec,
                    snapshot=snapshot,
                    user_id=attribution.user_id,
                    local_date_str=local_date_str,
                )

        # 2. Rejestrujemy również samo zdarzenie jako potencjalny wyzwalacz dla kolejnych kroków
        # (np. włączenie ekspresu do kawy przez człowieka LUB powrót device_trackera do domu)
        if trigger_key is not None:
            trig_desc = self.adapter.describe_trigger_pl(trigger_key, new_state.name)
            self.engine.observe_trigger_event(
                trigger_key=trigger_key,
                entity_id=new_state.entity_id,
                description_pl=trig_desc,
                snapshot=snapshot,
                user_id=attribution.user_id if attribution.is_human else sole_uid,
                local_date_str=local_date_str,
            )

        self.storage.schedule_save()
        async_dispatcher_send(self.hass, self.signal_updated)

        # 3. Ewaluacja MPC + RL: czy w tym momencie wysłać trafne pytanie na Androida?
        await self.async_evaluate_and_maybe_ask(
            snapshot=snapshot,
            active_user_id=attribution.user_id or sole_uid,
            current_trigger_key=trigger_key,
            just_executed_patterns=just_updated_patterns,
        )

    async def _async_mpc_horizon_tick(self, now_dt: Any) -> None:
        """Okresowe sprawdzenie horyzontu MPC dla zbliżających się rutyn porannych/wieczornych."""
        snapshot = self.feature_extractor.capture_snapshot(self.hass, now_dt)
        sole_uid = self.feature_extractor.sole_present_user_id(self.hass)
        await self.async_evaluate_and_maybe_ask(
            snapshot=snapshot,
            active_user_id=sole_uid,
            current_trigger_key=None,
            just_executed_patterns=[],
        )

    async def async_evaluate_and_maybe_ask(
        self,
        *,
        snapshot: Any,
        active_user_id: str | None,
        current_trigger_key: str | None,
        just_executed_patterns: list[PatternRecord],
    ) -> str | None:
        """Uruchamia planistę MPC i politykę RL; wysyła pytanie jeśli mieści się w budżecie."""
        if not self.notifier.can_send_now(snapshot.utc_timestamp):
            return None

        current_states = {
            p.action.entity_id: st.state
            for p in self.engine.patterns.values()
            if (st := self.hass.states.get(p.action.entity_id)) is not None
        }
        fatigue = self.notifier.current_fatigue_cost(snapshot.utc_timestamp)

        proposal = self.engine.plan_next_question(
            snapshot=snapshot,
            active_user_id=active_user_id,
            current_trigger_key=current_trigger_key,
            just_executed_patterns=just_executed_patterns,
            entity_current_states=current_states,
            domain_priority_fn=self.adapter.get_priority,
            fatigue_cost=fatigue,
        )
        if proposal is None:
            return None

        person_mappings = self.feature_extractor.get_person_mappings(self.hass)
        qid = await self.notifier.async_send_question(proposal, person_mappings)
        if qid is not None:
            self.engine.mark_question_sent(proposal.pattern, snapshot.utc_timestamp)
            self.storage.schedule_save()
            async_dispatcher_send(self.hass, self.signal_updated)
        return qid

    async def _async_execute_action_once(self, pattern: PatternRecord) -> None:
        """Jednorazowo wykonuje akcję po kliknięciu 'Tak' z oznaczeniem własnego Context [Przykazanie #2, #3]."""
        act = pattern.action
        st = self.hass.states.get(act.entity_id)
        if st is not None and act.desired_state and st.state == act.desired_state and not act.service_data:
            return

        own_ctx = Context()
        self.context_tracker.register_own_context(own_ctx)

        service_payload: dict[str, Any] = dict(act.service_data)
        await self.hass.services.async_call(
            act.domain,
            act.service,
            service_payload,
            target={"entity_id": act.entity_id},
            blocking=False,
            context=own_ctx,
        )

    async def async_handle_user_feedback(
        self, pending: PendingQuestion, feedback_type: str
    ) -> None:
        """Przetwarza odpowiedź z Androida (`Tak / Nie / Zawsze / Czasami`) [Przykazanie #9]."""
        now_ts = dt_util.utcnow().timestamp()
        pat, reward = self.engine.apply_user_feedback(
            pattern_id=pending.pattern_id,
            feedback=feedback_type,
            feature_vector=pending.feature_vector,
            snapshot=pending.snapshot,
            now_ts=now_ts,
        )
        if pat is None:
            return

        tag = f"rl_mpc_{pat.pattern_id}"

        if feedback_type == FEEDBACK_YES:
            await self._async_execute_action_once(pat)
            await self.notifier.async_send_confirmation(
                pending.notify_target,
                "Wykonano jednorazowo",
                f"Zrealizowano: {pat.action.description_pl} (wzmocniono wagę wzorca).",
                tag,
            )

        elif feedback_type == FEEDBACK_ALWAYS:
            await self._async_execute_action_once(pat)
            auto_id, _ = await self.automation_builder.async_create_or_update(
                pat, conditional=False
            )
            await self.notifier.async_send_confirmation(
                pending.notify_target,
                "Utworzono stałą automatyzację",
                f"Zapisano regułę {auto_id}: {pat.trigger_description_pl} -> {pat.action.description_pl}.",
                tag,
            )

        elif feedback_type == FEEDBACK_SOMETIMES:
            await self._async_execute_action_once(pat)
            preview_conds, preview_descs = pat.condition_learner.synthesize_ha_conditions(
                include_time_window=(pat.kind == PatternKind.SEQUENTIAL)
            )
            if preview_conds:
                auto_id, cond_descs = await self.automation_builder.async_create_or_update(
                    pat, conditional=True
                )
                cond_text = ", ".join(cond_descs) if cond_descs else "wyuczone warunki kontekstu"
                await self.notifier.async_send_confirmation(
                    pending.notify_target,
                    "Utworzono automatyzację warunkową",
                    f"Reguła {auto_id} aktywna tylko gdy: {cond_text}.",
                    tag,
                )
            else:
                # Gdy brak jeszcze wyraźnego czynnika różnicującego, nie zgadujemy na ślepo [Przykazanie #2]
                await self.notifier.async_send_confirmation(
                    pending.notify_target,
                    "Zapisano próbkę warunków (Czasami)",
                    "Wykonano akcję i zapamiętano kontekst (pora, obecność, pogoda, czujniki). "
                    "Reguła warunkowa powstanie po uchwyceniu warunku różnicującego.",
                    tag,
                )

        elif feedback_type == FEEDBACK_NO:
            await self.notifier.async_send_confirmation(
                pending.notify_target,
                "Odrzucono propozycję",
                f"Obniżono nagrodę ({reward:+.2f}) i wstrzymano pytania o ten wzorzec.",
                tag,
            )

        await self.storage.async_save_now()
        async_dispatcher_send(self.hass, self.signal_updated)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Inicjalizacja integracji z wpisu ConfigEntry."""
    hass.data.setdefault(DOMAIN, {})
    coordinator = RLMPCCoordinator(hass, entry)
    await coordinator.async_start()

    hass.data[DOMAIN][entry.entry_id] = coordinator
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _async_register_services(hass)
    return True


async def _async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Przeładowuje wpis po zmianie parametrów w OptionsFlow."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Bezpieczne wyładowanie integracji z zapisem modelu."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        coordinator: RLMPCCoordinator | None = hass.data[DOMAIN].pop(entry.entry_id, None)
        if coordinator is not None:
            await coordinator.async_stop()
        if not hass.data[DOMAIN]:
            for srv in (SERVICE_REVOKE_AUTOMATION, SERVICE_RESET_PATTERN, SERVICE_EVALUATE_NOW):
                if hass.services.has_service(DOMAIN, srv):
                    hass.services.async_remove(DOMAIN, srv)
    return unload_ok


@callback
def _async_register_services(hass: HomeAssistant) -> None:
    """Rejestruje usługi zarządzania i odwracalności decyzji [Przykazanie #10]."""
    if hass.services.has_service(DOMAIN, SERVICE_REVOKE_AUTOMATION):
        return

    async def _handle_revoke(call: ServiceCall) -> None:
        raw_pid = str(call.data["pattern_id"]).removeprefix("rl_mpc_")
        now_ts = dt_util.utcnow().timestamp()
        for coord in hass.data.get(DOMAIN, {}).values():
            await coord.automation_builder.async_revoke(raw_pid)
            coord.engine.revoke_pattern_automation(raw_pid, now_ts)
            await coord.storage.async_save_now()
            async_dispatcher_send(hass, coord.signal_updated)

    async def _handle_reset(call: ServiceCall) -> None:
        raw_pid = str(call.data["pattern_id"]).removeprefix("rl_mpc_")
        for coord in hass.data.get(DOMAIN, {}).values():
            await coord.automation_builder.async_revoke(raw_pid)
            coord.engine.reset_pattern(raw_pid)
            await coord.storage.async_save_now()
            async_dispatcher_send(hass, coord.signal_updated)

    async def _handle_evaluate_now(_call: ServiceCall) -> None:
        for coord in hass.data.get(DOMAIN, {}).values():
            snap = coord.feature_extractor.capture_snapshot(hass)
            uid = coord.feature_extractor.sole_present_user_id(hass)
            await coord.async_evaluate_and_maybe_ask(
                snapshot=snap,
                active_user_id=uid,
                current_trigger_key=None,
                just_executed_patterns=[],
            )

    hass.services.async_register(
        DOMAIN, SERVICE_REVOKE_AUTOMATION, _handle_revoke, schema=SCHEMA_PATTERN_ID
    )
    hass.services.async_register(
        DOMAIN, SERVICE_RESET_PATTERN, _handle_reset, schema=SCHEMA_PATTERN_ID
    )
    hass.services.async_register(
        DOMAIN, SERVICE_EVALUATE_NOW, _handle_evaluate_now, schema=vol.Schema({})
    )
