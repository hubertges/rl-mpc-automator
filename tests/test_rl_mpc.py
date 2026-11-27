"""Testy jednostkowe i integracyjne weryfikujące zgodność ze wszystkimi 10 Przykazaniami."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
import tempfile
import types
from typing import Any
import unittest


def _install_ha_stubs_if_needed() -> None:
    """Lekkie zaślepki klas rdzenia HA pozwalające testować cały pakiet bez pełnego HA Core."""
    if "voluptuous" not in sys.modules:
        vol_mod = types.ModuleType("voluptuous")
        vol_mod.Schema = lambda s, **kw: s  # type: ignore[attr-defined]
        vol_mod.Required = lambda k, **kw: k  # type: ignore[attr-defined]
        vol_mod.Optional = lambda k, **kw: k  # type: ignore[attr-defined]
        sys.modules["voluptuous"] = vol_mod

    if "homeassistant" in sys.modules:
        return

    ha_mod = types.ModuleType("homeassistant")
    core_mod = types.ModuleType("homeassistant.core")
    const_mod = types.ModuleType("homeassistant.const")
    ce_mod = types.ModuleType("homeassistant.config_entries")
    util_mod = types.ModuleType("homeassistant.util")
    dt_mod = types.ModuleType("homeassistant.util.dt")
    helpers_mod = types.ModuleType("homeassistant.helpers")
    cv_mod = types.ModuleType("homeassistant.helpers.config_validation")
    disp_mod = types.ModuleType("homeassistant.helpers.dispatcher")
    ev_mod = types.ModuleType("homeassistant.helpers.event")
    store_mod = types.ModuleType("homeassistant.helpers.storage")
    sel_mod = types.ModuleType("homeassistant.helpers.selector")
    dr_mod = types.ModuleType("homeassistant.helpers.device_registry")
    ep_mod = types.ModuleType("homeassistant.helpers.entity_platform")
    comp_mod = types.ModuleType("homeassistant.components")
    sensor_mod = types.ModuleType("homeassistant.components.sensor")

    @dataclass
    class Context:
        id: str = "ctx_default"
        user_id: str | None = None
        parent_id: str | None = None

    class State:
        def __init__(
            self,
            entity_id: str,
            state: str,
            attributes: dict[str, Any] | None = None,
            context: Context | None = None,
            last_changed: datetime | None = None,
            last_updated: datetime | None = None,
        ) -> None:
            self.entity_id = entity_id
            self.domain, _, self.object_id = entity_id.partition(".")
            self.state = state
            self.attributes = dict(attributes or {})
            self.name = str(self.attributes.get("friendly_name", entity_id))
            self.context = context or Context()
            now = datetime.now(UTC)
            self.last_changed = last_changed or now
            self.last_updated = last_updated or now
            self.last_reported = now

    @dataclass
    class Event:
        event_type: str
        data: dict[str, Any] = field(default_factory=dict)
        context: Context = field(default_factory=Context)

    class HomeAssistant:
        pass

    class ServiceCall:
        data: dict[str, Any] = {}

    class ConfigEntry:
        entry_id: str = "test_entry"
        data: dict[str, Any] = {}
        options: dict[str, Any] = {}

    class ConfigFlow:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__()

    class OptionsFlow:
        pass

    class Store:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._saved: Any = None

        async def async_load(self) -> Any:
            return self._saved

        def async_delay_save(self, producer: Any, delay: float) -> None:
            self._saved = producer()

        async def async_save(self, data: Any) -> None:
            self._saved = data

    def callback(func: Any) -> Any:
        return func

    const_mod.EVENT_CALL_SERVICE = "call_service"  # type: ignore[attr-defined]
    const_mod.EVENT_HOMEASSISTANT_STOP = "homeassistant_stop"  # type: ignore[attr-defined]
    const_mod.EVENT_STATE_CHANGED = "state_changed"  # type: ignore[attr-defined]

    ce_mod.ConfigEntry = ConfigEntry  # type: ignore[attr-defined]
    ce_mod.ConfigFlow = ConfigFlow  # type: ignore[attr-defined]
    ce_mod.ConfigFlowResult = dict  # type: ignore[attr-defined]
    ce_mod.OptionsFlow = OptionsFlow  # type: ignore[attr-defined]

    cv_mod.string = str  # type: ignore[attr-defined]
    disp_mod.async_dispatcher_send = lambda *a, **kw: None  # type: ignore[attr-defined]
    disp_mod.async_dispatcher_connect = lambda *a, **kw: (lambda: None)  # type: ignore[attr-defined]
    ev_mod.async_track_time_interval = lambda *a, **kw: (lambda: None)  # type: ignore[attr-defined]
    ev_mod.async_call_later = lambda *a, **kw: (lambda: None)  # type: ignore[attr-defined]
    store_mod.Store = Store  # type: ignore[attr-defined]

    dt_mod.utcnow = lambda: datetime.now(UTC)  # type: ignore[attr-defined]
    dt_mod.as_local = lambda dt_val: dt_val  # type: ignore[attr-defined]
    util_mod.dt = dt_mod  # type: ignore[attr-defined]
    util_mod.slugify = lambda s: str(s).lower().replace(" ", "_")  # type: ignore[attr-defined]

    core_mod.Context = Context  # type: ignore[attr-defined]
    core_mod.State = State  # type: ignore[attr-defined]
    core_mod.Event = Event  # type: ignore[attr-defined]
    core_mod.HomeAssistant = HomeAssistant  # type: ignore[attr-defined]
    core_mod.ServiceCall = ServiceCall  # type: ignore[attr-defined]
    core_mod.CALLBACK_TYPE = Any  # type: ignore[attr-defined]
    core_mod.callback = callback  # type: ignore[attr-defined]

    helpers_mod.config_validation = cv_mod  # type: ignore[attr-defined]
    helpers_mod.dispatcher = disp_mod  # type: ignore[attr-defined]
    helpers_mod.event = ev_mod  # type: ignore[attr-defined]
    helpers_mod.storage = store_mod  # type: ignore[attr-defined]
    helpers_mod.selector = sel_mod  # type: ignore[attr-defined]

    sys.modules["homeassistant"] = ha_mod
    sys.modules["homeassistant.core"] = core_mod
    sys.modules["homeassistant.const"] = const_mod
    sys.modules["homeassistant.config_entries"] = ce_mod
    sys.modules["homeassistant.util"] = util_mod
    sys.modules["homeassistant.util.dt"] = dt_mod
    sys.modules["homeassistant.helpers"] = helpers_mod
    sys.modules["homeassistant.helpers.config_validation"] = cv_mod
    sys.modules["homeassistant.helpers.dispatcher"] = disp_mod
    sys.modules["homeassistant.helpers.event"] = ev_mod
    sys.modules["homeassistant.helpers.storage"] = store_mod
    sys.modules["homeassistant.helpers.selector"] = sel_mod
    sys.modules["homeassistant.helpers.device_registry"] = dr_mod
    sys.modules["homeassistant.helpers.entity_platform"] = ep_mod
    sys.modules["homeassistant.components"] = comp_mod
    sys.modules["homeassistant.components.sensor"] = sensor_mod


_install_ha_stubs_if_needed()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from homeassistant.core import Context, Event, State  # noqa: E402

from custom_components.rl_mpc_automator.attribution import (  # noqa: E402
    ActorKind,
    ContextTracker,
)
from custom_components.rl_mpc_automator.automation_builder import (  # noqa: E402
    AutomationBuilder,
)
from custom_components.rl_mpc_automator.const import (  # noqa: E402
    FEEDBACK_ALWAYS,
    FEEDBACK_NO,
    FEEDBACK_SOMETIMES,
    FEEDBACK_YES,
    SUPPORTED_DOMAINS,
)
from custom_components.rl_mpc_automator.domains import (  # noqa: E402
    DOMAIN_SPECS,
    DomainAdapter,
)
from custom_components.rl_mpc_automator.features import ContextSnapshot  # noqa: E402
from custom_components.rl_mpc_automator.model import (  # noqa: E402
    DecisionMode,
    OnlineRLMPCEngine,
    PatternKind,
)


class _DummyUser:
    def __init__(self, system_generated: bool = False) -> None:
        self.system_generated = system_generated


class _DummyHass:
    def __init__(self, tmp_dir: str) -> None:
        self._tmp_dir = tmp_dir
        self.fired_events: list[tuple[str, dict[str, Any]]] = []
        self.reloaded = 0

        class _Config:
            def path(self_inner, rel: str) -> str:
                return str(Path(tmp_dir) / rel)

        class _Auth:
            async def async_get_user(self_inner, uid: str) -> _DummyUser:
                return _DummyUser(system_generated=(uid == "sys_bot"))

        class _Services:
            def has_service(self_inner, dom: str, srv: str) -> bool:
                return dom == "automation" and srv == "reload"

            async def async_call(
                self_inner, dom: str, srv: str, data: Any = None, blocking: bool = False
            ) -> None:
                if dom == "automation" and srv == "reload":
                    self.reloaded += 1

        class _Bus:
            def async_fire(self_inner, ev_type: str, ev_data: dict[str, Any]) -> None:
                self.fired_events.append((ev_type, ev_data))

        self.config = _Config()
        self.auth = _Auth()
        self.services = _Services()
        self.bus = _Bus()

    async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
        return func(*args)


def _make_snapshot(
    ts: float,
    hour: int = 7,
    minute: int = 15,
    weekday: int = 1,
    person: str = "person.hubert",
    is_dark: bool = False,
) -> ContextSnapshot:
    return ContextSnapshot(
        utc_timestamp=ts,
        local_hour=hour,
        local_minute=minute,
        weekday=weekday,
        is_weekend=(weekday >= 5),
        slot_30m=(hour * 60 + minute) // 30,
        present_user_ids=("user_hubert",),
        present_person_entities=(person,),
        home_ratio=1.0,
        recent_arrival=False,
        illuminance_lux=30.0 if is_dark else 350.0,
        illuminance_entity_id="sensor.salon_lux",
        is_dark=is_dark,
        weather_condition="sunny",
        weather_entity_id="weather.dom",
        outdoor_temp=18.0,
        calendar_busy=False,
        calendar_entity_id="calendar.praca",
        active_binary_sensors=("binary_sensor.kuchnia_ruch",),
        automation_recently_active=False,
    )


class TestRLMPCAutomator(unittest.TestCase):
    """Weryfikacja wszystkich 10 Przykazań projektu."""

    def test_all_46_domains_and_force_update_dedup(self) -> None:
        """[Przykazanie #4, #7] Obsługa 46 domen i odrzucanie duplikatów MQTT force_update."""
        self.assertEqual(len(SUPPORTED_DOMAINS), 46)
        self.assertEqual(set(DOMAIN_SPECS.keys()), set(SUPPORTED_DOMAINS))

        adapter = DomainAdapter()
        s1 = State(
            "light.salon",
            "on",
            {"brightness": 200, "rssi": -61, "friendly_name": "Salon"},
        )
        s2_force_update = State(
            "light.salon",
            "on",
            {"brightness": 202, "rssi": -65, "friendly_name": "Salon"},
        )
        s3_turned_off = State("light.salon", "off", {"friendly_name": "Salon"})

        self.assertTrue(adapter.is_meaningful_change(None, s1))
        self.assertFalse(adapter.is_meaningful_change(s1, s2_force_update))
        self.assertTrue(adapter.is_meaningful_change(s2_force_update, s3_turned_off))

        garage = State(
            "cover.brama_garazowa",
            "open",
            {"device_class": "garage", "friendly_name": "Brama garażowa"},
        )
        act = adapter.extract_action_spec(garage)
        self.assertIsNotNone(act)
        assert act is not None
        self.assertTrue(act.safety_critical)
        self.assertEqual(act.service, "open_cover")

    def test_context_parent_chain_attribution(self) -> None:
        """[Przykazanie #3] Śledzenie łańcucha parent_id odróżnia człowieka od automatyzacji."""
        tracker = ContextTracker()
        hass = _DummyHass("/tmp")

        c_human_root = Context(id="c1", user_id="user_hubert", parent_id=None)
        c_auto = Context(id="c2", user_id=None, parent_id="c1")
        c_script = Context(id="c3", user_id=None, parent_id="c2")
        c_light = Context(id="c4", user_id=None, parent_id="c3")

        tracker.record_automation_triggered(
            Event("automation_triggered", {}, context=c_auto)
        )
        tracker.record_service_call(
            Event("call_service", {"domain": "script"}, context=c_script)
        )

        auto_light_state = State("light.korytarz", "on", context=c_light)
        res_auto = asyncio.run(
            tracker.async_classify_state_change(
                hass,  # type: ignore[arg-type]
                auto_light_state,
                is_actuator_or_button=True,
                allow_physical_actions=True,
            )
        )
        self.assertEqual(res_auto.actor_kind, ActorKind.AUTOMATION)
        self.assertFalse(res_auto.is_human)

        human_switch = State("switch.ekspres", "on", context=c_human_root)
        res_human = asyncio.run(
            tracker.async_classify_state_change(
                hass,  # type: ignore[arg-type]
                human_switch,
                is_actuator_or_button=True,
                allow_physical_actions=True,
            )
        )
        self.assertEqual(res_human.actor_kind, ActorKind.HUMAN_USER)
        self.assertTrue(res_human.is_human)
        self.assertEqual(res_human.user_id, "user_hubert")

    def test_garage_morning_routine_detection(self) -> None:
        """[Przykazanie #2, #6, #8] Wykrywanie porannego otwierania bramy garażowej bez wyzwalacza."""
        adapter = DomainAdapter()
        engine = OnlineRLMPCEngine(min_occurrences=3, ucb_alpha=0.7)
        garage = State(
            "cover.brama_garazowa",
            "open",
            {"device_class": "garage", "friendly_name": "Brama garażowa"},
        )
        garage_action = adapter.extract_action_spec(garage)
        assert garage_action is not None

        base_dt = datetime(2026, 11, 23, 7, 15, tzinfo=UTC)
        just_updated = []
        # Dla safety_critical wymagane są min. 4 powtórzenia w różne dni [Przykazanie #2]
        for day in range(4):
            ts = (base_dt + timedelta(days=day)).timestamp()
            snap = _make_snapshot(ts, hour=7, minute=15, weekday=day)
            just_updated = engine.observe_human_action(
                action=garage_action,
                snapshot=snap,
                user_id="user_hubert",
                local_date_str=f"2026-11-{23 + day:02d}",
            )

        snap_last = _make_snapshot(
            (base_dt + timedelta(days=3)).timestamp(), hour=7, minute=15, weekday=3
        )
        proposal = engine.plan_next_question(
            snapshot=snap_last,
            active_user_id="user_hubert",
            current_trigger_key=None,
            just_executed_patterns=just_updated,
            entity_current_states={"cover.brama_garazowa": "open"},
            domain_priority_fn=adapter.get_priority,
            fatigue_cost=0.0,
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.mode, DecisionMode.CONFIRM_ROUTINE)
        self.assertEqual(proposal.pattern.kind, PatternKind.ROUTINE_TIME)

    def test_coffee_to_blinds_mpc_and_4_feedback_actions(self) -> None:
        """[Przykazanie #5, #6, #8, #9, #10] Sekwencja kawa -> rolety, MPC, LinUCB i automations.yaml."""
        adapter = DomainAdapter()
        engine = OnlineRLMPCEngine(min_occurrences=3, ucb_alpha=0.7)

        coffee_state = State(
            "switch.ekspres_do_kawy",
            "on",
            {"friendly_name": "Ekspres do kawy"},
        )
        blinds_open_state = State(
            "cover.rolety_salon",
            "open",
            {"friendly_name": "Rolety salon"},
        )
        trig_key = adapter.make_trigger_key(coffee_state)
        assert trig_key is not None
        trig_desc = adapter.describe_trigger_pl(trig_key, coffee_state.name)
        blinds_action = adapter.extract_action_spec(blinds_open_state)
        assert blinds_action is not None

        base_dt = datetime(2026, 11, 23, 7, 10, tzinfo=UTC)

        for day_offset in range(3):
            t0 = (base_dt + timedelta(days=day_offset)).timestamp()
            date_str = f"2026-11-{23 + day_offset:02d}"
            snap_trig = _make_snapshot(t0, hour=7, minute=10, weekday=day_offset)
            engine.observe_trigger_event(
                trigger_key=trig_key,
                entity_id=coffee_state.entity_id,
                description_pl=trig_desc,
                snapshot=snap_trig,
                user_id="user_hubert",
                local_date_str=date_str,
            )
            snap_act = _make_snapshot(t0 + 35.0, hour=7, minute=11, weekday=day_offset)
            engine.observe_human_action(
                action=blinds_action,
                snapshot=snap_act,
                user_id="user_hubert",
                local_date_str=date_str,
            )

        t4 = (base_dt + timedelta(days=3)).timestamp()
        snap_day4 = _make_snapshot(t4, hour=7, minute=12, weekday=3)
        engine.observe_trigger_event(
            trigger_key=trig_key,
            entity_id=coffee_state.entity_id,
            description_pl=trig_desc,
            snapshot=snap_day4,
            user_id="user_hubert",
            local_date_str="2026-11-26",
        )

        proposal = engine.plan_next_question(
            snapshot=snap_day4,
            active_user_id="user_hubert",
            current_trigger_key=trig_key,
            just_executed_patterns=[],
            entity_current_states={"cover.rolety_salon": "closed"},
            domain_priority_fn=adapter.get_priority,
            fatigue_cost=0.0,
        )
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.mode, DecisionMode.PROACTIVE)
        self.assertEqual(proposal.pattern.kind, PatternKind.SEQUENTIAL)
        self.assertEqual(proposal.target_user_id, "user_hubert")

        pat, reward = engine.apply_user_feedback(
            pattern_id=proposal.pattern.pattern_id,
            feedback=FEEDBACK_SOMETIMES,
            feature_vector=proposal.feature_vector,
            snapshot=snap_day4,
        )
        self.assertAlmostEqual(reward, 0.75)
        assert pat is not None

        with tempfile.TemporaryDirectory() as tmp_dir:
            hass = _DummyHass(tmp_dir)
            builder = AutomationBuilder(hass)  # type: ignore[arg-type]
            auto_id, cond_descs = asyncio.run(
                builder.async_create_or_update(pat, conditional=True)
            )
            self.assertTrue(auto_id.startswith("rl_mpc_"))
            self.assertGreater(len(cond_descs), 0)
            yaml_content = (Path(tmp_dir) / "automations.yaml").read_text(encoding="utf-8")
            self.assertIn("switch.ekspres_do_kawy", yaml_content)
            self.assertIn("cover.open_cover", yaml_content)

            revoked = asyncio.run(builder.async_revoke(pat.pattern_id))
            self.assertTrue(revoked)
            yaml_after = (Path(tmp_dir) / "automations.yaml").read_text(encoding="utf-8")
            self.assertNotIn(auto_id, yaml_after)

        # Test serializacji i deserializacji modelu (.storage) [Przykazanie #1]
        serialized = engine.to_dict()
        restored_engine = OnlineRLMPCEngine()
        restored_engine.load_from_dict(serialized)
        self.assertIn(pat.pattern_id, restored_engine.patterns)

        _, r_yes = engine.apply_user_feedback(
            pattern_id=pat.pattern_id,
            feedback=FEEDBACK_YES,
            feature_vector=proposal.feature_vector,
            snapshot=snap_day4,
        )
        self.assertAlmostEqual(r_yes, 0.50)

        _, r_no = engine.apply_user_feedback(
            pattern_id=pat.pattern_id,
            feedback=FEEDBACK_NO,
            feature_vector=proposal.feature_vector,
            snapshot=snap_day4,
        )
        self.assertAlmostEqual(r_no, -0.85)
        self.assertGreater(pat.cooldown_until_ts, t4 + 3600)

        _, r_always = engine.apply_user_feedback(
            pattern_id=pat.pattern_id,
            feedback=FEEDBACK_ALWAYS,
            feature_vector=proposal.feature_vector,
            snapshot=snap_day4,
        )
        self.assertAlmostEqual(r_always, 1.00)
        self.assertTrue(pat.automated)


if __name__ == "__main__":
    unittest.main()
