"""Obsługa powiadomień aktywnych Home Assistant Companion na Androida [Przykazanie #8, #9].

Wysyła zwięzłe, kontekstowe pytania z 4 przyciskami akcji (`Tak` / `Nie` / `Zawsze` / `Czasami`),
pilnuje budżetu uwagi domowników (globalny cooldown, limity godzinowe i dzienne, limit pytań
oczekujących) oraz odbiera zdarzenia `mobile_app_notification_action` bez opóźnień wsadowych.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import Any
import uuid

from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util, slugify

from .const import (
    ACTION_PREFIX,
    DEFAULT_MAX_PENDING_QUESTIONS,
    DEFAULT_MAX_QUESTIONS_PER_DAY,
    DEFAULT_MAX_QUESTIONS_PER_HOUR,
    DEFAULT_QUESTION_TTL_SECONDS,
    EVENT_MOBILE_ACTION,
    FEEDBACK_ALWAYS,
    FEEDBACK_NO,
    FEEDBACK_SOMETIMES,
    FEEDBACK_YES,
    VALID_FEEDBACK_TYPES,
)
from .features import ContextSnapshot, PersonMapping
from .model import DecisionMode, PatternKind, QuestionProposal

_LOGGER = logging.getLogger(__name__)

FeedbackCallback = Callable[["PendingQuestion", str], Awaitable[None]]


@dataclass(slots=True)
class PendingQuestion:
    """Oczekujące na odpowiedź pytanie wysłane na telefon z Androidem."""

    question_id: str
    pattern_id: str
    mode: DecisionMode
    feature_vector: list[float]
    snapshot: ContextSnapshot
    notify_target: str
    created_ts: float
    cancel_timer: CALLBACK_TYPE | None = None


class CompanionNotifier:
    """Zarządza wysyłką pytań na Androida, budżetem częstotliwości i odpowiedziami."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        default_notify_service: str,
        min_cooldown_seconds: float,
        max_per_hour: int = DEFAULT_MAX_QUESTIONS_PER_HOUR,
        max_per_day: int = DEFAULT_MAX_QUESTIONS_PER_DAY,
        on_feedback: FeedbackCallback,
    ) -> None:
        self._hass = hass
        self.default_notify_service = default_notify_service
        self.min_cooldown_seconds = max(30.0, min_cooldown_seconds)
        self.max_per_hour = max(1, max_per_hour)
        self.max_per_day = max(1, max_per_day)
        self._on_feedback = on_feedback

        self._pending: dict[str, PendingQuestion] = {}
        self._sent_timestamps: deque[float] = deque(maxlen=256)
        self._unsub_action: CALLBACK_TYPE | None = None

    def start(self) -> None:
        """Rejestruje nasłuchiwanie kliknięć przycisków z aplikacji HA Companion."""
        if self._unsub_action is None:
            self._unsub_action = self._hass.bus.async_listen(
                EVENT_MOBILE_ACTION, self._async_handle_mobile_action
            )

    def stop(self) -> None:
        """Zwalnia listenery i timery wygasania przy wyładowaniu integracji."""
        if self._unsub_action is not None:
            self._unsub_action()
            self._unsub_action = None
        for pq in list(self._pending.values()):
            if pq.cancel_timer is not None:
                pq.cancel_timer()
        self._pending.clear()

    def _prune_history(self, now_ts: float) -> None:
        while self._sent_timestamps and (now_ts - self._sent_timestamps[0]) > 86400.0:
            self._sent_timestamps.popleft()

    def sent_last_hour(self, now_ts: float) -> int:
        self._prune_history(now_ts)
        return sum(1 for ts in self._sent_timestamps if (now_ts - ts) <= 3600.0)

    def sent_last_day(self, now_ts: float) -> int:
        self._prune_history(now_ts)
        return len(self._sent_timestamps)

    def remaining_daily_budget(self, now_ts: float | None = None) -> int:
        ts = now_ts if now_ts is not None else dt_util.utcnow().timestamp()
        return max(0, self.max_per_day - self.sent_last_day(ts))

    def current_fatigue_cost(self, now_ts: float) -> float:
        """Koszt zmęczenia powiadomieniami odejmowany od UCB w polityce RL [Przykazanie #8]."""
        hour_ratio = self.sent_last_hour(now_ts) / self.max_per_hour
        day_ratio = self.sent_last_day(now_ts) / self.max_per_day
        pending_pen = 0.12 * len(self._pending)
        return round(0.25 * hour_ratio + 0.20 * day_ratio + pending_pen, 4)

    def can_send_now(self, now_ts: float) -> bool:
        """Twarda bramka antyspamowa przed wysłaniem powiadomienia."""
        self._prune_history(now_ts)
        if len(self._pending) >= DEFAULT_MAX_PENDING_QUESTIONS:
            return False
        if self._sent_timestamps and (now_ts - self._sent_timestamps[-1]) < self.min_cooldown_seconds:
            return False
        if self.sent_last_hour(now_ts) >= self.max_per_hour:
            return False
        if self.sent_last_day(now_ts) >= self.max_per_day:
            return False
        return True

    def _resolve_notify_service(
        self, target_user_id: str | None, person_mappings: list[PersonMapping]
    ) -> tuple[str, str]:
        """Dobiera usługę `notify.mobile_app_*` konkretnego domownika lub domyślną domu."""
        notify_services = self._hass.services.async_services().get("notify", {})

        if target_user_id:
            for pm in person_mappings:
                if pm.user_id == target_user_id:
                    for tracker_ent in pm.device_trackers:
                        obj_id = tracker_ent.split(".", 1)[-1]
                        candidate = f"mobile_app_{slugify(obj_id)}"
                        if candidate in notify_services:
                            return ("notify", candidate)
                    name_candidate = f"mobile_app_{slugify(pm.friendly_name)}"
                    if name_candidate in notify_services:
                        return ("notify", name_candidate)

        configured = self.default_notify_service
        if "." in configured:
            dom, srv = configured.split(".", 1)
            if dom == "notify" and srv in notify_services:
                return ("notify", srv)

        # Fallback: pierwsza dostępna usługa mobile_app_* lub notify.notify
        for srv_name in notify_services:
            if srv_name.startswith("mobile_app_"):
                return ("notify", srv_name)
        return ("notify", "notify")

    @staticmethod
    def _format_question_text(proposal: QuestionProposal) -> tuple[str, str]:
        """Tworzy zwięzły tytuł i treść pytania po polsku."""
        pat = proposal.pattern
        safety_prefix = "[Bezpieczeństwo] " if pat.action.safety_critical else ""

        if proposal.mode == DecisionMode.PROACTIVE:
            title = f"{safety_prefix}Propozycja akcji i automatyzacji"
            if pat.kind == PatternKind.SEQUENTIAL:
                delay_info = f" (zwykle po ~{int(round(pat.delay_mean_sec))} s)" if pat.delay_mean_sec >= 5 else ""
                msg = (
                    f"Wykryto wzorzec ({pat.household_occurrences}x): {pat.trigger_description_pl} "
                    f"zwykle chcesz {pat.action.description_pl}{delay_info}. "
                    f"Wykonać teraz czy zautomatyzować?"
                )
            else:
                msg = (
                    f"Zbliża się pora nawyku ({pat.distinct_days} dni, {pat.trigger_description_pl}): "
                    f"{pat.action.description_pl}. Wykonać czy zautomatyzować?"
                )
        else:
            title = f"{safety_prefix}Nowy kandydat na automatyzację"
            if pat.kind == PatternKind.SEQUENTIAL:
                msg = (
                    f"Już {pat.household_occurrences}x {pat.trigger_description_pl} ręcznie wybrałeś: "
                    f"{pat.action.description_pl}. Zautomatyzować ten wzorzec?"
                )
            else:
                msg = (
                    f"Wykryto ręczną rutynę ({pat.distinct_days} dni, {pat.trigger_description_pl}): "
                    f"{pat.action.description_pl} bez automatyzacji. Utworzyć regułę?"
                )
        return (title, msg)

    async def async_send_question(
        self,
        proposal: QuestionProposal,
        person_mappings: list[PersonMapping],
    ) -> str | None:
        """Wysyła aktywne powiadomienie z 4 akcjami na Androida i rejestruje timeout."""
        now_ts = proposal.snapshot.utc_timestamp
        if not self.can_send_now(now_ts):
            return None

        domain, service = self._resolve_notify_service(proposal.target_user_id, person_mappings)
        if not self._hass.services.has_service(domain, service):
            _LOGGER.debug("Brak usługi powiadomień %s.%s — pomijam pytanie", domain, service)
            return None

        qid = uuid.uuid4().hex[:10]
        title, message = self._format_question_text(proposal)
        tag = f"rl_mpc_{proposal.pattern.pattern_id}"

        # Format zgodny ze specyfikacją Home Assistant Companion dla Androida [Przykazanie #9]
        payload: dict[str, Any] = {
            "title": title,
            "message": message,
            "data": {
                "tag": tag,
                "group": "rl_mpc_automator",
                "ttl": 0,
                "priority": "high",
                "channel": "Propozycje automatyzacji RL+MPC",
                "actions": [
                    {"action": f"{ACTION_PREFIX}_{FEEDBACK_YES}_{qid}", "title": "Tak"},
                    {"action": f"{ACTION_PREFIX}_{FEEDBACK_NO}_{qid}", "title": "Nie"},
                    {"action": f"{ACTION_PREFIX}_{FEEDBACK_ALWAYS}_{qid}", "title": "Zawsze"},
                    {"action": f"{ACTION_PREFIX}_{FEEDBACK_SOMETIMES}_{qid}", "title": "Czasami"},
                ],
            },
        }

        pq = PendingQuestion(
            question_id=qid,
            pattern_id=proposal.pattern.pattern_id,
            mode=proposal.mode,
            feature_vector=proposal.feature_vector,
            snapshot=proposal.snapshot,
            notify_target=f"{domain}.{service}",
            created_ts=now_ts,
        )

        @callback
        def _on_expired(_now: Any) -> None:
            expired_q = self._pending.pop(qid, None)
            if expired_q is not None:
                self._hass.async_create_task(self._on_feedback(expired_q, "EXPIRED"))

        pq.cancel_timer = async_call_later(
            self._hass, DEFAULT_QUESTION_TTL_SECONDS, _on_expired
        )
        self._pending[qid] = pq
        self._sent_timestamps.append(now_ts)

        await self._hass.services.async_call(domain, service, payload, blocking=False)
        _LOGGER.info(
            "Wysłano pytanie %s (wzorzec=%s, tryb=%s, UCB=%.3f) przez %s.%s",
            qid,
            proposal.pattern.pattern_id,
            proposal.mode.value,
            proposal.ucb_score,
            domain,
            service,
        )
        return qid

    async def async_send_confirmation(
        self, notify_target: str, title: str, message: str, tag: str
    ) -> None:
        """Wysyła krótkie potwierdzenie zastępujące powiadomienie (ten sam `tag` na Androidzie)."""
        domain, _, service = notify_target.partition(".")
        if not service or not self._hass.services.has_service(domain, service):
            return
        await self._hass.services.async_call(
            domain,
            service,
            {
                "title": title,
                "message": message,
                "data": {"tag": tag, "group": "rl_mpc_automator"},
            },
            blocking=False,
        )

    @callback
    def _async_handle_mobile_action(self, event: Event[dict[str, Any]]) -> None:
        """Odbiera `mobile_app_notification_action` i natychmiast przekazuje do modelu RL."""
        raw_action = str(event.data.get("action", ""))
        prefix = f"{ACTION_PREFIX}_"
        if not raw_action.startswith(prefix):
            return

        remainder = raw_action[len(prefix) :]
        feedback_type, sep, qid = remainder.partition("_")
        if not sep or feedback_type not in VALID_FEEDBACK_TYPES:
            return

        pq = self._pending.pop(qid, None)
        if pq is None:
            _LOGGER.debug("Otrzymano akcję %s dla nieaktywnego już pytania %s", feedback_type, qid)
            return

        if pq.cancel_timer is not None:
            pq.cancel_timer()
            pq.cancel_timer = None

        self._hass.async_create_task(self._on_feedback(pq, feedback_type))
