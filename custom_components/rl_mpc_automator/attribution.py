"""Śledzenie łańcucha kontekstu (`context.id`, `parent_id`, `user_id`) [Przykazanie #3].

W Home Assistancie automatyzacja wyzwolona np. ruchem człowieka lub czasem tworzy nowy
obiekt `Context`, którego `parent_id` wskazuje na kontekst wyzwalacza, a `user_id` wynosi `None`.
Samo sprawdzenie jednego kontekstu nie wystarcza, gdy automatyzacja wywołuje skrypt, a skrypt
scenę lub grupę — dlatego `ContextTracker` utrzymuje graf ostatnich kontekstów w buforze LRU
i przechodzi w górę łańcucha `parent_id`, aby niezawodnie odróżnić intencjonalną akcję człowieka
od kaskady automatyzacji.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from homeassistant.core import Context, Event, HomeAssistant, State

MAX_TRACKED_CONTEXTS = 4096
MAX_PARENT_CHAIN_DEPTH = 16


class ActorKind(StrEnum):
    """Klasyfikacja sprawcy zmiany stanu encji."""

    HUMAN_USER = "human_user"  # Uwierzytelniony użytkownik HA (aplikacja / dashboard / głos)
    HUMAN_PHYSICAL = "human_physical"  # Fizyczny przełącznik lokalny / pilot bez rodzica-automatyzacji
    AUTOMATION = "automation"  # Własna lub cudza automatyzacja / skrypt
    SYSTEM = "system"  # Integracja tła, harmonogram systemowy lub czujnik pasywny


@dataclass(frozen=True, slots=True)
class ContextNode:
    """Pojedynczy węzeł w grafie przyczynowości kontekstów HA."""

    context_id: str
    parent_id: str | None
    user_id: str | None
    is_automation: bool
    origin_tag: str


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """Wynik analizy pochodzenia zdarzenia `state_changed`."""

    actor_kind: ActorKind
    user_id: str | None
    root_context_id: str
    chain_depth: int
    reason: str

    @property
    def is_human(self) -> bool:
        """Zwraca True wyłącznie dla działań człowieka [Przykazanie #3]."""
        return self.actor_kind in (ActorKind.HUMAN_USER, ActorKind.HUMAN_PHYSICAL)


class ContextTracker:
    """Bezpieczny pamięciowo graf kontekstów do śledzenia `context.parent_id`."""

    def __init__(self) -> None:
        self._nodes: OrderedDict[str, ContextNode] = OrderedDict()
        self._system_user_cache: dict[str, bool] = {}

    def _put_node(self, node: ContextNode) -> None:
        existing = self._nodes.get(node.context_id)
        if existing is not None:
            # Jeśli węzeł już był oznaczony jako automatyzacja, nigdy nie zdejmujemy tej flagi
            merged_auto = existing.is_automation or node.is_automation
            merged_parent = node.parent_id or existing.parent_id
            merged_user = node.user_id or existing.user_id
            node = ContextNode(
                context_id=node.context_id,
                parent_id=merged_parent,
                user_id=merged_user,
                is_automation=merged_auto,
                origin_tag=existing.origin_tag if existing.is_automation else node.origin_tag,
            )
            self._nodes.move_to_end(node.context_id)
        self._nodes[node.context_id] = node
        while len(self._nodes) > MAX_TRACKED_CONTEXTS:
            self._nodes.popitem(last=False)

    def register_own_context(self, context: Context) -> None:
        """Rejestruje kontekst wygenerowany przez sam dodatek (np. po kliknięciu 'Tak')."""
        self._put_node(
            ContextNode(
                context_id=context.id,
                parent_id=context.parent_id,
                user_id=context.user_id,
                is_automation=True,
                origin_tag="own_rl_mpc",
            )
        )

    def record_automation_triggered(self, event: Event[dict[str, Any]]) -> None:
        """Oznacza kontekst zdarzenia `automation_triggered` lub `script_started`."""
        ctx = event.context
        self._put_node(
            ContextNode(
                context_id=ctx.id,
                parent_id=ctx.parent_id,
                user_id=ctx.user_id,
                is_automation=True,
                origin_tag=event.event_type,
            )
        )

    def record_service_call(self, event: Event[dict[str, Any]]) -> None:
        """Śledzi wywołania usług `automation.*` i `script.*` oraz ich `parent_id`."""
        domain = str(event.data.get("domain", ""))
        ctx = event.context
        is_auto = domain in ("automation", "script")
        self._put_node(
            ContextNode(
                context_id=ctx.id,
                parent_id=ctx.parent_id,
                user_id=ctx.user_id,
                is_automation=is_auto,
                origin_tag=f"service:{domain}",
            )
        )

    async def _is_system_user(self, hass: HomeAssistant, user_id: str) -> bool:
        """Sprawdza w rejestrze auth HA, czy konto jest wygenerowane systemowo (np. Supervisor)."""
        if user_id in self._system_user_cache:
            return self._system_user_cache[user_id]
        user = await hass.auth.async_get_user(user_id)
        is_sys = bool(user.system_generated) if user is not None else False
        self._system_user_cache[user_id] = is_sys
        return is_sys

    async def async_classify_state_change(
        self,
        hass: HomeAssistant,
        new_state: State,
        *,
        is_actuator_or_button: bool,
        allow_physical_actions: bool,
        sole_present_user_id: str | None = None,
    ) -> AttributionResult:
        """Klasyfikuje sprawcę zmiany stanu po prześledzeniu pełnego łańcucha `parent_id`."""
        ctx = new_state.context
        domain = new_state.domain
        is_auto_entity = domain in ("automation", "script")

        self._put_node(
            ContextNode(
                context_id=ctx.id,
                parent_id=ctx.parent_id,
                user_id=ctx.user_id,
                is_automation=is_auto_entity,
                origin_tag=f"state:{domain}",
            )
        )

        if is_auto_entity:
            return AttributionResult(
                actor_kind=ActorKind.AUTOMATION,
                user_id=None,
                root_context_id=ctx.id,
                chain_depth=0,
                reason="automation_entity",
            )

        # Przejście w górę łańcucha parent_id (ochrona przed cyklami i obcięciem bufora)
        visited: set[str] = set()
        curr_id: str | None = ctx.id
        root_id: str = ctx.id
        found_user_id: str | None = ctx.user_id
        depth = 0

        while curr_id is not None and curr_id not in visited and depth < MAX_PARENT_CHAIN_DEPTH:
            visited.add(curr_id)
            root_id = curr_id
            node = self._nodes.get(curr_id)
            if node is None:
                break
            if node.is_automation:
                return AttributionResult(
                    actor_kind=ActorKind.AUTOMATION,
                    user_id=None,
                    root_context_id=root_id,
                    chain_depth=depth,
                    reason=f"ancestor_automation:{node.origin_tag}",
                )
            if found_user_id is None and node.user_id is not None:
                found_user_id = node.user_id
            curr_id = node.parent_id
            if curr_id is not None:
                depth += 1

        # Jeżeli kontekst miał rodzica, którego nie znamy, a nie ma user_id -> traktujemy ostrożnie jako automatyzację
        if ctx.parent_id is not None and found_user_id is None:
            return AttributionResult(
                actor_kind=ActorKind.AUTOMATION,
                user_id=None,
                root_context_id=ctx.parent_id,
                chain_depth=max(1, depth),
                reason="unattributed_parent_chain",
            )

        # Jawny user_id człowieka z aplikacji mobilnej, interfejsu WWW lub asystenta
        if found_user_id is not None:
            if await self._is_system_user(hass, found_user_id):
                return AttributionResult(
                    actor_kind=ActorKind.SYSTEM,
                    user_id=found_user_id,
                    root_context_id=root_id,
                    chain_depth=depth,
                    reason="system_generated_user",
                )
            return AttributionResult(
                actor_kind=ActorKind.HUMAN_USER,
                user_id=found_user_id,
                root_context_id=root_id,
                chain_depth=depth,
                reason="authenticated_user",
            )

        # Brak user_id i brak parent_id na aktuatorze (np. fizyczny klawisz ścienny lub pilot do bramy)
        if is_actuator_or_button and allow_physical_actions and ctx.parent_id is None:
            return AttributionResult(
                actor_kind=ActorKind.HUMAN_PHYSICAL,
                user_id=sole_present_user_id,
                root_context_id=root_id,
                chain_depth=0,
                reason="physical_local_actuator",
            )

        return AttributionResult(
            actor_kind=ActorKind.SYSTEM,
            user_id=None,
            root_context_id=root_id,
            chain_depth=depth,
            reason="passive_or_background_state",
        )
