"""Trwałe, wyłącznie lokalne przechowywanie stanu modelu RL+MPC w `.storage` [Przykazanie #1]."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import SAVE_DELAY_SECONDS, STORAGE_KEY, STORAGE_VERSION
from .model import OnlineRLMPCEngine

_LOGGER = logging.getLogger(__name__)


class ModelStorage:
    """Opakowanie nad `homeassistant.helpers.storage.Store` z opóźnionym zapisem nieblokującym."""

    def __init__(self, hass: HomeAssistant, entry_id: str, engine: OnlineRLMPCEngine) -> None:
        self._engine = engine
        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            f"{STORAGE_KEY}.{entry_id}",
            private=True,
            atomic_writes=True,
        )

    async def async_load(self) -> None:
        """Wczytuje zapisane wagi RL i wyuczone wzorce z lokalnego pliku JSON."""
        data = await self._store.async_load()
        if isinstance(data, dict):
            self._engine.load_from_dict(data)
            _LOGGER.info(
                "Wczytano model RL+MPC z .storage (%d wzorców, skumulowana nagroda=%.2f)",
                len(self._engine.patterns),
                self._engine.policy.cumulative_reward,
            )

    @callback
    def schedule_save(self) -> None:
        """Zleca odroczony zapis w tle, aby częste zdarzenia online nie obciążały dysku/karty SD."""
        data_producer: Callable[[], dict[str, Any]] = self._engine.to_dict
        self._store.async_delay_save(data_producer, SAVE_DELAY_SECONDS)

    async def async_save_now(self) -> None:
        """Wymusza natychmiastowy zapis (po odpowiedzi użytkownika lub przy zamykaniu HA)."""
        await self._store.async_save(self._engine.to_dict())
