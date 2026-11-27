"""Config Flow oraz Options Flow dla integracji RL+MPC Domowy Automatyzator."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector

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
)


def _discover_notify_options(hass: HomeAssistant) -> list[selector.SelectOptionDict]:
    """Wykrywa zarejestrowane usługi `notify.mobile_app_*` dla telefonów z Androidem."""
    notify_services = hass.services.async_services().get("notify", {})
    options: list[selector.SelectOptionDict] = []
    for srv_name in sorted(notify_services):
        full_name = f"notify.{srv_name}"
        label = f"{full_name} (Companion)" if srv_name.startswith("mobile_app_") else full_name
        options.append(selector.SelectOptionDict(value=full_name, label=label))

    if not options:
        options.append(
            selector.SelectOptionDict(
                value=DEFAULT_NOTIFY_SERVICE,
                label=f"{DEFAULT_NOTIFY_SERVICE} (domyślna)",
            )
        )
    return options


def _build_schema(hass: HomeAssistant, current: dict[str, Any]) -> vol.Schema:
    notify_opts = _discover_notify_options(hass)
    default_notify = current.get(CONF_NOTIFY_SERVICE) or notify_opts[0]["value"]

    return vol.Schema(
        {
            vol.Required(CONF_NOTIFY_SERVICE, default=default_notify): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=notify_opts,
                    custom_value=True,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_ALLOW_PHYSICAL_ACTIONS,
                default=current.get(CONF_ALLOW_PHYSICAL_ACTIONS, DEFAULT_ALLOW_PHYSICAL_ACTIONS),
            ): selector.BooleanSelector(),
            vol.Required(
                CONF_MIN_OCCURRENCES,
                default=current.get(CONF_MIN_OCCURRENCES, DEFAULT_MIN_OCCURRENCES),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=2, max=15, step=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_UCB_ALPHA,
                default=current.get(CONF_UCB_ALPHA, DEFAULT_UCB_ALPHA),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1, max=2.0, step=0.05, mode=selector.NumberSelectorMode.SLIDER
                )
            ),
            vol.Required(
                CONF_MIN_COOLDOWN_MINUTES,
                default=current.get(CONF_MIN_COOLDOWN_MINUTES, DEFAULT_MIN_COOLDOWN_MINUTES),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=240,
                    step=1,
                    unit_of_measurement="min",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_MAX_QUESTIONS_PER_DAY,
                default=current.get(CONF_MAX_QUESTIONS_PER_DAY, DEFAULT_MAX_QUESTIONS_PER_DAY),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, max=50, step=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_EXCLUDED_ENTITIES,
                default=list(current.get(CONF_EXCLUDED_ENTITIES) or []),
            ): selector.EntitySelector(selector.EntitySelectorConfig(multiple=True)),
        }
    )


class RLMPCAutomatorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Kreator konfiguracji początkowej w interfejsie Home Assistanta."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Pojedyncza instancja koordynatora na dom."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        if user_input is not None:
            return self.async_create_entry(
                title="RL+MPC Domowy Automatyzator",
                data={},
                options=user_input,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=_build_schema(self.hass, {}),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Udostępnia edycję parametrów RL/MPC oraz limitów pytań w locie."""
        return RLMPCAutomatorOptionsFlow(config_entry)


class RLMPCAutomatorOptionsFlow(OptionsFlow):
    """Edycja opcji bez utraty wyuczonego modelu."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        merged = {**self._entry.data, **self._entry.options}
        return self.async_show_form(
            step_id="init",
            data_schema=_build_schema(self.hass, merged),
        )
