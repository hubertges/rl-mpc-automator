"""Stałe konfiguracyjne i referencyjne dla integracji RL+MPC Domowy Automatyzator."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "rl_mpc_automator"
PLATFORMS: Final[list[str]] = ["sensor"]

# [Przykazanie #1] Wyłącznie lokalny zapis w .storage Home Assistanta
STORAGE_VERSION: Final = 1
STORAGE_KEY: Final = f"{DOMAIN}.learned_state"
SAVE_DELAY_SECONDS: Final = 30

# [Przykazanie #7] Kompletna, wiążąca lista wszystkich 46 obsługiwanych domen HA
SUPPORTED_DOMAINS: Final[tuple[str, ...]] = (
    "ai_task",
    "air_quality",
    "alarm_control_panel",
    "assist_satellite",
    "binary_sensor",
    "button",
    "calendar",
    "camera",
    "climate",
    "conversation",
    "cover",
    "date",
    "datetime",
    "device_tracker",
    "event",
    "fan",
    "geo_location",
    "humidifier",
    "image",
    "image_processing",
    "infrared",
    "lawn_mower",
    "light",
    "lock",
    "media_player",
    "notify",
    "number",
    "radio_frequency",
    "remote",
    "scene",
    "select",
    "sensor",
    "siren",
    "stt",
    "switch",
    "tag",
    "text",
    "time",
    "todo",
    "tts",
    "update",
    "vacuum",
    "valve",
    "wake_word",
    "water_heater",
    "weather",
)

# Domeny o podwyższonym priorytecie eksploracji i bogatych adapterach cech
PRIMARY_EMPHASIS_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "light",
        "cover",
        "lock",
        "switch",
        "climate",
        "media_player",
        "binary_sensor",
        "sensor",
        "device_tracker",
        "calendar",
        "weather",
        "notify",
    }
)

# Klucze ConfigFlow / OptionsFlow
CONF_NOTIFY_SERVICE: Final = "notify_service"
CONF_ALLOW_PHYSICAL_ACTIONS: Final = "allow_physical_actions"
CONF_MIN_OCCURRENCES: Final = "min_occurrences"
CONF_UCB_ALPHA: Final = "ucb_alpha"
CONF_MIN_COOLDOWN_MINUTES: Final = "min_cooldown_minutes"
CONF_MAX_QUESTIONS_PER_DAY: Final = "max_questions_per_day"
CONF_EXCLUDED_ENTITIES: Final = "excluded_entities"

# Wartości domyślne parametrów
DEFAULT_NOTIFY_SERVICE: Final = "notify.notify"
DEFAULT_ALLOW_PHYSICAL_ACTIONS: Final = True
DEFAULT_MIN_OCCURRENCES: Final = 3
DEFAULT_SAFETY_MIN_OCCURRENCES: Final = 4
DEFAULT_UCB_ALPHA: Final = 0.65
DEFAULT_MIN_COOLDOWN_MINUTES: Final = 15
DEFAULT_MAX_QUESTIONS_PER_HOUR: Final = 3
DEFAULT_MAX_QUESTIONS_PER_DAY: Final = 10
DEFAULT_MAX_PENDING_QUESTIONS: Final = 2
DEFAULT_QUESTION_TTL_SECONDS: Final = 1800  # 30 minut na odpowiedź

# Horyzont MPC i okno sekwencji zdarzeń
SEQUENCE_MIN_DELAY_SECONDS: Final = 1.5
SEQUENCE_WINDOW_SECONDS: Final = 600.0  # 10 minut na powiązanie np. kawa -> rolety
MPC_HORIZON_STEPS: Final = 6
MPC_STEP_SECONDS: Final = 150.0  # 6 * 150s = 15 minut predykcji w przód
MPC_DISCOUNT_GAMMA: Final = 0.92
HIERARCHICAL_SHRINKAGE_KAPPA: Final = 3.0

# Przestrzeń RL (Hybrid LinUCB) i funkcja nagrody [Przykazanie #8, #9]
RL_FEATURE_DIM: Final = 8
COND_FEATURE_DIM: Final = 10
DEFAULT_ASK_THRESHOLD: Final = 0.38
BASE_PATTERN_COOLDOWN_SECONDS: Final = 3600.0  # 1h bazowego backoffu

REWARD_ALWAYS: Final = 1.00
REWARD_SOMETIMES: Final = 0.75
REWARD_YES: Final = 0.50
REWARD_NO: Final = -0.85
REWARD_EXPIRED: Final = -0.15

# [Przykazanie #9] Akcje powiadomień Android Companion App
EVENT_MOBILE_ACTION: Final = "mobile_app_notification_action"
ACTION_PREFIX: Final = "RLMPC"
FEEDBACK_YES: Final = "YES"
FEEDBACK_NO: Final = "NO"
FEEDBACK_ALWAYS: Final = "ALWAYS"
FEEDBACK_SOMETIMES: Final = "SOMETIMES"

VALID_FEEDBACK_TYPES: Final[frozenset[str]] = frozenset(
    {FEEDBACK_YES, FEEDBACK_NO, FEEDBACK_ALWAYS, FEEDBACK_SOMETIMES}
)

# Zdarzenia i usługi integracji [Przykazanie #10]
AUTOMATION_ID_PREFIX: Final = "rl_mpc_"
EVENT_AUTOMATION_CREATED: Final = f"{DOMAIN}_automation_created"
EVENT_AUTOMATION_REVOKED: Final = f"{DOMAIN}_automation_revoked"

SERVICE_REVOKE_AUTOMATION: Final = "revoke_automation"
SERVICE_RESET_PATTERN: Final = "reset_pattern"
SERVICE_EVALUATE_NOW: Final = "evaluate_now"
