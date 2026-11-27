"""Silnik uczenia online: Model Predictive Control (MPC) + Reinforcement Learning (Hybrid LinUCB).

Formalizacja matematyczna systemu sterowania pytaniami i wykrywania nawyków:
- Przestrzeń stanów S:
    s_t = (x_ctx(t), H_seq(t), u_t), gdzie x_ctx(t) in R^10 opisuje czas kołowy, dzień tygodnia,
    obecność domowników, oświetlenie (lux), pogodę i kalendarz; H_seq(t) to bufor ostatnich
    zdarzeń wyzwalających w oknie [t - 600s, t]; u_t to identyfikator domownika (lub 'household').
- Przestrzeń akcji agenta RL A_RL:
    A_RL(s_t) = {WAIT} U {(ASK, p) : p in P_cand(s_t)}, gdzie p to kandydat na automatyzację
    (sekwencyjny 'wyzwalacz -> akcja' albo rutynowy 'okno_czasowe -> akcja').
- Planowanie MPC na skończonym horyzoncie H:
    Dla h = 0..H-1 (H=6 kroków po 150s = 15 min) MPC propaguje czas t + h*dt oraz graf przejść
    zdarzeń T(e' | e), obliczając zdyskontowane prawdopodobieństwo wystąpienia ręcznej akcji
    domownika: J_MPC(s_t, p) = max_{0 <= h < H} gamma^h * P_hier(p | s_{t+h}, u_t).
- Hierarchia domownik + dom [Przykazanie #6]:
    P_hier(p | s, u) = lambda(n_u) * P_u(p | s) + (1 - lambda(n_u)) * P_house(p | s),
    gdzie lambda(n_u) = n_u / (n_u + kappa).
- Kompromis eksploracja / eksploatacja (Hybrid LinUCB) [Przykazanie #8]:
    U(s_t, p) = theta^T phi(s_t, p) + q_p + alpha * sqrt(phi^T A^{-1} phi) - c_fatigue - b_reject(p).
    Macierz odwrotna kowariancji A^{-1} (8x8) aktualizowana jest online w czasie O(d^2) wzorem
    Shermana-Morrisona bez bibliotek zewnętrznych i bez zadań wsadowych [Przykazanie #5].
- Funkcja nagrody R z odpowiedzi Androida [Przykazanie #9]:
    Zawsze: +1.00 | Czasami: +0.75 | Tak: +0.50 | Nie: -0.85 | Brak odpowiedzi (TTL): -0.15.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
import hashlib
import math
from typing import Any

from .const import (
    BASE_PATTERN_COOLDOWN_SECONDS,
    COND_FEATURE_DIM,
    DEFAULT_ASK_THRESHOLD,
    DEFAULT_MIN_OCCURRENCES,
    DEFAULT_SAFETY_MIN_OCCURRENCES,
    DEFAULT_UCB_ALPHA,
    FEEDBACK_ALWAYS,
    FEEDBACK_NO,
    FEEDBACK_SOMETIMES,
    FEEDBACK_YES,
    HIERARCHICAL_SHRINKAGE_KAPPA,
    MPC_DISCOUNT_GAMMA,
    MPC_HORIZON_STEPS,
    MPC_STEP_SECONDS,
    REWARD_ALWAYS,
    REWARD_EXPIRED,
    REWARD_NO,
    REWARD_SOMETIMES,
    REWARD_YES,
    RL_FEATURE_DIM,
    SEQUENCE_MIN_DELAY_SECONDS,
    SEQUENCE_WINDOW_SECONDS,
)
from .domains import ActionSpec
from .features import ContextSnapshot

WEEKDAY_KEYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKDAY_PL: tuple[str, ...] = ("pon", "wt", "śr", "czw", "pt", "sob", "niedz")
MAX_PATTERNS = 512


class PatternKind(StrEnum):
    """Typ wyuczonego wzorca zachowania domowników."""

    SEQUENTIAL = "sequential"  # Zdarzenie A -> Akcja człowieka B (np. kawa -> rolety)
    ROUTINE_TIME = "routine_time"  # Powtarzalna pora i kontekst -> Akcja B (np. brama rano)


class DecisionMode(StrEnum):
    """Tryb zadania pytania na telefonie z Androidem."""

    PROACTIVE = "proactive"  # MPC przewiduje, że za chwilę wykonasz akcję ('Tak' wykona ją teraz)
    CONFIRM_ROUTINE = "confirm_routine"  # Właśnie ręcznie wykonałeś nawyk ('Zawsze'/'Czasami' zautomatyzuje)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -40.0))
    return z / (1.0 + z)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [_dot(row, vec) for row in mat]


@dataclass(slots=True)
class OnlineConditionLearner:
    """Inkrementalny klasyfikator warunków kontekstowych dla opcji 'Czasami' [Przykazanie #5, #9].

    Łączy regresję logistyczną aktualizowaną krokiem SGD (do ciągłej oceny dopasowania kontekstu)
    ze statystykami kołowymi i histogramami cech (do syntezy natywnych warunków HA).
    """

    weights: list[float] = field(default_factory=lambda: [0.0] * COND_FEATURE_DIM)
    pos_count: int = 0
    neg_count: int = 0
    sin_minute_sum: float = 0.0
    cos_minute_sum: float = 0.0
    weekday_counts: list[int] = field(default_factory=lambda: [0] * 7)
    dark_count: int = 0
    home_count: int = 0
    arrival_count: int = 0
    calendar_count: int = 0
    lux_entity_id: str | None = None
    weather_entity_id: str | None = None
    calendar_entity_id: str | None = None
    weather_counts: dict[str, int] = field(default_factory=dict)
    person_counts: dict[str, int] = field(default_factory=dict)
    binary_sensor_counts: dict[str, int] = field(default_factory=dict)

    def update(self, snapshot: ContextSnapshot, *, positive: bool, lr: float = 0.12) -> None:
        """Natychmiastowa aktualizacja wag i statystyk na podstawie pojedynczej obserwacji."""
        x = snapshot.to_condition_vector()
        target = 1.0 if positive else 0.0
        pred = _sigmoid(_dot(self.weights, x))
        err = target - pred
        l2_reg = 0.01
        for i in range(COND_FEATURE_DIM):
            reg_term = 0.0 if i == 0 else l2_reg * self.weights[i]
            self.weights[i] += lr * (err * x[i] - reg_term)

        if not positive:
            self.neg_count += 1
            return

        self.pos_count += 1
        angle = 2.0 * math.pi * (snapshot.minute_of_day / 1440.0)
        self.sin_minute_sum += math.sin(angle)
        self.cos_minute_sum += math.cos(angle)
        self.weekday_counts[snapshot.weekday] += 1

        if snapshot.is_dark:
            self.dark_count += 1
            if snapshot.illuminance_entity_id:
                self.lux_entity_id = snapshot.illuminance_entity_id
        if snapshot.anyone_home:
            self.home_count += 1
        if snapshot.recent_arrival:
            self.arrival_count += 1
        if snapshot.calendar_busy:
            self.calendar_count += 1
            if snapshot.calendar_entity_id:
                self.calendar_entity_id = snapshot.calendar_entity_id
        if snapshot.weather_condition:
            self.weather_counts[snapshot.weather_condition] = (
                self.weather_counts.get(snapshot.weather_condition, 0) + 1
            )
            if snapshot.weather_entity_id:
                self.weather_entity_id = snapshot.weather_entity_id
        for p_ent in snapshot.present_person_entities:
            self.person_counts[p_ent] = self.person_counts.get(p_ent, 0) + 1
        for bs_ent in snapshot.active_binary_sensors:
            self.binary_sensor_counts[bs_ent] = self.binary_sensor_counts.get(bs_ent, 0) + 1

    def score_context(self, snapshot: ContextSnapshot) -> float:
        """Zwraca prawdopodobieństwo [0, 1], że bieżący kontekst pasuje do warunków wzorca."""
        if self.pos_count == 0:
            return 0.5
        if self.neg_count == 0:
            # Zanim pojawi się negatywny przykład, mierzymy zgodność z rozkładem kołowym czasu i obecnością
            mean_min, concentration = self.circular_time_stats()
            diff = abs(snapshot.minute_of_day - mean_min)
            circ_diff = min(diff, 1440.0 - diff)
            time_match = math.exp(-(circ_diff / 90.0) ** 2) if concentration > 0.5 else 0.75
            return max(0.25, min(0.98, time_match))
        return _sigmoid(_dot(self.weights, snapshot.to_condition_vector()))

    def circular_time_stats(self) -> tuple[int, float]:
        """Zwraca (średnia_minuta_doby, koncentracja_R in [0, 1])."""
        if self.pos_count == 0:
            return (480, 0.0)
        mean_sin = self.sin_minute_sum / self.pos_count
        mean_cos = self.cos_minute_sum / self.pos_count
        r_len = math.hypot(mean_sin, mean_cos)
        angle = math.atan2(mean_sin, mean_cos)
        if angle < 0:
            angle += 2.0 * math.pi
        mean_minute = int(round((angle / (2.0 * math.pi)) * 1440.0)) % 1440
        return (mean_minute, min(1.0, r_len))

    def clarity_score(self) -> float:
        """Ocenia na ile jednoznaczne są warunki kontekstowe (do wektora cech RL)."""
        if self.pos_count < 2:
            return 0.2
        _, r_time = self.circular_time_stats()
        dark_ratio = self.dark_count / self.pos_count
        dark_clarity = abs(dark_ratio - 0.5) * 2.0
        workday_sum = sum(self.weekday_counts[:5])
        wknd_sum = sum(self.weekday_counts[5:])
        day_clarity = max(workday_sum, wknd_sum) / self.pos_count
        return min(1.0, 0.45 * r_time + 0.30 * day_clarity + 0.25 * dark_clarity)

    def synthesize_ha_conditions(
        self, *, include_time_window: bool = True
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Generuje natywne warunki `condition` Home Assistanta dla akcji 'Czasami' [Przykazanie #9]."""
        if self.pos_count == 0:
            return ([], [])

        conditions: list[dict[str, Any]] = []
        descriptions: list[str] = []
        n = self.pos_count

        # 1. Okno czasowe oraz dni tygodnia
        mean_min, r_conc = self.circular_time_stats()
        workdays = sum(self.weekday_counts[:5])
        weekends = sum(self.weekday_counts[5:])
        selected_days: list[str] | None = None
        days_desc = ""

        if n >= 2 and workdays / n >= 0.85 and weekends == 0:
            selected_days = list(WEEKDAY_KEYS[:5])
            days_desc = "dni robocze"
        elif n >= 2 and weekends / n >= 0.85 and workdays == 0:
            selected_days = list(WEEKDAY_KEYS[5:])
            days_desc = "weekendy"
        elif n >= 3:
            active_idx = [i for i, c in enumerate(self.weekday_counts) if c > 0]
            if 0 < len(active_idx) <= 4:
                selected_days = [WEEKDAY_KEYS[i] for i in active_idx]
                days_desc = ", ".join(WEEKDAY_PL[i] for i in active_idx)

        time_cond: dict[str, Any] = {"condition": "time"}
        has_time_constraint = False

        if include_time_window and r_conc >= 0.65:
            # Im większa koncentracja R, tym węższe okno (od +-45 min do +-90 min)
            half_span = int(round(90 - 45 * min(1.0, (r_conc - 0.65) / 0.35)))
            start_m = (mean_min - half_span) % 1440
            end_m = (mean_min + half_span) % 1440
            after_str = f"{start_m // 60:02d}:{start_m % 60:02d}:00"
            before_str = f"{end_m // 60:02d}:{end_m % 60:02d}:00"
            time_cond["after"] = after_str
            time_cond["before"] = before_str
            has_time_constraint = True
            descriptions.append(f"godz. {after_str[:5]}–{before_str[:5]}")

        if selected_days is not None:
            time_cond["weekday"] = selected_days
            has_time_constraint = True
            descriptions.append(days_desc)

        if has_time_constraint:
            conditions.append(time_cond)

        # 2. Obecność konkretnego domownika (`person.* == home`)
        for person_ent, p_cnt in self.person_counts.items():
            if p_cnt / n >= 0.80:
                conditions.append(
                    {
                        "condition": "state",
                        "entity_id": person_ent,
                        "state": "home",
                    }
                )
                descriptions.append(f"{person_ent} w domu")
                break

        # 3. Warunek oświetlenia (po zmroku / niski poziom lux)
        if self.dark_count / n >= 0.80:
            if self.lux_entity_id:
                conditions.append(
                    {
                        "condition": "numeric_state",
                        "entity_id": self.lux_entity_id,
                        "below": 85,
                    }
                )
                descriptions.append("niskie oświetlenie (<85 lx)")
            else:
                conditions.append(
                    {
                        "condition": "state",
                        "entity_id": "sun.sun",
                        "state": "below_horizon",
                    }
                )
                descriptions.append("po zachodzie słońca")

        # 4. Warunek pogodowy (gdy wzorzec występuje spójnie w określonej pogodzie)
        if self.weather_entity_id and n >= 2:
            dominant_weather = [
                w for w, wc in self.weather_counts.items() if wc / n >= 0.80
            ]
            if dominant_weather:
                conditions.append(
                    {
                        "condition": "state",
                        "entity_id": self.weather_entity_id,
                        "state": dominant_weather[0] if len(dominant_weather) == 1 else dominant_weather,
                    }
                )
                descriptions.append(f"pogoda: {', '.join(dominant_weather)}")

        # 5. Kalendarz lub czujnik binarny
        if self.calendar_entity_id and self.calendar_count / n >= 0.80:
            conditions.append(
                {
                    "condition": "state",
                    "entity_id": self.calendar_entity_id,
                    "state": "on",
                }
            )
            descriptions.append("trwa wydarzenie w kalendarzu")

        for bs_ent, bs_cnt in self.binary_sensor_counts.items():
            if n >= 2 and bs_cnt / n >= 0.85:
                conditions.append(
                    {
                        "condition": "state",
                        "entity_id": bs_ent,
                        "state": "on",
                    }
                )
                descriptions.append(f"aktywny {bs_ent}")
                break

        return (conditions, descriptions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": [round(w, 6) for w in self.weights],
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "sin_minute_sum": round(self.sin_minute_sum, 6),
            "cos_minute_sum": round(self.cos_minute_sum, 6),
            "weekday_counts": list(self.weekday_counts),
            "dark_count": self.dark_count,
            "home_count": self.home_count,
            "arrival_count": self.arrival_count,
            "calendar_count": self.calendar_count,
            "lux_entity_id": self.lux_entity_id,
            "weather_entity_id": self.weather_entity_id,
            "calendar_entity_id": self.calendar_entity_id,
            "weather_counts": dict(self.weather_counts),
            "person_counts": dict(self.person_counts),
            "binary_sensor_counts": dict(self.binary_sensor_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineConditionLearner:
        raw_w = list(data.get("weights") or [0.0] * COND_FEATURE_DIM)
        if len(raw_w) != COND_FEATURE_DIM:
            raw_w = [0.0] * COND_FEATURE_DIM
        raw_days = list(data.get("weekday_counts") or [0] * 7)
        if len(raw_days) != 7:
            raw_days = [0] * 7
        return cls(
            weights=[float(x) for x in raw_w],
            pos_count=int(data.get("pos_count", 0)),
            neg_count=int(data.get("neg_count", 0)),
            sin_minute_sum=float(data.get("sin_minute_sum", 0.0)),
            cos_minute_sum=float(data.get("cos_minute_sum", 0.0)),
            weekday_counts=[int(c) for c in raw_days],
            dark_count=int(data.get("dark_count", 0)),
            home_count=int(data.get("home_count", 0)),
            arrival_count=int(data.get("arrival_count", 0)),
            calendar_count=int(data.get("calendar_count", 0)),
            lux_entity_id=data.get("lux_entity_id"),
            weather_entity_id=data.get("weather_entity_id"),
            calendar_entity_id=data.get("calendar_entity_id"),
            weather_counts={str(k): int(v) for k, v in (data.get("weather_counts") or {}).items()},
            person_counts={str(k): int(v) for k, v in (data.get("person_counts") or {}).items()},
            binary_sensor_counts={
                str(k): int(v) for k, v in (data.get("binary_sensor_counts") or {}).items()
            },
        )


@dataclass(slots=True)
class PatternRecord:
    """Pojedynczy kandydat na automatyzację uczony online per domownik i na poziomie domu."""

    pattern_id: str
    kind: PatternKind
    trigger_key: str  # Dla SEQUENTIAL: 'switch.ekspres::on'; dla ROUTINE_TIME: 'slot::14'
    trigger_description_pl: str
    action: ActionSpec
    household_occurrences: int = 0
    user_occurrences: dict[str, int] = field(default_factory=dict)
    seen_day_keys: list[str] = field(default_factory=list)  # Max 30 ostatnich unikalnych dat YYYY-MM-DD
    delay_mean_sec: float = 30.0
    delay_m2_sec: float = 0.0
    q_residual: float = 0.0  # Tabularna poprawka Q_p dla Hybrid LinUCB
    ask_count: int = 0
    yes_count: int = 0
    no_count: int = 0
    sometimes_count: int = 0
    last_seen_ts: float = 0.0
    last_asked_ts: float = 0.0
    cooldown_until_ts: float = 0.0
    automated: bool = False
    automation_id: str | None = None
    learning_conditions_mode: bool = False
    condition_learner: OnlineConditionLearner = field(default_factory=OnlineConditionLearner)

    @property
    def distinct_days(self) -> int:
        return len(self.seen_day_keys)

    def record_positive_observation(
        self,
        snapshot: ContextSnapshot,
        user_id: str | None,
        local_date_str: str,
        delay_sec: float | None = None,
    ) -> None:
        """Aktualizacja online O(1) po zaobserwowaniu ludzkiego wykonania wzorca."""
        self.household_occurrences += 1
        if user_id:
            self.user_occurrences[user_id] = self.user_occurrences.get(user_id, 0) + 1

        if local_date_str not in self.seen_day_keys:
            self.seen_day_keys.append(local_date_str)
            if len(self.seen_day_keys) > 30:
                self.seen_day_keys.pop(0)

        if delay_sec is not None:
            # Algorytm Welforda dla średniego opóźnienia sekwencji (np. ~45s po włączeniu kawy)
            n = self.household_occurrences
            delta = delay_sec - self.delay_mean_sec
            self.delay_mean_sec += delta / n
            delta2 = delay_sec - self.delay_mean_sec
            self.delay_m2_sec += delta * delta2

        self.last_seen_ts = snapshot.utc_timestamp
        self.condition_learner.update(snapshot, positive=True)

    def dominant_user_id(self) -> str | None:
        """Zwraca `user_id` głównego właściciela nawyku, jeśli nawyk jest wyraźnie osobisty."""
        if not self.user_occurrences:
            return None
        best_uid, best_cnt = max(self.user_occurrences.items(), key=lambda kv: kv[1])
        if best_cnt >= max(2, int(0.65 * self.household_occurrences)):
            return best_uid
        return None

    def rejection_penalty(self, now_ts: float) -> float:
        """Kara wygaszająca wzorce odrzucone przyciskiem 'Nie' (z łagodnym zanikiem w czasie)."""
        if self.no_count == 0:
            return 0.0
        days_since_ask = max(0.0, (now_ts - self.last_asked_ts) / 86400.0)
        decay = math.exp(-days_since_ask / 14.0)
        return min(1.5, 0.35 * self.no_count * decay)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "kind": self.kind.value,
            "trigger_key": self.trigger_key,
            "trigger_description_pl": self.trigger_description_pl,
            "action": self.action.to_dict(),
            "household_occurrences": self.household_occurrences,
            "user_occurrences": dict(self.user_occurrences),
            "seen_day_keys": list(self.seen_day_keys),
            "delay_mean_sec": round(self.delay_mean_sec, 3),
            "delay_m2_sec": round(self.delay_m2_sec, 3),
            "q_residual": round(self.q_residual, 5),
            "ask_count": self.ask_count,
            "yes_count": self.yes_count,
            "no_count": self.no_count,
            "sometimes_count": self.sometimes_count,
            "last_seen_ts": self.last_seen_ts,
            "last_asked_ts": self.last_asked_ts,
            "cooldown_until_ts": self.cooldown_until_ts,
            "automated": self.automated,
            "automation_id": self.automation_id,
            "learning_conditions_mode": self.learning_conditions_mode,
            "condition_learner": self.condition_learner.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternRecord:
        return cls(
            pattern_id=str(data["pattern_id"]),
            kind=PatternKind(str(data["kind"])),
            trigger_key=str(data["trigger_key"]),
            trigger_description_pl=str(data.get("trigger_description_pl", "")),
            action=ActionSpec.from_dict(data["action"]),
            household_occurrences=int(data.get("household_occurrences", 0)),
            user_occurrences={
                str(k): int(v) for k, v in (data.get("user_occurrences") or {}).items()
            },
            seen_day_keys=[str(d) for d in (data.get("seen_day_keys") or [])],
            delay_mean_sec=float(data.get("delay_mean_sec", 30.0)),
            delay_m2_sec=float(data.get("delay_m2_sec", 0.0)),
            q_residual=float(data.get("q_residual", 0.0)),
            ask_count=int(data.get("ask_count", 0)),
            yes_count=int(data.get("yes_count", 0)),
            no_count=int(data.get("no_count", 0)),
            sometimes_count=int(data.get("sometimes_count", 0)),
            last_seen_ts=float(data.get("last_seen_ts", 0.0)),
            last_asked_ts=float(data.get("last_asked_ts", 0.0)),
            cooldown_until_ts=float(data.get("cooldown_until_ts", 0.0)),
            automated=bool(data.get("automated", False)),
            automation_id=data.get("automation_id"),
            learning_conditions_mode=bool(data.get("learning_conditions_mode", False)),
            condition_learner=OnlineConditionLearner.from_dict(
                data.get("condition_learner") or {}
            ),
        )


@dataclass(frozen=True, slots=True)
class RecentTriggerEvent:
    """Element okna ślizgowego zdarzeń do wiązania sekwencji przyczynowych."""

    trigger_key: str
    entity_id: str
    description_pl: str
    timestamp: float
    user_id: str | None


@dataclass(frozen=True, slots=True)
class QuestionProposal:
    """Rekomendacja wygenerowana wspólnie przez MPC i politykę RL do wysłania na Androida."""

    pattern: PatternRecord
    mode: DecisionMode
    mpc_probability: float
    ucb_score: float
    predicted_horizon_step: int
    feature_vector: list[float]
    target_user_id: str | None
    snapshot: ContextSnapshot


class HybridLinUCBPolicy:
    """Kontekstowy bandyta Hybrid LinUCB z aktualizacją Shermana-Morrisona [Przykazanie #5, #8]."""

    def __init__(self, ucb_alpha: float = DEFAULT_UCB_ALPHA) -> None:
        self.ucb_alpha = ucb_alpha
        # Inicjalizacja A = I_d oraz A_inv = I_d
        self.a_inv: list[list[float]] = [
            [1.0 if i == j else 0.0 for j in range(RL_FEATURE_DIM)]
            for i in range(RL_FEATURE_DIM)
        ]
        # Lekki optymistyczny prior na cechy mpc_prob (idx 1) oraz support (idx 2)
        self.b_vec: list[float] = [0.05, 0.35, 0.25, 0.20, 0.10, 0.05, 0.10, 0.05]
        self.cumulative_reward: float = 0.0
        self.total_updates: int = 0

    def theta(self) -> list[float]:
        """Aktualny wektor wag polityki theta = A^{-1} b."""
        return _mat_vec(self.a_inv, self.b_vec)

    def evaluate(
        self,
        phi: list[float],
        pattern: PatternRecord,
        fatigue_cost: float,
        now_ts: float,
    ) -> float:
        """Oblicza indeks UCB U(s_t, p) łączący eksploatację wiedzy i eksplorację niepewności."""
        th = self.theta()
        mean_reward = _dot(th, phi) + pattern.q_residual
        a_inv_phi = _mat_vec(self.a_inv, phi)
        variance = max(0.0, _dot(phi, a_inv_phi))
        exploration_bonus = self.ucb_alpha * math.sqrt(variance)
        reject_pen = pattern.rejection_penalty(now_ts)
        return mean_reward + exploration_bonus - fatigue_cost - reject_pen

    def update(self, phi: list[float], pattern: PatternRecord, reward: float) -> None:
        """Aktualizacja online O(d^2) macierzy A^{-1} (Sherman-Morrison) oraz poprawki Q_p."""
        a_inv_phi = _mat_vec(self.a_inv, phi)
        denom = 1.0 + _dot(phi, a_inv_phi)
        if denom > 1e-9:
            for i in range(RL_FEATURE_DIM):
                for j in range(RL_FEATURE_DIM):
                    self.a_inv[i][j] -= (a_inv_phi[i] * a_inv_phi[j]) / denom

        for i in range(RL_FEATURE_DIM):
            self.b_vec[i] += reward * phi[i]

        # Aktualizacja rezydualnej wartości wzorca Q_p (krok TD)
        pred_linear = _dot(self.theta(), phi)
        td_err = reward - (pred_linear + pattern.q_residual)
        pattern.q_residual = max(-1.2, min(1.2, pattern.q_residual + 0.25 * td_err))

        self.cumulative_reward += reward
        self.total_updates += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_inv": [[round(v, 7) for v in row] for row in self.a_inv],
            "b_vec": [round(v, 7) for v in self.b_vec],
            "cumulative_reward": round(self.cumulative_reward, 4),
            "total_updates": self.total_updates,
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        raw_inv = data.get("a_inv")
        raw_b = data.get("b_vec")
        if (
            isinstance(raw_inv, list)
            and len(raw_inv) == RL_FEATURE_DIM
            and all(isinstance(r, list) and len(r) == RL_FEATURE_DIM for r in raw_inv)
        ):
            self.a_inv = [[float(v) for v in r] for r in raw_inv]
        if isinstance(raw_b, list) and len(raw_b) == RL_FEATURE_DIM:
            self.b_vec = [float(v) for v in raw_b]
        self.cumulative_reward = float(data.get("cumulative_reward", 0.0))
        self.total_updates = int(data.get("total_updates", 0))


class OnlineRLMPCEngine:
    """Główny silnik łączący uczenie nawyków online, planowanie MPC i politykę RL."""

    def __init__(
        self,
        *,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        ucb_alpha: float = DEFAULT_UCB_ALPHA,
        ask_threshold: float = DEFAULT_ASK_THRESHOLD,
    ) -> None:
        self.min_occurrences = max(2, min_occurrences)
        self.ask_threshold = ask_threshold
        self.policy = HybridLinUCBPolicy(ucb_alpha=ucb_alpha)

        self.patterns: dict[str, PatternRecord] = {}
        # Liczniki wystąpień wyzwalaczy (ogółem i per user) do estymacji P(akcja | wyzwalacz)
        self.trigger_counts: dict[str, int] = {}
        self.user_trigger_counts: dict[str, dict[str, int]] = {}
        # Liczniki obserwacji slotów czasowych (po jednym razie na dzień i slot)
        self.slot_day_counts: dict[int, int] = {s: 0 for s in range(48)}
        self._slot_last_date: dict[int, str] = {}
        # Rzadka macierz przejść wyzwalacz -> wyzwalacz dla wielokrokowego MPC
        self.trigger_transitions: dict[str, dict[str, int]] = {}

        # Okno ślizgowe ostatnich zdarzeń wyzwalających w pamięci RAM
        self._recent_triggers: deque[RecentTriggerEvent] = deque(maxlen=64)

    @staticmethod
    def _make_pattern_id(kind: PatternKind, trigger_key: str, action_key: str) -> str:
        raw = f"{kind.value}|{trigger_key}|{action_key}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _prune_recent_triggers(self, now_ts: float) -> None:
        while (
            self._recent_triggers
            and (now_ts - self._recent_triggers[0].timestamp) > SEQUENCE_WINDOW_SECONDS
        ):
            self._recent_triggers.popleft()

    def _record_slot_exposure(self, slot_30m: int, local_date_str: str) -> None:
        if self._slot_last_date.get(slot_30m) != local_date_str:
            self._slot_last_date[slot_30m] = local_date_str
            self.slot_day_counts[slot_30m] = self.slot_day_counts.get(slot_30m, 0) + 1

    def observe_trigger_event(
        self,
        *,
        trigger_key: str,
        entity_id: str,
        description_pl: str,
        snapshot: ContextSnapshot,
        user_id: str | None,
        local_date_str: str,
    ) -> None:
        """Rejestruje zdarzenie kontekstowe/wyzwalające (np. ekspres włączony, powrót do domu)."""
        now_ts = snapshot.utc_timestamp
        self._prune_recent_triggers(now_ts)
        self._record_slot_exposure(snapshot.slot_30m, local_date_str)

        # Aktualizacja macierzy przejść T(e_now | e_prev) dla wielokrokowego planowania MPC
        for prev in self._recent_triggers:
            if prev.entity_id != entity_id and (now_ts - prev.timestamp) >= SEQUENCE_MIN_DELAY_SECONDS:
                row = self.trigger_transitions.setdefault(prev.trigger_key, {})
                row[trigger_key] = row.get(trigger_key, 0) + 1

        self.trigger_counts[trigger_key] = self.trigger_counts.get(trigger_key, 0) + 1
        if user_id:
            u_map = self.user_trigger_counts.setdefault(user_id, {})
            u_map[trigger_key] = u_map.get(trigger_key, 0) + 1

        # Zamień poprzedni wpis tej samej encji w oknie, aby uniknąć wielokrotnego liczenia
        self._recent_triggers = deque(
            (ev for ev in self._recent_triggers if ev.entity_id != entity_id),
            maxlen=64,
        )
        self._recent_triggers.append(
            RecentTriggerEvent(
                trigger_key=trigger_key,
                entity_id=entity_id,
                description_pl=description_pl,
                timestamp=now_ts,
                user_id=user_id,
            )
        )

    def observe_human_action(
        self,
        *,
        action: ActionSpec,
        snapshot: ContextSnapshot,
        user_id: str | None,
        local_date_str: str,
    ) -> list[PatternRecord]:
        """Uczy model online wyłącznie z akcji człowieka [Przykazanie #3, #5, #6].

        Zwraca listę zaktualizowanych wzorców (sekwencyjnych i/lub rutynowych czasowych).
        """
        now_ts = snapshot.utc_timestamp
        self._prune_recent_triggers(now_ts)
        self._record_slot_exposure(snapshot.slot_30m, local_date_str)

        updated: list[PatternRecord] = []
        matched_sequential = False

        # 1. Powiązanie z niedawnymi wyzwalaczami w oknie sekwencji (np. kawa -> rolety)
        for trig in self._recent_triggers:
            if trig.entity_id == action.entity_id:
                continue
            delay = now_ts - trig.timestamp
            if delay < SEQUENCE_MIN_DELAY_SECONDS or delay > SEQUENCE_WINDOW_SECONDS:
                continue

            matched_sequential = True
            pid = self._make_pattern_id(PatternKind.SEQUENTIAL, trig.trigger_key, action.action_key)
            pat = self.patterns.get(pid)
            if pat is None:
                self._evict_if_full()
                pat = PatternRecord(
                    pattern_id=pid,
                    kind=PatternKind.SEQUENTIAL,
                    trigger_key=trig.trigger_key,
                    trigger_description_pl=trig.description_pl,
                    action=action,
                    delay_mean_sec=delay,
                )
                self.patterns[pid] = pat

            effective_uid = user_id or trig.user_id
            pat.record_positive_observation(
                snapshot, effective_uid, local_date_str, delay_sec=delay
            )
            updated.append(pat)

        # 2. Rutyna czasowo-kontekstowa (np. brama garażowa otwierana ręcznie codziennie rano)
        # Rejestrujemy zawsze w bieżącym slocie 30-minutowym (szczególnie cenne gdy brak wyzwalacza)
        slot_key = f"slot::{snapshot.slot_30m}"
        slot_h = (snapshot.slot_30m * 30) // 60
        slot_m = (snapshot.slot_30m * 30) % 60
        end_total = (snapshot.slot_30m * 30 + 30) % 1440
        slot_desc = f"codziennie ok. {slot_h:02d}:{slot_m:02d}–{end_total // 60:02d}:{end_total % 60:02d}"

        r_pid = self._make_pattern_id(PatternKind.ROUTINE_TIME, slot_key, action.action_key)
        r_pat = self.patterns.get(r_pid)
        if r_pat is None and not matched_sequential:
            self._evict_if_full()
            r_pat = PatternRecord(
                pattern_id=r_pid,
                kind=PatternKind.ROUTINE_TIME,
                trigger_key=slot_key,
                trigger_description_pl=slot_desc,
                action=action,
            )
            self.patterns[r_pid] = r_pat

        if r_pat is not None:
            r_pat.record_positive_observation(snapshot, user_id, local_date_str)
            updated.append(r_pat)

        return updated

    def _evict_if_full(self) -> None:
        """Utrzymuje stały rozmiar pamięci wzorców (usuwa najsłabszy niezautomatyzowany wzorzec)."""
        if len(self.patterns) < MAX_PATTERNS:
            return
        removable = [p for p in self.patterns.values() if not p.automated]
        if not removable:
            return
        weakest = min(
            removable,
            key=lambda p: (p.household_occurrences - 2 * p.no_count, p.last_seen_ts),
        )
        self.patterns.pop(weakest.pattern_id, None)

    def hierarchical_probability(
        self,
        pattern: PatternRecord,
        snapshot: ContextSnapshot,
        active_user_id: str | None,
    ) -> float:
        """Oblicza hierarchiczne prawdopodobieństwo nawyku P_hier(p | s, u) [Przykazanie #6]."""
        ctx_fit = pattern.condition_learner.score_context(snapshot)

        if pattern.kind == PatternKind.SEQUENTIAL:
            denom_house = max(1, self.trigger_counts.get(pattern.trigger_key, 1))
            # Wygładzanie Laplace'a Beta(1, 2) zapobiega p=1.0 po pojedynczym trafieniu
            p_house = (pattern.household_occurrences + 0.5) / (denom_house + 2.0)

            if active_user_id and active_user_id in pattern.user_occurrences:
                n_u = pattern.user_occurrences[active_user_id]
                denom_u = max(
                    n_u,
                    self.user_trigger_counts.get(active_user_id, {}).get(pattern.trigger_key, n_u),
                )
                p_user = (n_u + 0.5) / (denom_u + 1.5)
                lam = n_u / (n_u + HIERARCHICAL_SHRINKAGE_KAPPA)
                base_prob = lam * p_user + (1.0 - lam) * p_house
            else:
                base_prob = p_house
        else:
            # ROUTINE_TIME: prawdopodobieństwo wykonania w obserwowanych dniach danego okna
            slot_idx = int(pattern.trigger_key.split("::")[1])
            observed_days = max(pattern.distinct_days, self.slot_day_counts.get(slot_idx, 1))
            p_house = (pattern.distinct_days + 0.5) / (observed_days + 2.0)
            if active_user_id and active_user_id in pattern.user_occurrences:
                n_u = pattern.user_occurrences[active_user_id]
                p_user = min(0.95, (n_u + 0.5) / (observed_days + 1.5))
                lam = n_u / (n_u + HIERARCHICAL_SHRINKAGE_KAPPA)
                base_prob = lam * p_user + (1.0 - lam) * p_house
            else:
                base_prob = p_house

        return max(0.01, min(0.99, base_prob * (0.4 + 0.6 * ctx_fit)))

    def _required_occurrences(self, pattern: PatternRecord) -> int:
        if pattern.action.safety_critical:
            return max(self.min_occurrences, DEFAULT_SAFETY_MIN_OCCURRENCES)
        return self.min_occurrences

    def _build_rl_features(
        self,
        pattern: PatternRecord,
        mpc_prob: float,
        domain_priority: float,
        active_user_id: str | None,
    ) -> list[float]:
        """Buduje 8-wymiarowy wektor cech phi(s_t, p) dla polityki Hybrid LinUCB."""
        support_norm = min(1.0, math.log1p(pattern.household_occurrences) / math.log1p(12.0))
        days_norm = min(1.0, pattern.distinct_days / 5.0)
        clarity = pattern.condition_learner.clarity_score()
        dom_uid = pattern.dominant_user_id()
        user_match = (
            1.0
            if (active_user_id and dom_uid == active_user_id)
            else (0.6 if dom_uid is None else 0.3)
        )
        novelty = 1.0 / (1.0 + 0.5 * pattern.ask_count)
        phi = [
            1.0,  # 0: wyraz wolny (bias)
            mpc_prob,  # 1: wartość oczekiwana z horyzontu MPC
            support_norm,  # 2: siła dowodów (liczba obserwacji)
            days_norm,  # 3: powtarzalność między różnymi dniami
            clarity,  # 4: wyrazistość warunków kontekstowych
            user_match,  # 5: dopasowanie do aktywnego domownika
            domain_priority,  # 6: waga priorytetu domeny encji
            novelty,  # 7: świeżość pytania
        ]
        assert len(phi) == RL_FEATURE_DIM
        return phi

    def plan_next_question(
        self,
        *,
        snapshot: ContextSnapshot,
        active_user_id: str | None,
        current_trigger_key: str | None,
        just_executed_patterns: list[PatternRecord],
        entity_current_states: dict[str, str],
        domain_priority_fn: Any,
        fatigue_cost: float,
    ) -> QuestionProposal | None:
        """Planowanie MPC na horyzoncie H + wybór najlepszego pytania przez RL [Przykazanie #8].

        Obsługuje dwa komplementarne momenty:
        1) PROACTIVE (wyprzedzający): zaszedł wyzwalacz (np. kawa = on) lub zbliża się okno
           rutyny na horyzoncie H kroków, a docelowa encja nie jest jeszcze w stanie docelowym.
        2) CONFIRM_ROUTINE (reaktywny): człowiek właśnie po raz N-ty ręcznie wykonał powtarzalną
           akcję (np. otworzył rano bramę garażową bez automatyzacji) — od razu proponujemy regułę.
        """
        now_ts = snapshot.utc_timestamp
        best_proposal: QuestionProposal | None = None

        # Krok A: Ewaluacja MPC na skończonym horyzoncie H = 6 kroków dla trybu PROACTIVE
        predicted_triggers_by_step: dict[int, dict[str, float]] = {0: {}}
        if current_trigger_key:
            predicted_triggers_by_step[0][current_trigger_key] = 1.0
            # Krok h=1: rozwinięcie następników z wyuczonej macierzy przejść T(e' | e)
            trans_row = self.trigger_transitions.get(current_trigger_key)
            total_t = max(1, self.trigger_counts.get(current_trigger_key, 1))
            if trans_row:
                predicted_triggers_by_step[1] = {
                    next_trig: min(0.9, cnt / total_t)
                    for next_trig, cnt in trans_row.items()
                    if cnt >= 2
                }

        future_slots: dict[int, int] = {}
        for step in range(MPC_HORIZON_STEPS):
            future_min = (snapshot.minute_of_day + int(step * MPC_STEP_SECONDS // 60)) % 1440
            future_slots[step] = future_min // 30

        just_ids = {p.pattern_id for p in just_executed_patterns}

        for pat in self.patterns.values():
            if pat.automated or now_ts < pat.cooldown_until_ts:
                continue
            if pat.household_occurrences < self._required_occurrences(pat):
                continue
            # Dla rutyn czasowych wymagamy co najmniej 2 różnych dni (by 3 kliknięcia w 1 minutę nie były „rutyną”)
            if pat.kind == PatternKind.ROUTINE_TIME and pat.distinct_days < min(2, self.min_occurrences):
                continue

            cur_target_state = entity_current_states.get(pat.action.entity_id)

            # Scenariusz 1: Wzorzec właśnie został wykonany ręcznie przez domownika -> pytanie o automatyzację
            if pat.pattern_id in just_ids:
                mpc_prob = self.hierarchical_probability(pat, snapshot, active_user_id)
                mode = DecisionMode.CONFIRM_ROUTINE
                best_step = 0
            else:
                # Scenariusz 2: Planowanie wyprzedzające MPC po horyzoncie h = 0..H-1
                # Jeśli urządzenie już jest w stanie docelowym, nie proponujemy akcji wyprzedzającej
                if pat.action.desired_state and cur_target_state == pat.action.desired_state:
                    continue

                mpc_prob = 0.0
                best_step = 0
                if pat.kind == PatternKind.SEQUENTIAL:
                    for step, trig_map in predicted_triggers_by_step.items():
                        trig_reach_prob = trig_map.get(pat.trigger_key, 0.0)
                        if trig_reach_prob <= 0.0:
                            continue
                        step_val = (
                            (MPC_DISCOUNT_GAMMA**step)
                            * trig_reach_prob
                            * self.hierarchical_probability(pat, snapshot, active_user_id)
                        )
                        if step_val > mpc_prob:
                            mpc_prob = step_val
                            best_step = step
                else:
                    # Rutyna czasowa zbliżająca się na horyzoncie H (kroki 0..2, tj. najbliższe ~5 min)
                    pat_slot = int(pat.trigger_key.split("::")[1])
                    for step in range(min(3, MPC_HORIZON_STEPS)):
                        if future_slots[step] == pat_slot:
                            # Sprawdzamy czy zbliżamy się do wyuczonej średniej minuty
                            mean_min, r_conc = pat.condition_learner.circular_time_stats()
                            fut_min = (snapshot.minute_of_day + int(step * MPC_STEP_SECONDS // 60)) % 1440
                            diff = abs(fut_min - mean_min)
                            circ_diff = min(diff, 1440 - diff)
                            if r_conc >= 0.55 and circ_diff <= 20:
                                step_val = (MPC_DISCOUNT_GAMMA**step) * self.hierarchical_probability(
                                    pat, snapshot, active_user_id
                                )
                                if step_val > mpc_prob:
                                    mpc_prob = step_val
                                    best_step = step
                mode = DecisionMode.PROACTIVE

            if mpc_prob < 0.22:
                continue

            dom_prio = float(domain_priority_fn(pat.action.domain))
            phi = self._build_rl_features(pat, mpc_prob, dom_prio, active_user_id)
            ucb_val = self.policy.evaluate(phi, pat, fatigue_cost, now_ts)

            if ucb_val < self.ask_threshold:
                continue

            if best_proposal is None or ucb_val > best_proposal.ucb_score:
                target_uid = active_user_id or pat.dominant_user_id()
                best_proposal = QuestionProposal(
                    pattern=pat,
                    mode=mode,
                    mpc_probability=round(mpc_prob, 4),
                    ucb_score=round(ucb_val, 4),
                    predicted_horizon_step=best_step,
                    feature_vector=phi,
                    target_user_id=target_uid,
                    snapshot=snapshot,
                )

        return best_proposal

    def mark_question_sent(self, pattern: PatternRecord, now_ts: float) -> None:
        """Odnotowuje wysłanie pytania i ustawia minimalny lokalny cooldown wzorca."""
        pattern.ask_count += 1
        pattern.last_asked_ts = now_ts
        pattern.cooldown_until_ts = now_ts + BASE_PATTERN_COOLDOWN_SECONDS

    def apply_user_feedback(
        self,
        *,
        pattern_id: str,
        feedback: str,
        feature_vector: list[float],
        snapshot: ContextSnapshot,
        now_ts: float | None = None,
    ) -> tuple[PatternRecord | None, float]:
        """Natychmiastowa aktualizacja RL i modelu warunków po odpowiedzi z Androida [Przykazanie #5, #9]."""
        pat = self.patterns.get(pattern_id)
        if pat is None:
            return (None, 0.0)

        ts = now_ts if now_ts is not None else snapshot.utc_timestamp

        if feedback == FEEDBACK_ALWAYS:
            reward = REWARD_ALWAYS
            pat.yes_count += 1
            pat.automated = True
            pat.learning_conditions_mode = False
            pat.condition_learner.update(snapshot, positive=True, lr=0.18)

        elif feedback == FEEDBACK_SOMETIMES:
            reward = REWARD_SOMETIMES
            pat.sometimes_count += 1
            pat.learning_conditions_mode = True
            pat.condition_learner.update(snapshot, positive=True, lr=0.20)
            # Krótszy cooldown, jeśli nadal zbieramy próbki warunkowe
            pat.cooldown_until_ts = ts + BASE_PATTERN_COOLDOWN_SECONDS * 2.0

        elif feedback == FEEDBACK_YES:
            reward = REWARD_YES
            pat.yes_count += 1
            pat.condition_learner.update(snapshot, positive=True, lr=0.15)
            pat.cooldown_until_ts = ts + BASE_PATTERN_COOLDOWN_SECONDS * 1.5

        elif feedback == FEEDBACK_NO:
            reward = REWARD_NO
            pat.no_count += 1
            # Negatywna próbka uczy klasyfikator 'Czasami', kiedy NIE uruchamiać akcji!
            pat.condition_learner.update(snapshot, positive=False, lr=0.22)
            # Wykładniczy backoff po odmowie: 2h, 4h, 8h, 16h... (maks. 14 dni)
            backoff = min(14 * 86400.0, BASE_PATTERN_COOLDOWN_SECONDS * (2.0 ** pat.no_count))
            pat.cooldown_until_ts = ts + backoff

        else:
            # Wygaśnięcie powiadomienia bez odpowiedzi (FEEDBACK EXPIRED)
            reward = REWARD_EXPIRED
            pat.cooldown_until_ts = ts + BASE_PATTERN_COOLDOWN_SECONDS * 3.0

        self.policy.update(feature_vector, pat, reward)
        return (pat, reward)

    def revoke_pattern_automation(self, pattern_id: str, now_ts: float) -> PatternRecord | None:
        """Cofa flagę automatyzacji i wstrzymuje ponowne pytania na 24h [Przykazanie #10]."""
        pat = self.patterns.get(pattern_id)
        if pat is None:
            return None
        pat.automated = False
        pat.automation_id = None
        pat.cooldown_until_ts = now_ts + 86400.0
        return pat

    def reset_pattern(self, pattern_id: str) -> bool:
        """Usuwa wyuczony wzorzec ze słownika modelu."""
        return self.patterns.pop(pattern_id, None) is not None

    def to_dict(self) -> dict[str, Any]:
        """Serializacja całego wyuczonego modelu RL+MPC do lokalnego pliku .storage."""
        return {
            "policy": self.policy.to_dict(),
            "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
            "trigger_counts": dict(self.trigger_counts),
            "user_trigger_counts": {u: dict(m) for u, m in self.user_trigger_counts.items()},
            "slot_day_counts": {str(k): v for k, v in self.slot_day_counts.items()},
            "slot_last_date": dict(self._slot_last_date),
            "trigger_transitions": {
                k: dict(v) for k, v in self.trigger_transitions.items()
            },
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Odtworzenie stanu modelu po restarcie Home Assistanta."""
        if not data:
            return
        if "policy" in data:
            self.policy.load_from_dict(data["policy"])
        raw_patterns = data.get("patterns") or {}
        self.patterns = {
            str(pid): PatternRecord.from_dict(pdata) for pid, pdata in raw_patterns.items()
        }
        self.trigger_counts = {
            str(k): int(v) for k, v in (data.get("trigger_counts") or {}).items()
        }
        self.user_trigger_counts = {
            str(u): {str(tk): int(tv) for tk, tv in (tmap or {}).items()}
            for u, tmap in (data.get("user_trigger_counts") or {}).items()
        }
        for k, v in (data.get("slot_day_counts") or {}).items():
            try:
                self.slot_day_counts[int(k)] = int(v)
            except (TypeError, ValueError):
                continue
        self._slot_last_date = {
            int(k): str(v)
            for k, v in (data.get("slot_last_date") or {}).items()
            if str(k).isdigit()
        }
        self.trigger_transitions = {
            str(src): {str(dst): int(c) for dst, c in (dmap or {}).items()}
            for src, dmap in (data.get("trigger_transitions") or {}).items()
        }
