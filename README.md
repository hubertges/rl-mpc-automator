# RL+MPC Domowy Automatyzator dla Home Assistanta

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=hubertges&repository=rl-mpc-automator&category=integration)
[![GitHub release](https://img.shields.io/github/v/release/hubertges/rl-mpc-automator)](https://github.com/hubertges/rl-mpc-automator/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Lokalna integracja `custom_components` do **Home Assistanta**, która łączy **Reinforcement Learning (Hybrid LinUCB)** z **Model Predictive Control (MPC)** w trybie **online (`O(1)`)**, aby obserwować intencjonalne zachowania domowników, przewidywać nadchodzące akcje i proponować natywne automatyzacje przez aktywne powiadomienia na Androida (**Tak / Nie / Zawsze / Czasami**).

Pełna specyfikacja zasad projektowych znajduje się w pliku [PRZYKAZANIA.md](PRZYKAZANIA.md).

---

## Najważniejsze cechy

* **100% lokalnie (`local_push`)** — zero chmury, zero zewnętrznych zależności Pythona (`requirements: []`), stan wyuczonego modelu przechowywany atomowo w `.storage`.
* **Ścisłe odróżnianie ludzi od automatyzacji** — graf `ContextTracker` śledzi pełne łańcuchy `context.parent_id` oraz `context.user_id` (z odfiltrowaniem kont systemowych, `automation_triggered`, `script_started` i własnych wywołań integracji).
* **Odporność na `force_update` (MQTT / KNX)** — warstwa semantycznej deduplikacji i kwantyzacji atrybutów ciągłych odrzuca pozorne aktualizacje stanów.
* **Wsparcie wszystkich 46 domen Home Assistanta** — wspólny rejestr adapterów ze szczególnym naciskiem na `light`, `cover` (rolety, bramy garażowe), `lock`, `switch` (np. ekspres do kawy), `climate`, `media_player`, `binary_sensor`, `sensor` (lux), `device_tracker`, `calendar`, `weather` oraz `notify`.
* **Hierarchiczne uczenie online (Domownik + Dom)** — wykrywa zarówno **sekwencje zdarzeń** (np. *włączenie ekspresu → otwarcie rolet*), jak i **rutyny czasowo-kontekstowe** (np. *ręczne otwieranie bramy garażowej w dni robocze ok. 07:15*).
* **Interaktywne powiadomienia Android Companion**:
  * **Tak** — wykonuje akcję jednorazowo teraz i wzmacnia nagrodę RL (`+0.50`).
  * **Zawsze** — tworzy stałą, natywną automatyzację HA w `automations.yaml` (`+1.00`).
  * **Czasami** — uczy klasyfikator warunków kontekstowych (okno godzinowe, dni tygodnia, obecność `person.*`, natężenie światła `lx`, pogoda, kalendarz, czujniki binarne) i tworzy automatyzację warunkową (`+0.75`).
  * **Nie** — obniża nagrodę (`-0.85`), uczy klasyfikator przykładu negatywnego i nakłada wykładniczy backoff na wzorzec.
* **Pełna odwracalność** — każda utworzona automatyzacja jest widoczna w standardowym UI Home Assistanta i może zostać wycofana usługą `rl_mpc_automator.revoke_automation`.

---

## Instalacja przez HACS (zalecana)

1. Kliknij przycisk powyżej (**Open in HACS**) lub w panelu Home Assistanta przejdź do:
   **HACS → Integracje → menu `⋮` (prawy górny róg) → Niestandardowe repozytoria (Custom repositories)**.
2. Wklej adres repozytorium:
   ```text
   https://github.com/hubertges/rl-mpc-automator
   ```
   i wybierz kategorię **Integracja (Integration)**, a następnie kliknij **Dodaj**.
3. Odszukaj **RL+MPC Domowy Automatyzator**, kliknij **Pobierz (Download)** i **uruchom ponownie Home Assistanta**.
4. Przejdź do **Ustawienia → Urządzenia oraz usługi → Dodaj integrację**, wyszukaj **RL+MPC Domowy Automatyzator** i wskaż swoją domyślną usługę powiadomień `notify.mobile_app_<twoj_telefon>`.

---

## Instalacja ręczna

1. Skopiuj katalog `custom_components/rl_mpc_automator` do `<katalog_konfiguracji_HA>/custom_components/rl_mpc_automator`.
2. Zrestartuj Home Assistanta.
3. Dodaj integrację w **Ustawienia → Urządzenia oraz usługi → Dodaj integrację → RL+MPC Domowy Automatyzator**.

---

## Encje diagnostyczne i usługi

Po dodaniu integracji tworzone są 4 sensory (`local_push`):
* `sensor.rl_mpc_domowy_automatyzator_zaobserwowane_wzorce` (z listą top wzorców i ich `pattern_id` w atrybutach)
* `sensor.rl_mpc_domowy_automatyzator_utworzone_automatyzacje`
* `sensor.rl_mpc_domowy_automatyzator_skumulowana_nagroda_rl`
* `sensor.rl_mpc_domowy_automatyzator_pozostaly_budzet_pytan_dzis`

Dostępne usługi w *Narzędzia deweloperskie → Akcje*:
* `rl_mpc_automator.revoke_automation` — usuwa wygenerowaną automatyzację z `automations.yaml`, przeładowuje automatyzacje i wstrzymuje wzorzec.
* `rl_mpc_automator.reset_pattern` — całkowicie usuwa wzorzec z modelu RL+MPC.
* `rl_mpc_automator.evaluate_now` — wymusza natychmiastową ewaluację horyzontu MPC.
