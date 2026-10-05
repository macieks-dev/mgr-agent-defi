# Dynamiczna optymalizacja opłaty w Uniswap V4 z użyciem PPO

Repozytorium zawiera kod źródłowy i artefakty eksperymentalne wykorzystane w pracy magisterskiej poświęconej adaptacyjnemu sterowaniu opłatą LP w Uniswap V4 z użyciem uczenia przez wzmacnianie. Celem repozytorium jest odtworzenie wyników raportowanych dla wariantu V4: przygotowania danych, środowiska symulacyjnego, treningu modelu PPO, ewaluacji końcowej oraz wykresów i tabel wykorzystanych w pracy.

## Zakres repozytorium

Repozytorium obejmuje pięć obszarów:

- pipeline danych rynkowych i budowę cech wejściowych,
- środowisko RL z modelowaniem LVR i dynamiki opłat,
- trening i ewaluację wariantu V4,
- generowanie artefaktów wynikowych do rozdziałów 5-6 oraz aneksu,
- minimalny komponent on-chain potrzebny do walidacji Hooka i testów Foundry.

Repozytorium zawiera finalny checkpoint modelu V4, stan normalizacji, metadane zbioru oraz zapisane wyniki. Tablice podziałów danych (`features_*.npy` i `prices_*.npy`) trzeba wygenerować pipeline'em opisanym niżej; do odtworzenia ewaluacji nie trzeba ponownie trenować modelu.

To projekt badawczy. Kontrakt jest przeznaczony do lokalnej walidacji na Anvil; repozytorium nie deklaruje gotowości wdrożenia na sieć produkcyjną.

## Wymagania systemowe

- Python 3.10 lub nowszy,
- system Windows, Linux lub macOS,
- Docker i Docker Compose opcjonalnie,
- Foundry wymagane tylko do testów i uruchomień części Solidity,
- GPU z CUDA opcjonalne; przydatne przy pełnym treningu, niewymagane do ewaluacji i generowania wykresów.

Projekt był uruchamiany lokalnie z Pythonem 3.11. Konfiguracja zależności znajduje się w `pyproject.toml`.

## Instalacja

Jeżeli repozytorium jest pobierane z Git:

```bash
git clone <repository-url>
cd <repository-directory>
```

Jeżeli repozytorium jest przekazywane jako archiwum, wystarczy przejść do katalogu głównego projektu.

### Instalacja lokalna

Windows:

```cmd
scripts\setup.bat
```

Linux lub macOS:

```bash
bash scripts/setup.sh
```

Skrypty tworzą środowisko `.venv`, instalują zależności Python i przygotowują plik `.env` na podstawie `.env.example`.
Wdrożenie Hooka wymaga jawnego ustawienia `DEPLOYER_PRIVATE_KEY`; skrypt wdrożeniowy przyjmuje wyłącznie sieć Anvil o chain ID `31337`.

Jeżeli konfiguracja ma być wykonana ręcznie:

```bash
python -m venv .venv
```

Aktywacja środowiska:

```cmd
.venv\Scripts\activate
```

lub:

```bash
source .venv/bin/activate
```

Następnie należy zainstalować zależności:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install gymnasium "stable-baselines3[extra]" shimmy numpy pandas pyarrow web3 torch tensorboard matplotlib scipy python-dotenv requests tqdm pytest pytest-cov black isort mypy
```

Na końcu należy przygotować plik środowiskowy:

```cmd
copy .env.example .env
```

lub:

```bash
cp .env.example .env
```

## Konfiguracja eksperymentu

Najważniejsze parametry eksperymentu są zdefiniowane w dwóch miejscach:

- `src/config.py` zawiera centralną konfigurację projektu,
- `.env.example` definiuje zmienne środowiskowe i domyślne wartości uruchomieniowe.

Kluczowe założenia odtwarzalności:

- seed eksperymentu jest ustawiany przez `SEED` i domyślnie ma wartość `42`,
- podział czasowy zbiorów jest ustalony w `src/config.py`,
- finalna ewaluacja V4 korzysta z zapisanych artefaktów `models/ppo_v4_final_final.zip` oraz `models/ppo_v4_final_vecnormalize.pkl`.

Domyślny podział danych:

- trening: 2023-07-01 do 2025-03-31,
- walidacja: 2025-04-01 do 2025-09-30,
- test: 2025-10-01 do 2026-01-31.

W archiwum znajduje się `data/processed/dataset_metadata.json`, ale nie ma tablic `features_*.npy` ani `prices_*.npy`. Przed ewaluacją należy je wygenerować poleceniem `python scripts/fetch_data.py --fetch-cex -v`; skrypt pobiera dane Binance i zapisuje podziały w `data/processed/`. Po pobraniu sprawdź `data/processed/dataset_metadata.json`: jeśli `cex_source` ma wartość `none`, pipeline przygotował dane bez rynku CEX i nie odtwarza ewaluacji na rzeczywistych danych.

## Odtworzenie wyników z pracy

### Wariant rekomendowany

To jest zalecana ścieżka dla promotora, recenzenta i każdej osoby, która chce odtworzyć końcowe wyniki wykorzystane w pracy. Korzysta z dołączonego finalnego modelu V4 bez ponownego treningu. Przed uruchomieniem ewaluacji przygotuj tablice testowe zgodnie z instrukcją powyżej.

Po aktywacji środowiska należy uruchomić:

```bash
python scripts/final_evaluation_v4.py
python scripts/model_explainability.py
python poprawki/generate_all_diagrams_v4.py
python poprawki/generate_rozdzial5_figures.py
```

Skrypty `scripts/final_evaluation_v4.py` oraz `scripts/model_explainability.py` domyślnie zachowują istniejące, zaakceptowane pliki wynikowe w `results/`. Jeżeli celem jest świadome przeliczenie i nadpisanie tych artefaktów, należy użyć flagi `--overwrite-results`.

Najważniejsze artefakty wyjściowe:

- `results/v4_final_evaluation.json`,
- `results/benchmark_table.json`,
- pliki graficzne w `results/figures/`,
- pliki graficzne w `poprawki/figures/`.

### Pełny pipeline od danych do modelu

Ten wariant przygotowuje tablice danych wymagane przez ewaluację i może posłużyć do własnego retreningu. Do ewaluacji używa dołączonego modelu oraz stanu normalizacji; retrening nie jest konieczny.

Przykładowa sekwencja uruchomień:

```bash
python scripts/fetch_data.py --fetch-cex -v
pytest tests/ -v
forge test -vvv
python -m src.training.train_ppo
```

Ponowny trening może wygenerować checkpoint o innej nazwie i nie jest równoważny z artefaktami końcowymi użytymi w pracy, o ile nie zostanie jawnie podmieniony finalny model V4.

## Uruchomienie w Dockerze

Docker Compose 2.24 lub nowszy jest opcjonalny. Repozytorium zawiera `Dockerfile`, `docker-compose.yml` oraz `scripts/docker-entrypoint.sh`. Plik `.env` jest opcjonalny przy uruchomieniu; skopiuj `.env.example`, jeśli chcesz przekazać kontenerowi własne ustawienia.

Podstawowe użycie:

```bash
docker compose build
docker compose run --rm app python scripts/final_evaluation_v4.py
docker compose run --rm app python scripts/model_explainability.py
docker compose run --rm app pytest
```

## Struktura repozytorium

Najważniejsze pliki i katalogi:

### Konfiguracja i uruchomienie

- `README.md`
- `pyproject.toml`
- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- `scripts/setup.sh`
- `scripts/setup.bat`
- `scripts/docker-entrypoint.sh`
- `.gitignore`

### Konfiguracja eksperymentu

- `src/config.py`
- `.env.example`

### Pipeline danych

- `scripts/fetch_data.py`
- `src/data/cex_fetcher.py`
- `src/data/dataset_builder.py`
- `src/data/feature_engineering.py`
- `src/data/tick_loader.py`
- `data/processed/dataset_metadata.json`

### Środowisko i trening V4

- `src/env/lvr.py`
- `src/env/uniswap_v4_env.py`
- `src/training/train_ppo.py`
- `src/utils/visualization.py`

### Ewaluacja i artefakty V4

- `scripts/final_evaluation_v4.py`
- `scripts/model_explainability.py`
- `results/v4_final_evaluation.json`
- `results/benchmark_table.json`
- `models/ppo_v4_final_final.zip`
- `models/ppo_v4_final_vecnormalize.pkl`

### Testy

- `tests/test_env.py`
- `tests/test_data_pipeline.py`
- `tests/test_cex_fetcher.py`

### Część on-chain

- `contracts/src/VolatilityFeeHook.sol`
- `contracts/script/DeployHook.s.sol`
- `contracts/test/VolatilityFeeHook.t.sol`
- `foundry.toml`
- `remappings.txt`

### Diagramy i wykresy do pracy

- `poprawki/generate_all_diagrams_v4.py`
- `poprawki/generate_rozdzial5_figures.py`

## Testy

Testy Python:

```bash
pytest tests/ -v
```

Testy Solidity:

```bash
forge test -vvv
```

## Uwagi dotyczące odtwarzalności

- Repozytorium jest zorientowane na wariant końcowy V4.
- Wyniki raportowane w pracy są związane z dołączonym checkpointem modelu i zapisanym stanem normalizacji.
- Zmiana seeda, zakresów czasowych, danych wejściowych lub checkpointu modelu może prowadzić do innych wyników niż w wersji opisanej w pracy.
- Główna logika eksperymentu znajduje się w plikach Python w `src/` i `scripts/`; katalog `poprawki/` służy do generowania materiałów wykorzystanych w tekście pracy.
