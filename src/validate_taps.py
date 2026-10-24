"""
validate_taps.py - Verifica que un CSV de taps cumple el esquema acordado del dataset.
=====================================================================================

Lo usamos antes de correr cualquier task: si el CSV está mal, todos los resultados
(results/*.json) y el dashboard salen mal. Solo lee el CSV, no lo modifica.

Qué revisa
----------
  FAIL (el CSV no sirve, hay que corregir el generador):
    - Columnas: exactamente las 7 del esquema.
    - Al menos 10.000 filas.
    - tap_id = "T" + 8 dígitos, sin repetidos.
    - timestamp "AAAA-MM-DD HH:MM:SS" (se acepta también con "T" en vez de espacio).
    - card_id = "C" + 6 dígitos.
    - route_id y station no vacíos.
    - fare_amount numérico y >= 0.
    - transfer_flag booleano, y consistente: transferencia <=> fare_amount == 0.0.
  WARN (cumple el esquema pero se aleja de los valores objetivo):
    - ~3.000 tarjetas distintas, 20-30 rutas, transferencias entre 5% y 40%.
  INFO: periodo simulado, popularidad de rutas (Zipf), hora pico detectada y su fuerza,
        distribución por hora.

Hora pico: NO se asume
----------------------
No hay una hora pico acordada (puede salir en cualquier hora, o no haber pico), así que
no buscamos 7-8 am: la DETECTAMOS. Se cuentan los taps por hora del día (0-23) y la hora
pico es la de más taps. Para medir qué tan fuerte es la comparamos con un reparto parejo:
si los taps se repartieran igual, cada hora tendría 1/24 ≈ 4,2% del total. Usamos el
cociente  pico / promedio por hora  (= participación del pico × 24):
  >= 1.5     -> pico claro (al menos 50% más taps que una hora promedio);
  1.1 - 1.5  -> pico débil;
  < 1.1      -> sin pico: reparto casi parejo, la "hora pico" es casi ruido.
También se reporta la hora pico de cada ruta, que es la X del Task 5 (Markov).

Decisiones (y por qué)
---------------------
1. Hora pico al azar (cambio de B, 2026-10-16): si la fijáramos no tendría sentido
   analizar cuál es. Por eso el validador la detecta y nunca asume 7-8 am.
2. Fuerza del pico: claro >= 1,5x, débil 1,1-1,5x, sin pico < 1,1x una hora promedio.
   Son umbrales simples de explicar. Se reporta como INFO: que no haya pico (o que salga
   en cualquier hora) no es un error del CSV (2026-10-18, datos del generador de Felipe:
   reparto parejo, 1,03x).
3. Aceptamos el formato que entregó B (2026-10-17): tap_id con 8 dígitos (el esquema
   original decía 7) y timestamp con espacio en vez de "T". Se cambió el esquema, no el
   generador. Si cambia otra vez, basta con ajustar TAP_ID_DIGITS.
4. Si no hay data/taps.csv se usa data/taps_10k.csv: mismo orden que los scripts de C.
5. El CSV se lee todo como texto: se valida el valor tal como está escrito, antes de que
   pandas lo convierta.
6. Transferencias entre 5% y 40% (2026-10-18): el generador sortea el % en ese rango para
   cada dataset (p. ej. 23,7% en 10k y 5,5% en 100k), así que no se exige ~20%.

Uso
---
    python src/validate_taps.py                     # valida data/taps.csv (o data/taps_10k.csv)
    python src/validate_taps.py --taps otro.csv     # valida otro archivo
Código de salida: 0 si no hay FAIL, 1 si hay algún FAIL (sirve para automatizar).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Mismo orden de búsqueda que usan los scripts de C (decisión 4).
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]

EXPECTED_COLUMNS = ["tap_id", "timestamp", "card_id", "route_id", "station", "fare_amount", "transfer_flag"]
MIN_ROWS = 10_000

# Valores objetivo del esquema (solo generan WARN, no FAIL).
TARGET_CARDS = 3_000
CARDS_TOLERANCE = 0.5            # aceptamos entre 1.500 y 4.500 tarjetas
ROUTES_RANGE = (20, 30)
TRANSFER_RANGE = (0.05, 0.40)    # el generador sortea el % en este rango (decisión 6)
PEAK_STRONG = 1.5                # pico / promedio por hora: pico claro (decisión 2)
PEAK_WEAK = 1.1                  # por debajo de esto no hay pico (decisión 2)

# Formato acordado con lo que entrega el generador de B (decisión 3). El esquema original
# decía 7 dígitos y "T" en el timestamp; aceptamos lo que llegó. Si cambia, se ajusta aquí.
TAP_ID_DIGITS = 8
TAP_ID_EXAMPLE = "T" + "0" * (TAP_ID_DIGITS - 1) + "1"

BOOL_VALUES = {"true": True, "false": False, "1": True, "0": False}


class Report:
    """Acumula los resultados de cada chequeo y los imprime con su nivel."""

    def __init__(self) -> None:
        self.fails = 0
        self.warns = 0

    def ok(self, msg: str) -> None:
        print(f"  [ OK ] {msg}")

    def fail(self, msg: str) -> None:
        self.fails += 1
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        print(f"  [WARN] {msg}")

    def info(self, msg: str) -> None:
        print(f"  [INFO] {msg}")

    def check(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        (self.ok if condition else self.fail)(ok_msg if condition else fail_msg)
        return condition


def examples(series: pd.Series, n: int = 3) -> str:
    """Primeros valores problemáticos, para que sea fácil encontrarlos en el CSV."""
    return ", ".join(repr(v) for v in series.head(n))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def check_columns(df: pd.DataFrame, r: Report) -> bool:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        r.fail(f"Faltan columnas: {missing}")
    if extra:
        r.fail(f"Columnas que no están en el esquema: {extra}")
    if not missing and not extra:
        r.ok(f"Columnas: las 7 del esquema {EXPECTED_COLUMNS}")
    # Sin las 7 columnas no tiene sentido seguir: los demás chequeos fallarían en cascada.
    return not missing


def check_formats(df: pd.DataFrame, r: Report) -> dict:
    """Chequeos fila a fila. Devuelve las columnas ya convertidas para el análisis."""
    n = len(df)
    r.check(n >= MIN_ROWS, f"Filas: {n:,} (mínimo {MIN_ROWS:,})", f"Filas: {n:,}, se piden al menos {MIN_ROWS:,}")

    # tap_id: formato y unicidad (es la llave de cada registro).
    bad = df.loc[~df["tap_id"].str.fullmatch(rf"T\d{{{TAP_ID_DIGITS}}}"), "tap_id"]
    r.check(bad.empty, f'tap_id con formato "T" + {TAP_ID_DIGITS} dígitos (p. ej. {TAP_ID_EXAMPLE})',
            f"tap_id con formato inválido en {len(bad):,} filas, p. ej. {examples(bad)}")
    dup = df.loc[df["tap_id"].duplicated(), "tap_id"]
    r.check(dup.empty, "tap_id sin repetidos", f"tap_id repetidos: {len(dup):,}, p. ej. {examples(dup)}")

    # timestamp: format="ISO8601" acepta "2025-10-05 07:42:13" (lo que entrega B) y también
    # "2025-10-05T07:42:13"; rechaza formatos ambiguos como "05/10/2025 7:42".
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    bad = df.loc[ts.isna(), "timestamp"]
    r.check(bad.empty, "timestamp con fecha y hora válidas (AAAA-MM-DD HH:MM:SS)",
            f"timestamp con formato inválido en {len(bad):,} filas, p. ej. {examples(bad)}")

    bad = df.loc[~df["card_id"].str.fullmatch(r"C\d{6}"), "card_id"]
    r.check(bad.empty, 'card_id con formato "C" + 6 dígitos',
            f"card_id con formato inválido en {len(bad):,} filas, p. ej. {examples(bad)}")

    for col in ("route_id", "station"):
        empty = int((df[col].str.strip() == "").sum())
        r.check(empty == 0, f"{col} sin valores vacíos", f"{col} vacío en {empty:,} filas")

    fare = pd.to_numeric(df["fare_amount"], errors="coerce")
    bad = df.loc[fare.isna(), "fare_amount"]
    r.check(bad.empty, "fare_amount numérico",
            f"fare_amount no numérico en {len(bad):,} filas, p. ej. {examples(bad)}")
    neg = int((fare < 0).sum())
    r.check(neg == 0, "fare_amount >= 0", f"fare_amount negativo en {neg:,} filas")

    flag = df["transfer_flag"].str.strip().str.lower().map(BOOL_VALUES)
    bad = df.loc[flag.isna(), "transfer_flag"]
    r.check(bad.empty, "transfer_flag booleano (true/false)",
            f"transfer_flag no booleano en {len(bad):,} filas, p. ej. {examples(bad)}")

    # Regla del esquema: una transferencia no se cobra (0.0) y un viaje normal sí se cobra.
    valid = fare.notna() & flag.notna()
    transfer = flag[valid].astype(bool)
    free = fare[valid] == 0.0
    inconsistent = int((transfer != free).sum())
    r.check(inconsistent == 0, "Consistencia: transferencia <=> fare_amount == 0.0",
            f"{inconsistent:,} filas inconsistentes (transferencia con cobro, o viaje normal en 0.0)")

    return {"ts": ts, "fare": fare, "flag": flag}


def check_targets(df: pd.DataFrame, parsed: dict, r: Report) -> None:
    cards = df["card_id"].nunique()
    lo, hi = TARGET_CARDS * (1 - CARDS_TOLERANCE), TARGET_CARDS * (1 + CARDS_TOLERANCE)
    (r.ok if lo <= cards <= hi else r.warn)(f"Tarjetas distintas: {cards:,} (objetivo ~{TARGET_CARDS:,})")

    routes = df["route_id"].nunique()
    (r.ok if ROUTES_RANGE[0] <= routes <= ROUTES_RANGE[1] else r.warn)(
        f"Rutas distintas: {routes} (objetivo {ROUTES_RANGE[0]}-{ROUTES_RANGE[1]})")

    share = parsed["flag"].dropna().astype(bool).mean()
    (r.ok if TRANSFER_RANGE[0] <= share <= TRANSFER_RANGE[1] else r.warn)(
        f"Transferencias: {pct(share)} (esperado entre 5% y 40%)")

    # Zipf: pocas rutas concentran muchos taps. Se reporta, no se exige un valor exacto.
    counts = df["route_id"].value_counts()
    top3 = counts.head(3)
    r.info("Rutas más usadas: " + ", ".join(f"{k} {pct(v / len(df))}" for k, v in top3.items())
           + f" | menos usada: {counts.index[-1]} {pct(counts.iloc[-1] / len(df))}")

    ts = parsed["ts"].dropna()
    if not ts.empty:
        days = ts.dt.normalize().nunique()
        r.info(f"Periodo: {ts.min()} -> {ts.max()} ({days} días con taps)")


def peak_strength(ratio: float) -> str:
    """Clasifica el pico según cuántas veces supera a una hora promedio (decisión 2)."""
    if ratio >= PEAK_STRONG:
        return "pico claro"
    if ratio >= PEAK_WEAK:
        return "pico débil"
    return "sin pico (reparto casi parejo)"


def check_peak(df: pd.DataFrame, parsed: dict, r: Report) -> None:
    ts = parsed["ts"]
    hours = ts.dropna().dt.hour
    if hours.empty:
        r.fail("No hay timestamps válidos para analizar la hora pico")
        return

    # Taps por hora del día (0-23), incluyendo las horas sin taps como 0.
    hourly = hours.value_counts().reindex(range(24), fill_value=0)
    total = int(hourly.sum())
    peak_hour = int(hourly.idxmax())
    peak_share = hourly[peak_hour] / total
    ratio = peak_share * 24  # = pico / promedio por hora

    r.info(f"Hora pico detectada: {peak_hour:02d}:00-{(peak_hour + 1) % 24:02d}:00 con "
           f"{pct(peak_share)} de los taps ({ratio:.2f}x una hora promedio) -> {peak_strength(ratio)}")

    top = hourly.sort_values(ascending=False).head(3)
    r.info("Top 3 horas: " + ", ".join(f"{h:02d}h {pct(c / total)}" for h, c in top.items()))
    r.info("Taps por hora: " + " ".join(f"{h:02d}:{pct(c / total)}" for h, c in hourly.items()))

    # ¿La hora pico es la misma todos los días? Si el generador sortea una por día, aquí se ve.
    days = ts.dropna().dt.date
    per_day = pd.DataFrame({"day": days, "hour": hours}).groupby("day")["hour"].agg(lambda h: h.value_counts().idxmax())
    if len(per_day) > 1:
        dist = per_day.value_counts()
        r.info(f"Hora pico por día ({len(per_day)} días): "
               + ", ".join(f"{h:02d}h en {c} días" for h, c in dist.head(5).items())
               + f" | días que coinciden con la global: {int((per_day == peak_hour).sum())} de {len(per_day)}")

    # Hora pico por ruta: es la X del Task 5 (taps de una ruta en su hora pico).
    by_route = pd.DataFrame({"route": df.loc[hours.index, "route_id"], "hour": hours})
    route_peak = by_route.groupby("route")["hour"].agg(lambda h: h.value_counts().idxmax())
    same = int((route_peak == peak_hour).sum())
    r.info(f"Rutas cuya hora pico coincide con la global: {same} de {len(route_peak)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Valida el CSV de taps contra el esquema del dataset")
    ap.add_argument("--taps", help="CSV a validar (por defecto data/taps.csv o data/taps_10k.csv)")
    args = ap.parse_args()
    # En Windows la consola a veces no usa UTF-8 y las tildes salen como "�".
    sys.stdout.reconfigure(encoding="utf-8")

    if args.taps:
        path = Path(args.taps)
    else:
        path = next((p for p in DEFAULT_TAPS_CANDIDATES if p.is_file()), DEFAULT_TAPS_CANDIDATES[0])
    if not path.is_file():
        print(f"No encuentro el archivo: {path}. Genera los datos primero (python src/generator.py).")
        return 1

    # Todo como texto: queremos validar el valor tal como está escrito en el CSV,
    # sin que pandas lo convierta (p. ej. "T00000001" o "False") antes de revisarlo.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    print(f"Validando {path} ...\n")

    r = Report()
    print("Esquema")
    if check_columns(df, r):
        parsed = check_formats(df, r)
        print("\nValores objetivo")
        check_targets(df, parsed, r)
        print("\nHora pico")
        check_peak(df, parsed, r)

    print(f"\nResultado: {r.fails} FAIL, {r.warns} WARN -> "
          + ("el CSV NO cumple el esquema" if r.fails else "el CSV cumple el esquema"))
    return 1 if r.fails else 0


if __name__ == "__main__":
    sys.exit(main())
