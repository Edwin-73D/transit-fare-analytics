"""
task5_markov.py - Task 5: Desigualdad de Markov (y Chebyshev) para la capacidad en hora pico.
=============================================================================================

Pregunta: ¿qué capacidad C ponerle a una ruta en la hora pico para que casi nunca se sature?

X = taps de una ruta en la hora pico de un día. Comparamos lo que garantizan las cotas
con lo que pasó de verdad (frecuencia empírica = fracción de días con X >= C).

Teoría
------
- Markov (solo necesita la media, X >= 0):      P(X >= C) <= E[X] / C
- Chebyshev (usa también la varianza):          P(|X - mu| >= t) <= sigma^2 / t^2
  Para la cola de arriba con C > mu, t = C - mu:  P(X >= C) <= sigma^2 / (C - mu)^2
- Evaluamos C como múltiplos de la media, C = a * mu. Así la cota de Markov vale 1/a
  para CUALQUIER ruta: con C = 2 * mu, a lo sumo la mitad de los días se supera C.
- Esperado: empírico <= Chebyshev <= Markov casi siempre. Markov siempre se cumple pero
  es holgada; con más información (la varianza) Chebyshev se ajusta más.

Decisiones (y por qué)
---------------------
1. Hora pico detectada, no fija: el generador de B la elige al azar (si la fijáramos no
   tendría sentido analizar cuál es). Es la hora del día con más taps en todo el dataset.
2. Se usa la MISMA hora pico todos los días: tomar el máximo de cada día inflaría X,
   porque siempre escogeríamos el valor más alto de ese día.
3. Por ruta + todas las rutas juntas. Por ruta hay un X por día; con 7 días la frecuencia
   real solo puede ser 0/7, 1/7, ... y es muy gruesa. Juntamos las rutas con
   Y = X / media de su ruta ("veces la media"): E[Y] = 1 y Markov da P(Y >= a) <= 1/a para
   cualquier ruta, así que se pueden mezclar (25 rutas x 7 días = 175 datos).
4. Incluimos Chebyshev (opcional en el enunciado): Markov solo usa la media y es holgada;
   Chebyshev usa también la varianza y se ajusta más. Ojo: con C cerca de la media
   (p. ej. 1,5 x media) Chebyshev da 1 y es peor que Markov; ninguna cota gana siempre.
5. C como múltiplos de la media (a = 1 ... 4): la cota de Markov queda en 1/a para todas
   las rutas, fácil de leer y de comparar.
6. mu y sigma (poblacional) salen de los mismos datos: así las cotas se cumplen sobre la
   muestra; lo que se analiza es qué tan holgadas son.
7. Capacidad del Panel C = ceil(2 x media) por ruta: "el doble de la demanda promedio";
   Markov garantiza que se supera a lo sumo la mitad de los días. Además la línea queda
   en la escala de la gráfica de C (4 x media aplastaría las curvas).
8. La hora pico puede salir en cualquier hora o no existir (2026-10-18: el generador de
   Felipe reparte los taps casi parejo, 1,03x). Se reporta su fuerza con el mismo criterio del
   validador (pico / hora promedio: claro >= 1,5x, débil 1,1-1,5x, sin pico < 1,1x). Si no hay
   pico, X es "la hora más cargada", que es casi ruido; la comparación sigue siendo válida
   porque Markov y Chebyshev se cumplen para cualquier X >= 0.

Uso
---
    python src/task5_markov.py                          # data/taps.csv o data/taps_10k.csv
    python src/task5_markov.py --taps data/taps_100k.csv --capacity-factor 2
Salida: results/task5_markov.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]
DEFAULT_OUT = ROOT / "results" / "task5_markov.json"

FACTORS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]  # C = a * media (decisión 5)
CAPACITY_FACTOR = 2.0                       # capacidad que se dibuja en el Panel C (decisión 7)
PEAK_STRONG = 1.5                           # pico / hora promedio: pico claro (decisión 8)
PEAK_WEAK = 1.1                             # por debajo no hay pico (decisión 8)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_taps(path: Path) -> pd.DataFrame:
    """Lee el CSV (solo timestamp y ruta) y agrega el día y la hora de cada tap."""
    df = pd.read_csv(path, usecols=["timestamp", "route_id"])
    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["day"] = ts.dt.date.astype(str)
    df["hour"] = ts.dt.hour
    return df


def detect_peak_hour(df: pd.DataFrame) -> tuple[int, float, dict[str, int]]:
    """Hora pico = hora del día con más taps. Devuelve (hora, % de taps, hora pico de cada día)."""
    hourly = df["hour"].value_counts().reindex(range(24), fill_value=0)
    peak = int(hourly.idxmax())
    per_day = df.groupby("day")["hour"].agg(lambda h: int(h.value_counts().idxmax()))
    return peak, float(hourly[peak] / hourly.sum()), per_day.to_dict()


def peak_strength(ratio: float) -> str:
    """Fuerza del pico según cuántas veces supera a una hora promedio (decisión 8)."""
    if ratio >= PEAK_STRONG:
        return "pico claro"
    if ratio >= PEAK_WEAK:
        return "pico débil"
    return "sin pico"


def peak_counts(df: pd.DataFrame, hour: int) -> pd.DataFrame:
    """Matriz ruta x día con X = taps de la ruta en la hora pico de ese día (0 si no hubo)."""
    days = sorted(df["day"].unique())
    routes = df["route_id"].value_counts().index  # de más a menos concurrida
    in_peak = df[df["hour"] == hour]
    return in_peak.groupby(["route_id", "day"]).size().unstack(fill_value=0).reindex(
        index=routes, columns=days, fill_value=0)


# --------------------------------------------------------------------------- #
# Cotas
# --------------------------------------------------------------------------- #
def markov_bound(mu: float, c: float) -> float:
    """Markov: P(X >= C) <= mu / C (como es una probabilidad, nunca más de 1)."""
    return min(1.0, mu / c)


def chebyshev_bound(mu: float, sigma: float, c: float) -> float:
    """Chebyshev, cola de arriba: P(X >= C) <= sigma^2 / (C - mu)^2 si C > mu; si no, no informa (1)."""
    if c <= mu:
        return 1.0
    return min(1.0, sigma ** 2 / (c - mu) ** 2)


def empirical(xs: list[float], c: float) -> float:
    """Frecuencia real: fracción de observaciones con X >= C."""
    return sum(x >= c for x in xs) / len(xs)


def compare(xs: list[float], mu: float, sigma: float, factors: list[float]) -> list[dict]:
    """Para cada C = a * mu: cota de Markov, de Chebyshev, frecuencia real y si las cotas se cumplen."""
    rows = []
    for a in factors:
        c = a * mu
        mk, ch, em = markov_bound(mu, c), chebyshev_bound(mu, sigma, c), empirical(xs, c)
        rows.append({
            "factor": a,
            "C": round(c, 3),
            "markov": round(mk, 4),
            "chebyshev": round(ch, 4),
            "empirical": round(em, 4),
            "markov_holds": em <= mk + 1e-12,
            "chebyshev_holds": em <= ch + 1e-12,
        })
    return rows


# --------------------------------------------------------------------------- #
# Análisis
# --------------------------------------------------------------------------- #
def analyze_route(route: str, xs: list[int], factors: list[float]) -> dict:
    """Análisis de una ruta: media y desviación de X y la comparación para cada capacidad."""
    mu = statistics.fmean(xs)
    sigma = statistics.pstdev(xs)
    return {
        "route_id": route,
        "mean": round(mu, 3),
        "sigma": round(sigma, 3),
        "x_by_day": xs,
        "rows": compare(xs, mu, sigma, factors) if mu > 0 else [],
    }


def analyze_pooled(matrix: pd.DataFrame, factors: list[float]) -> dict:
    """Todas las rutas juntas con Y = X / media de su ruta (E[Y] = 1): más datos para comparar."""
    ys = []
    for _, row in matrix.iterrows():
        mu = row.mean()
        if mu > 0:  # una ruta sin taps en hora pico no se puede normalizar
            ys.extend(float(x / mu) for x in row)
    mu_y = statistics.fmean(ys)        # = 1 (cada ruta aporta media 1)
    sigma_y = statistics.pstdev(ys)
    return {
        "n_obs": len(ys),
        "mean_y": round(mu_y, 4),
        "var_y": round(sigma_y ** 2, 4),
        "rows": compare(ys, mu_y, sigma_y, factors),
    }


def find_taps(arg: str | None) -> Path:
    """Ruta del CSV: la que se pasa con --taps o la primera de las rutas por defecto que exista."""
    if arg:
        path = Path(arg)
        if not path.is_file():
            sys.exit(f"No encuentro el archivo de taps: {path}")
        return path
    for path in DEFAULT_TAPS_CANDIDATES:
        if path.is_file():
            return path
    sys.exit("No encuentro data/taps.csv. Genera los datos primero (python src/generator.py) o usa --taps.")


def print_table(title: str, rows: list[dict]) -> None:
    """Imprime la comparación cotas vs frecuencia real."""
    print(title)
    print(f"  {'C = a*media':>12} {'C':>8} {'Markov':>8} {'Chebyshev':>10} {'Real':>7}")
    for r in rows:
        print(f"  {r['factor']:>12} {r['C']:>8.2f} {r['markov']:>8.3f} {r['chebyshev']:>10.3f} {r['empirical']:>7.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 5: Markov y Chebyshev para la capacidad en hora pico")
    ap.add_argument("--taps", help="CSV de taps (por defecto data/taps.csv o data/taps_10k.csv)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida (por defecto results/task5_markov.json)")
    ap.add_argument("--capacity-factor", type=float, default=CAPACITY_FACTOR,
                    help=f"capacidad del dashboard = factor * media (por defecto {CAPACITY_FACTOR})")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    path = find_taps(args.taps)
    df = load_taps(path)
    peak, peak_share, per_day = detect_peak_hour(df)
    matrix = peak_counts(df, peak)
    print(f"{len(df):,} taps, {matrix.shape[0]} rutas, {matrix.shape[1]} días ({path.name})")
    peak_ratio = peak_share * 24                       # = pico / promedio por hora
    strength = peak_strength(peak_ratio)
    days_match = sum(h == peak for h in per_day.values()) / len(per_day)
    print(f"Hora pico detectada: {peak:02d}:00-{(peak + 1) % 24:02d}:00 ({peak_share * 100:.1f}% de los taps, "
          f"{peak_ratio:.2f}x una hora promedio -> {strength}; coincide en {days_match:.0%} de los días)")
    if strength != "pico claro":
        print("  Ojo: no hay un pico claro; X es la hora más cargada (casi ruido). Las cotas siguen "
              "siendo válidas porque Markov aplica a cualquier X >= 0.")
    print()

    routes = [analyze_route(r, [int(x) for x in matrix.loc[r]], FACTORS) for r in matrix.index]
    pooled = analyze_pooled(matrix, FACTORS)
    capacity = {r["route_id"]: math.ceil(args.capacity_factor * r["mean"]) for r in routes if r["mean"] > 0}

    all_rows = [row for r in routes for row in r["rows"]] + pooled["rows"]
    result = {
        "meta": {
            "task": "5 - desigualdad de Markov (y Chebyshev)",
            "source": path.name,
            "n_taps": len(df),
            "n_routes": matrix.shape[0],
            "days": list(matrix.columns),
            "peak_hour": peak,
            "peak_share": round(peak_share, 4),
            "peak_hour_by_day": per_day,
            "peak_ratio": round(peak_ratio, 3),
            "peak_strength": strength,
            "peak_days_match": round(days_match, 3),
            "X": "taps de la ruta en la hora pico detectada, un valor por día",
            "factors": FACTORS,
            "capacity_factor": args.capacity_factor,
            "markov": "P(X >= C) <= E[X]/C",
            "chebyshev": "P(X >= C) <= sigma^2/(C-mu)^2 para C > mu",
            "pooled": "Y = X / media de su ruta, todas las rutas y días juntos",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "capacity": capacity,  # taps/hora por ruta; lo lee el Panel C del dashboard
        "checks": {
            "markov_always_holds": all(r["markov_holds"] for r in all_rows),
            "chebyshev_always_holds": all(r["chebyshev_holds"] for r in all_rows),
        },
        "pooled": pooled,
        "by_route": routes,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    top = routes[0]
    print(f"Ruta más concurrida {top['route_id']}: X por día = {top['x_by_day']}, "
          f"media {top['mean']}, desviación {top['sigma']}")
    print_table(f"  Cotas vs real ({len(top['x_by_day'])} días):", top["rows"])
    print_table(f"Todas las rutas juntas (Y = X / media de su ruta, {pooled['n_obs']} datos, "
                f"Var(Y) = {pooled['var_y']}):", pooled["rows"])
    print(f"Markov se cumple siempre: {result['checks']['markov_always_holds']} | "
          f"Chebyshev se cumple siempre: {result['checks']['chebyshev_always_holds']}")
    print(f"Capacidad para el dashboard ({args.capacity_factor} x media): "
          f"{top['route_id']} = {capacity[top['route_id']]} taps/hora")
    print(f"Escrito {out_path}")


if __name__ == "__main__":
    main()
