"""
task2_sampling.py - Task 2a: Reservoir sampling (Algoritmo R) por ruta.
=======================================================================

Pregunta: ¿cuál es la tarifa promedio por ruta si solo podemos guardar k taps por ruta?

Los taps llegan como un stream (en orden de tiempo) y no sabemos cuántos vendrán.
El Algoritmo R mantiene, para cada ruta, una muestra uniforme de tamaño k en una sola
pasada y con memoria O(k) por ruta. Con esa muestra estimamos el promedio y lo
comparamos contra el promedio exacto (calculado con todos los taps).

Teoría que comparamos con los resultados
----------------------------------------
- Cada tap queda en la muestra con probabilidad k/N (N = taps de la ruta).
- El promedio de la muestra es insesgado: en promedio da el valor exacto.
- Su error típico (muestreo sin reemplazo de una población finita) es
      SE = sigma / sqrt(k) * sqrt((N - k) / (N - 1))
  sigma = desviación de las tarifas de la ruta. Sin el factor de corrección el error
  baja como 1/sqrt(k): en escala log-log es una recta de pendiente -1/2.
- Si k >= N el reservorio guarda la ruta completa y el error es exactamente 0.
  Esos casos se reportan aparte, como comprobación, no como resultado.

Decisiones (y por qué)
---------------------
1. k = 100, 500, 1000 los pide el enunciado; agregamos k = 10, 25, 50.
   Lo que importa no es k sino k/N: el muestreo tiene sentido cuando N >> k. Con 10k taps
   muchas rutas tienen pocos taps (B-119 tiene 70), así que con k = 500 casi todas quedan
   completas y no hay nada que medir. Los k pequeños hacen útil el experimento incluso con
   10k, y con 6 puntos la pendiente -1/2 se ve mejor en la gráfica log-log.
2. Si k >= N el reservorio guarda la ruta completa (error 0 exacto). Se marca "full" y se
   reporta aparte, como comprobación de que el código está bien, no como resultado.
3. 30 repeticiones por k con semilla fija: un solo sorteo puede salir con suerte; 30 da un
   error promedio estable y resultados reproducibles.
4. Los taps se recorren en orden de timestamp, que es como llegarían en un stream real.
5. Promedios con math.fsum (suma exacta): así el caso k >= N da 0 exacto y no 1e-16.
6. Pendiente: correr de nuevo con el CSV de 100k cuando B lo entregue (más taps por ruta).
7. El % de transferencias lo sortea el generador por dataset (entre 5% y 40%; p. ej. 23,7% en 10k
   y 5,5% en 100k). Como las transferencias cuestan 0, los promedios exactos de 10k y 100k no
   son comparables entre sí; lo que se compara es el error del muestreo dentro de cada dataset.

Salida: results/task2_sampling.json

Uso
---
    python src/task2_sampling.py                          # data/taps.csv o data/taps_10k.csv
    python src/task2_sampling.py --taps data/taps_100k.csv --reps 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]
DEFAULT_OUT = ROOT / "results" / "task2_sampling.json"

KS = [10, 25, 50, 100, 500, 1000]  # 100/500/1000 del enunciado + 10/25/50 (decisión 1)
REPS = 30                          # repeticiones por k (decisión 3)
SEED = 42                          # semilla base: resultados reproducibles


# --------------------------------------------------------------------------- #
# Algoritmo R
# --------------------------------------------------------------------------- #
class Reservoir:
    """Muestra uniforme de tamaño k de un stream de longitud desconocida (Algoritmo R)."""

    def __init__(self, k: int, rng: random.Random) -> None:
        self.k = k
        self.rng = rng
        self.items: list[float] = []
        self.seen = 0  # cuántos elementos han pasado por el stream (la i del algoritmo)

    def add(self, x: float) -> None:
        """Procesa un elemento: los primeros k entran; el i-ésimo reemplaza a uno con prob. k/i."""
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(x)
            return
        # j uniforme en {0, ..., i-1}. P(j < k) = k/i: el nuevo entra y saca al de la posición j.
        j = self.rng.randrange(self.seen)
        if j < self.k:
            self.items[j] = x


def reservoir_sample(stream, k: int, rng: random.Random) -> list[float]:
    """Algoritmo R sobre cualquier iterable: una pasada, guarda solo k elementos."""
    r = Reservoir(k, rng)
    for x in stream:
        r.add(x)
    return r.items


def sample_by_route(taps: list[tuple[str, float]], k: int, rng: random.Random) -> dict[str, Reservoir]:
    """Una sola pasada por el stream de taps, con un reservorio independiente por ruta."""
    reservoirs: dict[str, Reservoir] = {}
    for route, fare in taps:
        if route not in reservoirs:
            reservoirs[route] = Reservoir(k, rng)
        reservoirs[route].add(fare)
    return reservoirs


# --------------------------------------------------------------------------- #
# Estadística
# --------------------------------------------------------------------------- #
def mean(xs) -> float:
    """Promedio con math.fsum: suma exacta, no depende del orden (k >= N da error 0 exacto)."""
    xs = list(xs)
    return math.fsum(xs) / len(xs)


def theory_se(sigma: float, n: int, k: int) -> float:
    """Error típico teórico del promedio de una muestra de k sin reemplazo entre n."""
    if k >= n:
        return 0.0
    return sigma / math.sqrt(k) * math.sqrt((n - k) / (n - 1))


def loglog_slope(ks: list[int], errors: list[float]) -> float | None:
    """Pendiente de la recta log(error) vs log(k) por mínimos cuadrados (esperado: -0.5)."""
    pts = [(math.log(k), math.log(e)) for k, e in zip(ks, errors) if e > 0]
    if len(pts) < 3:
        return None
    mx = statistics.fmean(p[0] for p in pts)
    my = statistics.fmean(p[1] for p in pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den


# --------------------------------------------------------------------------- #
# Experimento
# --------------------------------------------------------------------------- #
def load_stream(path: Path) -> list[tuple[str, float]]:
    """Lee el CSV y lo devuelve en orden de llegada (timestamp) como pares (ruta, tarifa)."""
    df = pd.read_csv(path, usecols=["timestamp", "route_id", "fare_amount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp", kind="stable")
    return list(zip(df["route_id"], df["fare_amount"].astype(float)))


def exact_stats(taps: list[tuple[str, float]]) -> dict[str, dict]:
    """Valores exactos por ruta (con todos los taps): N, promedio y desviación sigma."""
    fares: dict[str, list[float]] = {}
    for route, fare in taps:
        fares.setdefault(route, []).append(fare)
    return {
        route: {"n": len(xs), "mean": mean(xs), "sigma": statistics.pstdev(xs)}
        for route, xs in fares.items()
    }


def run_experiment(taps, exact, ks: list[int], reps: int) -> dict[str, dict[int, list[float]]]:
    """Para cada k y cada semilla, muestrea todas las rutas y guarda el promedio estimado."""
    estimates = {route: {k: [] for k in ks} for route in exact}
    for k in ks:
        for rep in range(reps):
            rng = random.Random(SEED + rep)  # semilla fija -> resultados reproducibles
            for route, res in sample_by_route(taps, k, rng).items():
                estimates[route][k].append(mean(res.items))
    return estimates


def summarize_route(route: str, ex: dict, est: dict[int, list[float]], ks: list[int]) -> dict:
    """Compara, para una ruta y cada k, los promedios estimados con el exacto y con la teoría."""
    rows = []
    for k in ks:
        errs = [e - ex["mean"] for e in est[k]]
        rows.append({
            "k": k,
            "full": k >= ex["n"],                       # el reservorio guardó la ruta completa
            "mean_estimate": round(statistics.fmean(est[k]), 6),
            "bias": round(statistics.fmean(errs), 6),   # ~0 si el estimador es insesgado
            "mae": round(statistics.fmean(abs(e) for e in errs), 6),
            "rmse": round(math.sqrt(statistics.fmean(e * e for e in errs)), 6),
            "theory_se": round(theory_se(ex["sigma"], ex["n"], k), 6),
        })

    # Pendiente log-log sin la corrección por población finita: rmse / sqrt((N-k)/(N-1)) ~ sigma/sqrt(k).
    sampled = [r for r in rows if not r["full"]]
    slope = loglog_slope(
        [r["k"] for r in sampled],
        [r["rmse"] / math.sqrt((ex["n"] - r["k"]) / (ex["n"] - 1)) for r in sampled],
    )
    return {
        "route_id": route,
        "n_taps": ex["n"],
        "exact_mean": round(ex["mean"], 6),
        "sigma": round(ex["sigma"], 6),
        "loglog_slope": None if slope is None else round(slope, 3),
        "results": rows,
    }


def summarize_k(routes: list[dict], ks: list[int]) -> list[dict]:
    """Resumen por k sobre todas las rutas; las rutas con k >= N se cuentan aparte."""
    out = []
    for k in ks:
        rows = [r for route in routes for r in route["results"] if r["k"] == k]
        sampled = [r for r in rows if not r["full"]]
        full = [r for r in rows if r["full"]]
        ratios = [r["rmse"] / r["theory_se"] for r in sampled if r["theory_se"] > 0]
        out.append({
            "k": k,
            "routes_sampled": len(sampled),
            "routes_full": len(full),
            "full_routes_exact": all(r["mae"] == 0 for r in full),  # comprobación: error 0
            "mean_mae": round(statistics.fmean(r["mae"] for r in sampled), 6) if sampled else None,
            "mean_rmse": round(statistics.fmean(r["rmse"] for r in sampled), 6) if sampled else None,
            "mean_theory_se": round(statistics.fmean(r["theory_se"] for r in sampled), 6) if sampled else None,
            "rmse_over_theory": round(statistics.fmean(ratios), 3) if ratios else None,  # ~1 = cuadra
            "max_abs_bias": round(max(abs(r["bias"]) for r in sampled), 6) if sampled else None,
        })
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2a: reservoir sampling de la tarifa promedio por ruta")
    ap.add_argument("--taps", help="CSV de taps (por defecto data/taps.csv o data/taps_10k.csv)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida (por defecto results/task2_sampling.json)")
    ap.add_argument("--reps", type=int, default=REPS, help=f"repeticiones por k (por defecto {REPS})")
    ap.add_argument("--ks", type=int, nargs="+", default=KS, help=f"tamaños de muestra (por defecto {KS})")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    path = find_taps(args.taps)
    taps = load_stream(path)
    exact = exact_stats(taps)
    print(f"{len(taps):,} taps, {len(exact)} rutas ({path.name}); k = {args.ks}, {args.reps} repeticiones\n")

    estimates = run_experiment(taps, exact, args.ks, args.reps)
    routes = sorted(
        (summarize_route(r, exact[r], estimates[r], args.ks) for r in exact),
        key=lambda r: r["n_taps"], reverse=True,
    )
    by_k = summarize_k(routes, args.ks)
    slopes = [r["loglog_slope"] for r in routes if r["loglog_slope"] is not None]

    result = {
        "meta": {
            "task": "2a - reservoir sampling (Algoritmo R)",
            "source": path.name,
            "n_taps": len(taps),
            "n_routes": len(exact),
            "ks": args.ks,
            "repetitions": args.reps,
            "seed": SEED,
            "stream_order": "timestamp ascendente",
            "metric": "error = promedio estimado (muestra de k) - promedio exacto de fare_amount por ruta",
            "theory": "SE = sigma/sqrt(k) * sqrt((N-k)/(N-1)); pendiente log-log esperada -0.5",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "by_k": by_k,
        "median_loglog_slope": round(statistics.median(slopes), 3) if slopes else None,
        "by_route": routes,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'k':>5} {'rutas k<N':>10} {'k>=N':>5} {'MAE':>8} {'RMSE':>8} {'SE teo':>8} {'RMSE/SE':>8}")
    for row in by_k:
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        print(f"{row['k']:>5} {row['routes_sampled']:>10} {row['routes_full']:>5} "
              f"{f(row['mean_mae'], '.4f'):>8} {f(row['mean_rmse'], '.4f'):>8} "
              f"{f(row['mean_theory_se'], '.4f'):>8} {f(row['rmse_over_theory'], '.3f'):>8}")
    print(f"\nRutas con k >= N dan error exactamente 0: {all(r['full_routes_exact'] for r in by_k)}")
    print(f"Pendiente log-log mediana (esperada -0.5): {result['median_loglog_slope']}")
    print(f"Escrito {out_path}")


if __name__ == "__main__":
    main()
