"""
Task 2b - QuickSort aleatorizado vs. determinista (Grupo 8 - Smart Card Fare Analytics)
=======================================================================================

Qué hace este módulo
--------------------
1. Implementa QuickSort con dos reglas de pivote:
     * determinista : el pivote es SIEMPRE el primer elemento del subarreglo.
     * aleatorizado : el pivote se elige uniformemente al azar dentro del subarreglo.
2. Hace un benchmark (n = 1k, 5k, 10k, 50k, 100k) con dos tipos de entrada:
     * "random" : orden aleatorio.
     * "sorted" : ya ordenada de menor a mayor (el peor caso del pivote fijo).
   Mide tiempo de pared y número de comparaciones, y lo compara con la teoría.
3. Usa los DOS algoritmos para ordenar las rutas reales de data/taps.csv por
   número de pasajeros (taps) y verifica que el resultado coincide con el
   ordenamiento exacto (np.sort). Ese ranking es el de planeación de servicio.

Salida: results/panel_b.json  (lo lee el Panel B del dashboard).

Teoría que se contrasta (para n claves distintas)
-------------------------------------------------
* Aleatorizado, cualquier entrada:  E[comparaciones] = 2(n+1)·H_n - 4n  ≈ 1,39·n·log2(n)
  (se prueba con variables indicadoras: P(i y j se comparan) = 2/(j-i+1)).
* Determinista (pivote fijo) con entrada ya ordenada:  n(n-1)/2 comparaciones, O(n^2).
* Determinista con entrada aleatoria: mismo valor esperado que el aleatorizado.
* Es un algoritmo Las Vegas: la respuesta siempre es correcta, solo el tiempo es aleatorio.

Decisión de implementación (importante para la sustentación)
------------------------------------------------------------
La partición se hace con máscaras de numpy en tres grupos (menores / iguales / mayores
al pivote) y con una pila explícita en lugar de recursión. Motivos:
  - Con pivote fijo y entrada ordenada, la recursión llegaría a profundidad n = 100.000
    y Python se queda sin pila; la pila explícita no tiene ese problema.
  - Una partición en Python puro tardaría horas en el caso cuadrático con n = 100.000;
    con máscaras termina en segundos y se puede medir TODO el rango de tamaños.
El conteo de comparaciones NO depende de esto: se cuenta una comparación por elemento
y por partición (m - 1 para un subarreglo de tamaño m), que es el modelo de costo
estándar del análisis de QuickSort.

Uso
---
    python src/task2_quicksort.py                    # benchmark completo (~1-2 min)
    python src/task2_quicksort.py --quick            # prueba rápida (segundos)
    python src/task2_quicksort.py --taps data/taps.csv --out results/panel_b.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]
DEFAULT_OUT = ROOT / "results" / "panel_b.json"
DEFAULT_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
SEED = 42  # semilla fija: mismas entradas y mismos pivotes en cada corrida

ALGORITHMS = ("deterministic", "randomized")
INPUT_KINDS = ("random", "sorted")


# --------------------------------------------------------------------------- #
# 1. El algoritmo
# --------------------------------------------------------------------------- #
def quicksort(values, randomized: bool, seed: int | None = None):
    """Ordena `values` de menor a mayor. Devuelve (arreglo_ordenado, comparaciones).

    randomized=False -> pivote fijo: el primer elemento del subarreglo.
    randomized=True  -> pivote uniforme al azar en el subarreglo (generador con `seed`).
    No modifica `values`.
    """
    a = np.array(values, copy=True)
    rng = random.Random(seed)
    comparisons = 0

    stack = [(0, len(a))] if len(a) > 1 else []  # subarreglos pendientes [lo, hi)
    while stack:
        lo, hi = stack.pop()
        m = hi - lo
        seg = a[lo:hi]  # vista sobre `a`

        pivot = seg[rng.randrange(m)] if randomized else seg[0]
        comparisons += m - 1  # cada elemento (menos el pivote) se compara una vez con el pivote

        # Partición en tres grupos. `less` y `greater` son copias, así que es seguro
        # sobrescribir `a[lo:hi]` a continuación.
        less = seg[seg < pivot]
        greater = seg[seg > pivot]
        n_less, n_greater = len(less), len(greater)
        n_equal = m - n_less - n_greater  # el pivote y sus duplicados quedan en su lugar final

        a[lo : lo + n_less] = less
        a[lo + n_less : lo + n_less + n_equal] = pivot
        a[hi - n_greater : hi] = greater

        # Solo quedan por ordenar los grupos "menores" y "mayores" (si tienen > 1 elemento).
        if n_less > 1:
            stack.append((lo, lo + n_less))
        if n_greater > 1:
            stack.append((hi - n_greater, hi))

    return a, comparisons


# --------------------------------------------------------------------------- #
# 2. Teoría
# --------------------------------------------------------------------------- #
def harmonic(n: int) -> float:
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def expected_comparisons_randomized(n: int) -> float:
    """E[C(n)] = 2(n+1)H_n - 4n, claves distintas, pivote uniforme (o entrada aleatoria)."""
    return 2.0 * (n + 1) * harmonic(n) - 4.0 * n


def worst_case_comparisons(n: int) -> float:
    """Pivote fijo (primer elemento) con entrada ya ordenada: (n-1) + (n-2) + ... + 1."""
    return n * (n - 1) / 2.0


def theory_comparisons(n: int, algorithm: str, kind: str) -> float:
    if algorithm == "deterministic" and kind == "sorted":
        return worst_case_comparisons(n)
    return expected_comparisons_randomized(n)


# --------------------------------------------------------------------------- #
# 3. Benchmark
# --------------------------------------------------------------------------- #
def make_input(n: int, kind: str, rep: int) -> np.ndarray:
    """n claves distintas. 'random' cambia con `rep`; 'sorted' es siempre 0..n-1."""
    if kind == "sorted":
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng([SEED, n, rep])
    return rng.permutation(n).astype(np.int64)


def repeats_for(n: int, algorithm: str, kind: str, base: int) -> int:
    """Menos repeticiones donde cada corrida es lenta (el caso cuadrático)."""
    if algorithm == "deterministic" and kind == "sorted" and n >= 50_000:
        return 1
    if n >= 50_000:
        return min(base, 3)
    return base


def run_benchmark(sizes: list[int], base_repeats: int) -> list[dict]:
    records = []
    for n in sizes:
        for kind in INPUT_KINDS:
            for algorithm in ALGORITHMS:
                reps = repeats_for(n, algorithm, kind, base_repeats)
                times, cmps, ok = [], [], True
                for rep in range(reps):
                    data = make_input(n, kind, rep)
                    t0 = time.perf_counter()
                    out, c = quicksort(data, randomized=(algorithm == "randomized"), seed=SEED + rep)
                    times.append(time.perf_counter() - t0)
                    cmps.append(c)
                    ok = ok and bool(np.array_equal(out, np.sort(data)))  # Las Vegas: siempre correcto

                theory = theory_comparisons(n, algorithm, kind)
                mean_c = statistics.fmean(cmps)
                records.append(
                    {
                        "n": n,
                        "input": kind,
                        "algorithm": algorithm,
                        "repeats": reps,
                        "time_s_median": round(statistics.median(times), 6),
                        "time_s_min": round(min(times), 6),
                        "time_s_all": [round(t, 6) for t in times],
                        "comparisons_mean": round(mean_c, 1),
                        "comparisons_theory": round(theory, 1),
                        "comparisons_ratio": round(mean_c / theory, 4),
                        "correct": ok,
                    }
                )
                print(
                    f"  n={n:>7,}  {kind:<6}  {algorithm:<13}  "
                    f"t={records[-1]['time_s_median']:>9.4f}s  "
                    f"cmp={mean_c:>16,.0f}  teoría={theory:>16,.0f}  ok={ok}"
                )
    return records


# --------------------------------------------------------------------------- #
# 4. Ranking real de rutas (uso del algoritmo sobre los datos del escenario)
# --------------------------------------------------------------------------- #
def rank_routes(taps_path: Path) -> dict:
    """Ordena las rutas por pasajeros (= número de taps) con ambos QuickSort y compara con np.sort."""
    taps = pd.read_csv(taps_path, usecols=["route_id"])
    counts = taps["route_id"].value_counts().sort_index()  # orden "natural": por route_id
    routes = list(counts.index)
    passengers = counts.to_numpy(dtype=np.int64)
    m = len(routes)

    # Clave distinta por ruta: pasajeros * m + posición. Así no hay empates y se puede
    # recuperar la ruta a partir de la clave (idx = clave % m, pasajeros = clave // m).
    keys = passengers * m + np.arange(m, dtype=np.int64)

    exact = np.sort(keys)  # referencia exacta
    out_det, c_det = quicksort(keys, randomized=False)
    out_rnd, c_rnd = quicksort(keys, randomized=True, seed=SEED)
    verified = bool(np.array_equal(out_det, exact) and np.array_equal(out_rnd, exact))

    total = int(passengers.sum())
    ranking = []
    for rank, key in enumerate(out_rnd[::-1], start=1):  # de mayor a menor
        idx = int(key % m)
        ranking.append(
            {
                "rank": rank,
                "route_id": routes[idx],
                "passengers": int(passengers[idx]),
                "share_pct": round(100.0 * int(passengers[idx]) / total, 2),
            }
        )
    return {
        "source": str(taps_path.name),
        "metric": "pasajeros = número de taps (validaciones) de la ruta en todo el periodo",
        "n_routes": m,
        "total_passengers": total,
        "comparisons_deterministic": c_det,
        "comparisons_randomized": c_rnd,
        "verified_vs_exact": verified,
        "ranking": ranking,
    }


# --------------------------------------------------------------------------- #
# 5. Programa principal
# --------------------------------------------------------------------------- #
def find_taps(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
        sys.exit(f"No encuentro el archivo de taps: {path}")
    for cand in DEFAULT_TAPS_CANDIDATES:
        if cand.exists():
            return cand
    sys.exit(
        "No encuentro data/taps.csv. Genera los datos primero (python src/generator.py) "
        "o pasa la ruta con --taps."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2b: QuickSort aleatorizado vs determinista")
    ap.add_argument("--taps", help="CSV de taps (por defecto data/taps.csv)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida (por defecto results/panel_b.json)")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="tamaños n del benchmark")
    ap.add_argument("--repeats", type=int, default=5, help="repeticiones base por configuración")
    ap.add_argument("--quick", action="store_true", help="prueba rápida: n = 1k, 5k, 10k y 2 repeticiones")
    args = ap.parse_args()

    sizes, repeats = (DEFAULT_SIZES[:3], 2) if args.quick else (sorted(args.sizes), args.repeats)
    taps_path = find_taps(args.taps)

    print(f"Benchmark QuickSort (semilla {SEED}) - tamaños: {sizes}")
    records = run_benchmark(sizes, repeats)

    print("\nRanking real de rutas por pasajeros ...")
    route_ranking = rank_routes(taps_path)
    print(f"  {route_ranking['n_routes']} rutas, verificado contra np.sort: {route_ranking['verified_vs_exact']}")
    for r in route_ranking["ranking"][:5]:
        print(f"  #{r['rank']} {r['route_id']:<6} {r['passengers']:>6,} pasajeros ({r['share_pct']}%)")

    result = {
        "meta": {
            "task": "2b",
            "description": "QuickSort aleatorizado vs determinista (pivote fijo = primer elemento)",
            "seed": SEED,
            "sizes": sizes,
            "base_repeats": repeats,
            "input_kinds": list(INPUT_KINDS),
            "algorithms": list(ALGORITHMS),
            "cost_model": "comparaciones = suma de (m-1) por cada partición de un subarreglo de tamaño m",
            "theory": {
                "randomized_expected": "2(n+1)H_n - 4n",
                "deterministic_sorted": "n(n-1)/2",
            },
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "results": records,
        "route_ranking": route_ranking,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
