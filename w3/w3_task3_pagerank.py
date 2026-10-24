"""
w3_task3_pagerank.py - Task 3 (W3): PageRank de las rutas sobre el grafo de transbordos.
=======================================================================================

Pregunta: ¿qué rutas son importantes por la red de transbordos, y no solo por sus pasajeros?

Idea: el pasajero aleatorio
---------------------------
Un pasajero va de ruta en ruta siguiendo la matriz de transbordos M. El PageRank de una ruta
es la fracción del tiempo que pasa en ella a largo plazo: una ruta es importante si le llegan
transbordos desde rutas que también son importantes.

Fórmula (teletransportación con d = 0,85)
-----------------------------------------
    PR_i = (1 - d)/n + d * sum_j PR_j * M_ji
Con prob. d el pasajero sigue un transbordo de M; con prob. 1 - d "se teletransporta" a una
ruta al azar (empieza un viaje nuevo). Una ruta sin salida (dangling, fila de ceros) reparte
su masa uniformemente entre las n rutas, así la matriz vuelve a ser estocástica.

Por qué la teletransportación garantiza un PageRank único
---------------------------------------------------------
Sin ella: una ruta sin salida atrapa y "pierde" probabilidad, una ruta transitoria queda en 0
y, si hubiera varias clases cerradas, el resultado dependería del punto de partida.
Con ella, desde cualquier ruta se llega a cualquier otra en un paso (prob. >= (1-d)/n): la
matriz tiene todas sus entradas positivas, así que es irreducible (una sola clase) y
aperiódica. Por Perron-Frobenius tiene una ÚNICA distribución estacionaria y la iteración de
potencias converge a ella desde cualquier vector inicial.
Velocidad: el error se multiplica en cada iteración por d * |lambda_2|, con lambda_2 el segundo
valor propio más grande (en módulo) de G (M con filas dangling = 1/n). Como |lambda_2| <= 1, el
peor caso es d^t; si la red "mezcla" rápido (|lambda_2| pequeño) converge mucho antes.

Por qué d = 0,85
----------------
Es el valor de Brin y Page (Google). d alto respeta más los transbordos reales pero converge
más lento (peor caso ~log(tol)/log(d) iteraciones: ~140 con 0,85, ~2.300 con 0,99) y es más sensible a
cambios del grafo; d bajo converge rápido pero todo se parece al reparto uniforme. Con 0,85 el
pasajero sigue en promedio 1/(1-d) ≈ 6,7 transbordos antes de empezar otro viaje.
Se corre también d = 0,5 y 0,95 para mostrar el compromiso con datos.

Decisiones (y por qué)
---------------------
1. Lee results/w3_matrix.json (Task 1); si no existe usa la matriz de juguete
   (w3/toy_matrix.py), para poder probar mientras llega la matriz real.
2. Iteración de potencias desde el vector uniforme hasta ||PR(t+1) - PR(t)||_1 < tol = 1e-10,
   guardando el error de cada iteración (serie de convergencia para el informe).
3. Comprobación: el mismo PageRank se obtiene resolviendo el sistema lineal
   (I - d G^T) PR = (1 - d)/n, con G = M y las filas dangling cambiadas por 1/n.
4. Pasajeros = taps por ruta del CSV (con la matriz de juguete, los de toy_matrix.py).
5. Para explicar por qué una ruta sube o baja se reporta de dónde le llega su PageRank:
   la parte de teletransportación y lo que aporta cada ruta vecina (PR_j * M_ji * d).
6. Correlación de Spearman entre PageRank y pasajeros (idea tomada de la versión de Felipe,
   2026-10-23): compara los dos RANKINGS con un solo número. 1 = mismo orden, 0 = sin relación,
   -1 = orden inverso. Se usa Spearman (rangos) y no Pearson porque interesa el orden, no los
   valores: PageRank es una fracción y los pasajeros son conteos.

Uso
---
    python w3/w3_task3_pagerank.py                 # results/w3_matrix.json o matriz de juguete
    python w3/w3_task3_pagerank.py --toy           # forzar la matriz de juguete
Salida: results/w3_task3_pagerank.json (la lee el Panel D)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from toy_matrix import toy_matrix, toy_ridership

ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "results" / "w3_matrix.json"
TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]
DEFAULT_OUT = ROOT / "results" / "w3_task3_pagerank.json"

D = 0.85
TOL = 1e-10
MAX_ITER = 10_000
D_SENSITIVITY = [0.5, 0.85, 0.95]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_matrix(force_toy: bool, path: Path = MATRIX_PATH) -> dict:
    """Matriz del Task 1 si existe; si no (o con --toy), la matriz de juguete."""
    if not force_toy and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source"] = path.name
        data.setdefault("dangling", [r for r, row in zip(data["routes"], data["M"]) if sum(row) == 0])
        return data
    return toy_matrix()


def load_ridership(routes: list[str], is_toy: bool) -> dict[str, int]:
    """Pasajeros por ruta: taps del CSV (o los de juguete). Rutas sin taps quedan en 0."""
    if is_toy:
        rides = toy_ridership()
    else:
        import pandas as pd
        path = next((p for p in TAPS_CANDIDATES if p.is_file()), None)
        if path is None:
            sys.exit("No encuentro data/taps.csv para contar pasajeros por ruta.")
        rides = pd.read_csv(path, usecols=["route_id"])["route_id"].value_counts().to_dict()
    return {r: int(rides.get(r, 0)) for r in routes}


# --------------------------------------------------------------------------- #
# PageRank
# --------------------------------------------------------------------------- #
def pagerank(m: np.ndarray, dangling: np.ndarray, d: float, tol: float = TOL,
             max_iter: int = MAX_ITER) -> tuple[np.ndarray, list[float]]:
    """Iteración de potencias: PR_i = (1-d)/n + d*(sum_j PR_j M_ji + masa dangling / n)."""
    n = len(m)
    pr = np.full(n, 1.0 / n)           # vector inicial uniforme
    errors = []
    for _ in range(max_iter):
        dangling_mass = pr[dangling].sum()               # lo que tienen las rutas sin salida
        new = (1 - d) / n + d * (pr @ m + dangling_mass / n)
        err = float(np.abs(new - pr).sum())              # norma L1 del cambio
        errors.append(err)
        pr = new
        if err < tol:
            break
    return pr, errors


def pagerank_linear(m: np.ndarray, dangling: np.ndarray, d: float) -> np.ndarray:
    """El mismo PageRank resolviendo (I - d G^T) PR = (1-d)/n, con filas dangling = 1/n."""
    n = len(m)
    g = m.copy()
    g[dangling] = 1.0 / n
    return np.linalg.solve(np.eye(n) - d * g.T, np.full(n, (1 - d) / n))


def second_eigenvalue(m: np.ndarray, dangling: np.ndarray) -> float:
    """|lambda_2| de G (M con filas dangling = 1/n): junto con d define la velocidad real."""
    g = m.copy()
    g[dangling] = 1.0 / len(m)
    mods = sorted(np.abs(np.linalg.eigvals(g)), reverse=True)
    return float(mods[1]) if len(mods) > 1 else 0.0


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Correlación de Spearman: Pearson entre los rangos de los dos valores (decisión 6)."""
    import pandas as pd
    keys = list(a)
    return float(pd.Series([a[k] for k in keys]).corr(pd.Series([b[k] for k in keys]), method="spearman"))


def ranking(values: dict[str, float]) -> list[str]:
    """Rutas de mayor a menor valor."""
    return sorted(values, key=lambda r: values[r], reverse=True)


def inbound(routes: list[str], m: np.ndarray, dangling: np.ndarray, pr: np.ndarray, d: float, i: int) -> dict:
    """De dónde sale el PageRank de la ruta i: teletransportación, masa dangling y cada vecino."""
    n = len(routes)
    parts = {"teletransportación": (1 - d) / n,
             "rutas sin salida": d * pr[dangling].sum() / n}
    for j in range(n):
        if m[j, i] > 0:
            parts[routes[j]] = d * pr[j] * m[j, i]
    return {k: round(float(v), 6) for k, v in sorted(parts.items(), key=lambda kv: kv[1], reverse=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 3 (W3): PageRank de rutas con teletransportación")
    ap.add_argument("--matrix", default=str(MATRIX_PATH), help="JSON de la matriz (por defecto results/w3_matrix.json)")
    ap.add_argument("--toy", action="store_true", help="usar la matriz de juguete aunque exista la real")
    ap.add_argument("--d", type=float, default=D, help=f"factor de amortiguación (por defecto {D})")
    ap.add_argument("--tol", type=float, default=TOL, help=f"tolerancia L1 (por defecto {TOL})")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    data = load_matrix(args.toy, Path(args.matrix))
    routes = data["routes"]
    is_toy = data["source"].startswith("toy")
    m = np.array(data["M"], dtype=float)
    dangling = np.array([r in data["dangling"] for r in routes])

    pr, errors = pagerank(m, dangling, args.d, args.tol)
    pr_lin = pagerank_linear(m, dangling, args.d)
    max_diff = float(np.abs(pr - pr_lin).max())

    # Tasa de convergencia observada: cociente entre errores seguidos (teoría: <= d).
    ratios = [errors[t + 1] / errors[t] for t in range(len(errors) - 1) if errors[t] > 0]
    rate = float(np.median(ratios)) if ratios else None
    lam2 = second_eigenvalue(m, dangling)

    pagerank_map = {r: round(float(v), 8) for r, v in zip(routes, pr)}
    rides = load_ridership(routes, is_toy)
    rank_pr, rank_rides = ranking(pagerank_map), ranking(rides)
    rho = spearman(pagerank_map, rides)
    total_rides = sum(rides.values()) or 1

    movers = []
    for r in routes:
        i = routes.index(r)
        movers.append({
            "route_id": r,
            "rank_ridership": rank_rides.index(r) + 1,
            "rank_pagerank": rank_pr.index(r) + 1,
            "change": rank_rides.index(r) - rank_pr.index(r),   # > 0 = sube en PageRank
            "ridership_share": round(rides[r] / total_rides, 4),
            "pagerank": pagerank_map[r],
            "inbound": inbound(routes, m, dangling, pr, args.d, i),
        })
    movers.sort(key=lambda x: x["change"], reverse=True)

    sensitivity = []
    for d in D_SENSITIVITY:
        p, e = pagerank(m, dangling, d, args.tol)
        vals = {r: round(float(v), 6) for r, v in zip(routes, p)}
        sensitivity.append({"d": d, "iterations": len(e),
                            "worst_case_iterations": math.ceil(math.log(args.tol) / math.log(d)),
                            "predicted_rate": round(d * lam2, 4),
                            "pagerank": vals, "rank": ranking(vals)})

    result = {
        "d": args.d,
        "tol": args.tol,
        "iterations": len(errors),
        "pagerank": pagerank_map,
        "ridership": rides,
        "rank_pagerank": rank_pr,
        "rank_ridership": rank_rides,
        "spearman_vs_ridership": round(rho, 4),
        "meta": {
            "task": "W3 Task 3 - PageRank con teletransportación",
            "source": data["source"],
            "formula": "PR_i = (1-d)/n + d * sum_j PR_j M_ji; rutas dangling reparten su masa en 1/n",
            "dangling": data["dangling"],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "convergence": {"errors_l1": errors, "median_error_ratio": rate,
                        "lambda2": round(lam2, 6), "predicted_rate": round(args.d * lam2, 6),
                        "worst_case_rate": args.d},
        "linear_check": {"max_abs_diff": max_diff, "ok": max_diff < 1e-8},
        "sum_pagerank": round(float(pr.sum()), 12),
        "movers": movers,
        "d_sensitivity": sensitivity,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Matriz: {data['source']} ({len(routes)} rutas; sin salida: {data['dangling']})")
    print(f"d = {args.d}, tol = {args.tol}: convergió en {len(errors)} iteraciones; "
          f"suma PR = {pr.sum():.12f}")
    print(f"Error baja por un factor ~{rate:.3f} por iteración; teoría d*|λ2| = "
          f"{args.d}*{lam2:.3f} = {args.d * lam2:.3f} (peor caso d = {args.d})")
    print(f"Comprobación con el sistema lineal: diferencia máx = {max_diff:.1e}")
    print(f"Spearman PageRank vs pasajeros = {rho:.3f} (1 = mismo orden)\n")
    print(f"{'ruta':>7} {'PageRank':>9} {'rank PR':>8} {'pasajeros':>10} {'rank pas.':>10} {'cambio':>7}")
    for x in sorted(movers, key=lambda x: x["rank_pagerank"]):
        arrow = f"+{x['change']}" if x["change"] > 0 else str(x["change"])
        print(f"{x['route_id']:>7} {x['pagerank']:>9.4f} {x['rank_pagerank']:>8} "
              f"{rides[x['route_id']]:>10} {x['rank_ridership']:>10} {arrow:>7}")

    up, down = movers[0], movers[-1]
    for label, x in (("Más sube", up), ("Más baja", down)):
        if x["change"] != 0:
            src = ", ".join(f"{k} {v:.3f}" for k, v in list(x["inbound"].items())[:5])  # las 5 mayores
            print(f"\n{label}: {x['route_id']} (pasajeros #{x['rank_ridership']} -> PageRank #{x['rank_pagerank']})")
            print(f"  su PageRank viene de: {src}")

    print("\nSensibilidad a d:")
    for s in sensitivity:
        print(f"  d = {s['d']:<5} {s['iterations']:>4} iteraciones (factor d*|λ2| = {s['predicted_rate']}, "
              f"peor caso ~{s['worst_case_iterations']}); ranking {s['rank']}")
    print(f"\nEscrito {out_path}")


if __name__ == "__main__":
    main()
