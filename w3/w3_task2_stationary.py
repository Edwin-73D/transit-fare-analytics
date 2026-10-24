"""
w3_task2_stationary.py - Task 2 (W3): distribución estacionaria de la cadena de transbordos.
=========================================================================================

Task de Pedro (C). Escrito durante la integración (2026-10-23) sobre la matriz del Task 1.

Pregunta: a largo plazo, ¿en qué rutas "está" el flujo de transbordos?

Teoría
------
π es una distribución sobre las rutas que no cambia al dar un paso de la cadena:
    π M = π,   sum_i π_i = 1,   π_i >= 0.
- Si la cadena tiene UNA sola clase cerrada (recurrente), π es ÚNICA.
- Si además esa clase es aperiódica, π_t = π_0 M^t converge a π desde cualquier π_0
  (iteración de potencias). El error se multiplica en cada paso por |λ2|, el segundo valor
  propio más grande (en módulo) de M.
- Los estados TRANSITORIOS tienen π = 0: con prob. 1 se salen y no vuelven, así que a largo
  plazo no queda nadie ahí. En nuestros datos son las alimentadoras.
- Si hay DOS o más clases cerradas, π no es única: el resultado depende de dónde se empiece
  (pasa con la matriz de juguete, donde la ruta sin salida B-202 es una segunda clase cerrada).

Qué significa π aquí
--------------------
π_i = fracción de los transbordos que, a largo plazo, llegan a la ruta i. Mide la posición de
la ruta en la RED de transbordos, no sus pasajeros: una alimentadora puede tener muchos
pasajeros (la gente empieza ahí sus viajes) y aun así π = 0, porque ningún transbordo llega a
ella. Por eso se compara con la fracción real de pasajeros (taps por ruta del CSV).

Decisiones (y por qué)
---------------------
1. Dos métodos que deben coincidir:
   - sistema lineal: (M^T - I) π = 0 tiene una ecuación redundante (las filas suman 0); se
     cambia la última por sum π = 1 y se resuelve. Solo funciona si π es única (si no, la
     matriz queda singular y se reporta así);
   - iteración de potencias desde el vector uniforme hasta ||π_(t+1) - π_t||_1 < 1e-12.
2. Rutas sin salida (fila de ceros): mismo tratamiento del Task 1, estado absorbente (lazo).
   Con los datos reales no hay; aparecen con --toy.
3. Además de los pasajeros se reporta la fracción de transbordos que LLEGAN a cada ruta
   (columna de la matriz de conteos): sirve para explicar π, que se parece mucho a ella.
4. Comparación de rankings con la correlación de Spearman (orden, no valores).

Uso
---
    python w3/w3_task2_stationary.py           # results/w3_matrix.json (Task 1)
    python w3/w3_task2_stationary.py --toy     # matriz de juguete: π no es única
Salida: results/w3_task2.json (solo con la matriz real)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(RAIZ, "results")
MATRIX = os.path.join(RESULTS, "w3_matrix.json")
TAPS = [os.path.join(RAIZ, "data", "taps.csv"), os.path.join(RAIZ, "data", "taps_10k.csv")]
TOL = 1e-12
MAX_ITER = 10_000


def estocastica(m: np.ndarray) -> np.ndarray:
    """Decisión 2: una fila de ceros se vuelve un lazo (estado absorbente)."""
    p = m.copy()
    for i in np.flatnonzero(p.sum(axis=1) == 0):
        p[i, i] = 1.0
    return p


def pi_lineal(p: np.ndarray) -> np.ndarray | None:
    """(M^T - I) π = 0 con la última ecuación cambiada por sum π = 1. None si π no es única."""
    n = len(p)
    a = p.T - np.eye(n)
    a[-1] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    if np.linalg.matrix_rank(a) < n:          # singular: hay más de una clase cerrada
        return None
    # los transitorios salen como ±1e-17 (ruido de redondeo): se dejan en 0 exacto
    pi = np.linalg.solve(a, b)
    return np.where(np.abs(pi) < 1e-12, 0.0, pi)


def pi_potencias(p: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """π_(t+1) = π_t M desde el uniforme hasta que el cambio (norma L1) sea < TOL."""
    pi = np.full(len(p), 1 / len(p))
    errores = []
    for _ in range(MAX_ITER):
        nuevo = pi @ p
        errores.append(float(np.abs(nuevo - pi).sum()))
        pi = nuevo
        if errores[-1] < TOL:
            break
    return pi, errores


def lambda2(p: np.ndarray) -> float:
    """|λ2|: segundo valor propio más grande en módulo (velocidad de convergencia)."""
    return float(sorted(np.abs(np.linalg.eigvals(p)), reverse=True)[1])


def pasajeros(rutas: list[str]) -> dict[str, float]:
    """Fracción de taps de cada ruta en el CSV."""
    path = next((t for t in TAPS if os.path.exists(t)), None)
    if path is None:
        sys.exit("No encuentro data/taps.csv")
    conteo = pd.read_csv(path, usecols=["route_id"])["route_id"].value_counts()
    return {r: float(conteo.get(r, 0) / conteo.sum()) for r in rutas}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="matriz de juguete (π no es única)")
    args = ap.parse_args()

    if args.toy:
        from toy_matrix import toy_matrix
        data = toy_matrix()
    else:
        if not os.path.exists(MATRIX):
            sys.exit("Falta results/w3_matrix.json: corre primero python w3/w3_task1_dtmc.py")
        with open(MATRIX, encoding="utf-8") as f:
            data = json.load(f)
    rutas = data["routes"]
    p = estocastica(np.array(data["M"], dtype=float))
    cerradas = [c["routes"] for c in data["classes"] if c["closed"]]
    transitorias = [r for c in data["classes"] if c["type"] == "transient" for r in c["routes"]]

    lin = pi_lineal(p)
    pot, errores = pi_potencias(p)
    l2 = lambda2(p)
    print(f"Matriz: {len(rutas)} rutas | clases cerradas: {len(cerradas)} | transitorias: {transitorias or 'ninguna'}")
    print(f"Iteración de potencias: {len(errores)} iteraciones, |λ2| = {l2:.4f} "
          f"(el error se multiplica por ~{l2:.3f} en cada paso)")

    if lin is None:
        print(f"\nSistema lineal singular: hay {len(cerradas)} clases cerradas {cerradas}, así que π NO es única.")
        print("(Por eso |λ2| = 1: el valor propio 1 aparece una vez por cada clase cerrada.)")
        print("La iteración de potencias desde el uniforme da UNA de ellas (depende del inicio):")
        for r, v in zip(rutas, pot):
            print(f"  {r:>6}  {v:.4f}")
        if not args.toy:
            sys.exit("La matriz real debería tener una sola clase cerrada: revisa el generador.")
        return

    dif = float(np.abs(lin - pot).max())
    print(f"Sistema lineal vs potencias: diferencia máxima {dif:.2e} | suma π = {lin.sum():.12f}")

    # Decisión 3: fracción de transbordos que llegan a cada ruta (columna de conteos)
    if "counts" in data:
        llegan = np.array(data["counts"], dtype=float).sum(axis=0)
        llegan = llegan / llegan.sum()
    else:
        llegan = np.full(len(rutas), np.nan)
    share = pasajeros(rutas)

    tabla = pd.DataFrame({"route_id": rutas, "pi": lin, "inflow_share": llegan,
                          "ridership_share": [share[r] for r in rutas]})
    tabla["rank_pi"] = tabla["pi"].rank(ascending=False, method="min").astype(int)
    tabla["rank_ridership"] = tabla["ridership_share"].rank(ascending=False, method="min").astype(int)
    tabla = tabla.sort_values("pi", ascending=False).reset_index(drop=True)
    rho = float(tabla["pi"].corr(tabla["ridership_share"], method="spearman"))
    rho_rec = float(tabla[~tabla.route_id.isin(transitorias)][["pi", "ridership_share"]]
                    .corr(method="spearman").iloc[0, 1])
    print(f"\nπ vs fracción de pasajeros (Spearman): {rho:.3f} con todas las rutas, "
          f"{rho_rec:.3f} sin las transitorias")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nTransitorias {transitorias}: π = {[round(float(lin[rutas.index(r)]), 12) for r in transitorias]} "
          f"aunque tienen {sum(share[r] for r in transitorias):.1%} de los pasajeros")

    salida = {
        "pi": {r: float(v) for r, v in zip(rutas, lin)},
        "pi_power": {r: float(v) for r, v in zip(rutas, pot)},
        "iterations": len(errores),
        "ridership_share": share,
        "inflow_share": {r: float(v) for r, v in zip(rutas, llegan)},
        "errors": errores,
        "lambda2": l2,
        "max_diff_linear_power": dif,
        "transient": transitorias,
        "closed_classes": cerradas,
        "spearman_vs_ridership": rho,
        "spearman_vs_ridership_recurrent_only": rho_rec,
        "meta": {
            "task": "W3 Task 2 - distribución estacionaria πM = π, sum π = 1",
            "source": "results/w3_matrix.json",
            "methods": "sistema lineal (M^T - I con una ecuación cambiada por sum π = 1) e iteración de potencias",
            "tol": TOL,
            "dangling_treatment": "absorbente (lazo), como el Task 1",
        },
    }
    out = os.path.join(RESULTS, "w3_task2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print("Escrito", out)


if __name__ == "__main__":
    main()
