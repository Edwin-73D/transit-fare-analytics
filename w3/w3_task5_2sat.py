"""
w3_task5_2sat.py - Task 5 (W3): 2-SAT aleatorizado (Papadimitriou), variante S9 "failure probability".
=====================================================================================================

Pregunta: ¿qué decisiones de servicio tomar (subir frecuencia, asignar vehículos extra) en las
rutas para cumplir 5 restricciones de planeación a la vez?

Variables (booleanas, 2 por ruta, 4 rutas -> n = 8)
---------------------------------------------------
  freq_X = subir la frecuencia en la ruta X;   veh_X = asignar vehículos extra a la ruta X.

Restricciones (2-CNF: cada una es "literal o literal")
-----------------------------------------------------
R1, R2, R3 = las 3 rutas más importantes; R4 = la menos importante. "pi alta" = importancia
mayor que el promedio 1/n_rutas, evaluada con los datos (la condición se decide antes de
correr el algoritmo, así cada restricción queda como una cláusula de 2 literales):
  1. Si pi(R1) es alta, reforzarla:                         freq_R1 v veh_R1
  2. Si pi(R2) es alta, reforzarla:                         freq_R2 v veh_R2
  3. Presupuesto: no hay vehículos extra para las dos:       ¬veh_R1 v ¬veh_R2
  4. Transbordos: si sube la frecuencia de R1, subir R3:     ¬freq_R1 v freq_R3   ("A -> B" = ¬A v B)
  5. Conductores limitados: no subir R3 y R4 a la vez:       ¬freq_R3 v ¬freq_R4

Algoritmo de Papadimitriou
--------------------------
Asignación inicial al azar; mientras haya una cláusula falsa: elegir una al azar e invertir una
de sus 2 variables al azar. Parar al satisfacer todo o al llegar al máximo de pasos.

Por qué termina en O(n^2) pasos esperados
-----------------------------------------
Sea S una solución y X = número de variables que coinciden con S. Si una cláusula es falsa, sus
2 variables no pueden coincidir ambas con S (S la cumple), así que al menos una está mal: al
invertir una al azar, X sube con prob. >= 1/2 y baja con prob. <= 1/2. X es una caminata
aleatoria en {0, ..., n} sesgada hacia n, y su tiempo esperado para llegar a n es <= n^2.

Probabilidad de falla (S9)
--------------------------
Por Markov: P(no terminar tras t pasos) <= E[pasos]/t <= n^2/t. Con t = 2n^2 falla con prob.
<= 1/2; con k intentos independientes de 2n^2 pasos, falla con prob. <= 2^(-k).

Decisiones (y por qué)
---------------------
1. La importancia de cada ruta sale de pi (results/w3_task2.json, Task 2) si existe; si no, del
   PageRank (results/w3_task3_pagerank.json, Task 3) como valor provisional.
2. Rutas: las 3 más importantes + la menos importante, para que las restricciones mezclen
   rutas que conviene reforzar con una que no. Si una condición "pi alta" no se cumple, esa
   cláusula se omite (queda trivialmente cierta) y se reporta.
3. R = 1000 corridas con semillas distintas y un tope alto de pasos (100 n^2), para medir la
   distribución real de pasos sin cortarla.
4. Satisfacibilidad comprobada por fuerza bruta (con n = 8 son solo 2^8 = 256 asignaciones).
5. Las 5 restricciones de negocio tienen muchas soluciones y el algoritmo termina en pocos pasos.
   Para comparar con la cota se agrega una instancia de referencia con UNA sola solución
   (x1, y cadena x_i -> x_(i+1)), donde los pasos se acercan más al peor caso.
6. Ejemplo insatisfacible: las 4 combinaciones de signos sobre (freq_R1, veh_R1). El algoritmo
   nunca termina con éxito; se reporta la cota: si existiera solución, la prob. de no haberla
   encontrado tras k intentos de 2n^2 pasos sería <= 2^(-k).

Uso
---
    python w3/w3_task5_2sat.py                 # R = 1000 corridas
    python w3/w3_task5_2sat.py --runs 5000
Salida: results/w3_task5_2sat.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PI_PATH = ROOT / "results" / "w3_task2.json"
PR_PATH = ROOT / "results" / "w3_task3_pagerank.json"
DEFAULT_OUT = ROOT / "results" / "w3_task5_2sat.json"

RUNS = 1000
SEED = 42
ATTEMPTS_UNSAT = 10  # intentos de 2n^2 pasos para el ejemplo insatisfacible

# Un literal es (índice de variable, True si va sin negar); una cláusula es una pareja de literales.
Literal = tuple[int, bool]
Clause = tuple[Literal, Literal]


# --------------------------------------------------------------------------- #
# Importancia de las rutas y construcción de las restricciones
# --------------------------------------------------------------------------- #
def load_importance() -> tuple[dict[str, float], str]:
    """Importancia por ruta: pi del Task 2 si existe; si no, PageRank del Task 3 (provisional)."""
    if PI_PATH.is_file():
        return json.loads(PI_PATH.read_text(encoding="utf-8"))["pi"], "pi (w3_task2.json)"
    if PR_PATH.is_file():
        return json.loads(PR_PATH.read_text(encoding="utf-8"))["pagerank"], "PageRank provisional (w3_task3_pagerank.json)"
    sys.exit("No encuentro ni results/w3_task2.json ni results/w3_task3_pagerank.json. "
             "Corre primero python w3/w3_task3_pagerank.py")


def build_instance(importance: dict[str, float]) -> dict:
    """Elige las 4 rutas, crea las 8 variables y arma las 5 cláusulas (omitiendo las de condición falsa)."""
    ranked = sorted(importance, key=importance.get, reverse=True)
    r1, r2, r3, r4 = ranked[0], ranked[1], ranked[2], ranked[-1]
    threshold = 1 / len(importance)          # "pi alta" = mayor que el promedio
    routes = [r1, r2, r3, r4]
    variables = [f"{kind}_{r}" for r in routes for kind in ("freq", "veh")]
    v = {name: i for i, name in enumerate(variables)}

    def lit(name: str, positive: bool = True) -> Literal:
        return (v[name], positive)

    candidates = [
        (importance[r1] > threshold, f"Si pi({r1}) es alta, reforzarla",
         (lit(f"freq_{r1}"), lit(f"veh_{r1}"))),
        (importance[r2] > threshold, f"Si pi({r2}) es alta, reforzarla",
         (lit(f"freq_{r2}"), lit(f"veh_{r2}"))),
        (True, f"Presupuesto: no hay vehículos extra para {r1} y {r2} a la vez",
         (lit(f"veh_{r1}", False), lit(f"veh_{r2}", False))),
        (True, f"Transbordos: si sube la frecuencia de {r1}, subir la de {r3}",
         (lit(f"freq_{r1}", False), lit(f"freq_{r3}"))),
        (True, f"Conductores limitados: no subir la frecuencia de {r3} y {r4} a la vez",
         (lit(f"freq_{r3}", False), lit(f"freq_{r4}", False))),
    ]
    clauses, descriptions, omitted = [], [], []
    for active, text, clause in candidates:
        if active:
            clauses.append(clause)
            descriptions.append(text)
        else:
            omitted.append(text)  # condición "pi alta" falsa: la cláusula queda trivialmente cierta
    return {"routes": {"R1": r1, "R2": r2, "R3": r3, "R4": r4}, "threshold": threshold,
            "variables": variables, "clauses": clauses, "descriptions": descriptions, "omitted": omitted}


def unsat_instance(var_a: str, var_b: str) -> tuple[list[str], list[Clause]]:
    """Las 4 combinaciones de signos sobre 2 variables: ninguna asignación las cumple todas."""
    return [var_a, var_b], [((0, True), (1, True)), ((0, True), (1, False)),
                            ((0, False), (1, True)), ((0, False), (1, False))]


def chain_instance(n: int) -> tuple[list[str], list[Clause]]:
    """Instancia de referencia con UNA sola solución (todo verdadero): x1, y x_i -> x_(i+1)."""
    variables = [f"x{i + 1}" for i in range(n)]
    clauses = [((0, True), (0, True))] + [((i, False), (i + 1, True)) for i in range(n - 1)]
    return variables, clauses


# --------------------------------------------------------------------------- #
# Algoritmo
# --------------------------------------------------------------------------- #
def satisfied(clause: Clause, assign: list[bool]) -> bool:
    """Una cláusula se cumple si al menos uno de sus 2 literales es verdadero."""
    return any(assign[i] == positive for i, positive in clause)


def papadimitriou(n: int, clauses: list[Clause], max_steps: int, rng: random.Random) -> tuple[bool, int, list[bool]]:
    """2-SAT aleatorizado. Devuelve (encontró solución, pasos = variables invertidas, asignación)."""
    assign = [rng.random() < 0.5 for _ in range(n)]       # asignación inicial al azar
    for step in range(max_steps + 1):
        false_clauses = [c for c in clauses if not satisfied(c, assign)]
        if not false_clauses:
            return True, step, assign
        if step == max_steps:
            break
        i, _ = rng.choice(rng.choice(false_clauses))      # cláusula falsa al azar, y una de sus 2 variables
        assign[i] = not assign[i]
    return False, max_steps, assign


def brute_force(n: int, clauses: list[Clause]) -> list[list[bool]]:
    """Todas las asignaciones que cumplen las cláusulas (con n pequeño se pueden probar las 2^n)."""
    return [list(a) for a in itertools.product([False, True], repeat=n)
            if all(satisfied(c, list(a)) for c in clauses)]


def experiment(n: int, clauses: list[Clause], runs: int) -> dict:
    """R corridas con semillas distintas: pasos hasta satisfacer y prob. empírica de no terminar tras t pasos."""
    cap = 100 * n * n
    steps = []
    for r in range(runs):
        ok, s, _ = papadimitriou(n, clauses, cap, random.Random(SEED + r))
        steps.append(s if ok else cap)
    n2 = n * n
    ts = sorted({1, 2, 4, 8, n2 // 4, n2 // 2, n2, 2 * n2, 4 * n2})
    failure = [{"t": t, "empirical": round(sum(s > t for s in steps) / runs, 4),
                "bound": round(min(1.0, n2 / t), 4)} for t in ts]
    return {"n_vars": n, "runs": runs, "steps_mean": round(statistics.fmean(steps), 3),
            "steps_median": statistics.median(steps), "steps_max": max(steps),
            "bound_n2": n2, "failure_prob": failure, "steps": steps}


def fmt_clause(clause: Clause, variables: list[str]) -> str:
    """Cláusula legible, p. ej. '¬veh_L-A ∨ ¬veh_L-B'."""
    return " ∨ ".join(("" if pos else "¬") + variables[i] for i, pos in clause)


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 5 (W3): 2-SAT aleatorizado de Papadimitriou")
    ap.add_argument("--runs", type=int, default=RUNS, help=f"corridas (por defecto {RUNS})")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    importance, source = load_importance()
    inst = build_instance(importance)
    variables, clauses = inst["variables"], inst["clauses"]
    n = len(variables)

    solutions = brute_force(n, clauses)
    main_exp = experiment(n, clauses, args.runs)
    _, _, example = papadimitriou(n, clauses, 100 * n * n, random.Random(SEED))

    chain_vars, chain_clauses = chain_instance(n)
    chain_exp = experiment(n, chain_clauses, args.runs)

    # Ejemplo insatisfacible: k intentos de 2n^2 pasos, ninguno puede terminar con éxito.
    u_vars, u_clauses = unsat_instance(f"freq_{inst['routes']['R1']}", f"veh_{inst['routes']['R1']}")
    budget = 2 * n * n
    attempts = [papadimitriou(len(u_vars), u_clauses, budget, random.Random(SEED + k))[0]
                for k in range(ATTEMPTS_UNSAT)]
    unsat = {
        "variables": u_vars,
        "clauses": [fmt_clause(c, u_vars) for c in u_clauses],
        "meaning": "reforzar y no reforzar a la vez: ninguna asignación cumple las 4",
        "satisfiable_bruteforce": bool(brute_force(len(u_vars), u_clauses)),
        "attempts": ATTEMPTS_UNSAT,
        "steps_per_attempt": budget,
        "found_solution": any(attempts),
        "report": (f"Sin solución tras {ATTEMPTS_UNSAT} intentos de {budget} pasos. Si existiera "
                   f"solución, la prob. de no haberla encontrado sería <= 2^-{ATTEMPTS_UNSAT} "
                   f"= {2 ** -ATTEMPTS_UNSAT:.1e}: se concluye 'probablemente insatisfacible'."),
    }

    result = {
        "n_vars": n,
        "variables": variables,
        "clauses": [{"clause": fmt_clause(c, variables), "meaning": d}
                    for c, d in zip(clauses, inst["descriptions"])],
        "runs": args.runs,
        "steps_mean": main_exp["steps_mean"],
        "steps_median": main_exp["steps_median"],
        "steps_max": main_exp["steps_max"],
        "bound_n2": main_exp["bound_n2"],
        "failure_prob": main_exp["failure_prob"],
        "unsat_example": unsat,
        "meta": {
            "task": "W3 Task 5 - 2-SAT aleatorizado (Papadimitriou), variante S9",
            "importance_source": source,
            "routes": inst["routes"],
            "high_threshold": round(inst["threshold"], 4),
            "omitted_clauses": inst["omitted"],
            "seed": SEED,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "satisfiable_bruteforce": bool(solutions),
        "n_solutions": len(solutions),
        "solution_example": {name: val for name, val in zip(variables, example)},
        "reference_worst_case": {
            "description": "una sola solución: x1 y la cadena x_i -> x_(i+1)",
            "clauses": [fmt_clause(c, chain_vars) for c in chain_clauses],
            **{k: v for k, v in chain_exp.items() if k != "steps"},
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Importancia: {source}; 'pi alta' = mayor que 1/{len(importance)} = {inst['threshold']:.3f}")
    print(f"Rutas: {inst['routes']}  ->  n = {n} variables\n")
    print("Restricciones:")
    for c, d in zip(clauses, inst["descriptions"]):
        print(f"  {fmt_clause(c, variables):<32} {d}")
    if inst["omitted"]:
        print(f"  (omitidas por condición falsa: {inst['omitted']})")
    print(f"\nFuerza bruta: satisfacible = {bool(solutions)} ({len(solutions)} de {2 ** n} asignaciones)")
    print("Ejemplo de solución encontrada: " + ", ".join(f"{k}={'sí' if v else 'no'}" for k, v in result["solution_example"].items()))

    for label, e in (("Restricciones de negocio", main_exp), ("Referencia peor caso (1 solución)", chain_exp)):
        print(f"\n{label}: {e['runs']} corridas, pasos media {e['steps_mean']}, mediana {e['steps_median']}, "
              f"máx {e['steps_max']}  vs  cota n² = {e['bound_n2']}")
        print(f"  {'t':>5} {'P(no terminó) real':>19} {'cota n²/t':>10}")
        for f in e["failure_prob"]:
            print(f"  {f['t']:>5} {f['empirical']:>19.4f} {f['bound']:>10.4f}")

    print(f"\nEjemplo insatisfacible ({', '.join(unsat['clauses'])}):")
    print(f"  fuerza bruta: satisfacible = {unsat['satisfiable_bruteforce']}; "
          f"encontró solución = {unsat['found_solution']}")
    print(f"  {unsat['report']}")
    print(f"\nEscrito {out_path}")


if __name__ == "__main__":
    main()
