"""
toy_matrix.py - Matriz de juguete de 5 rutas para probar los Tasks 2, 3 y 5 del W3.
==================================================================================

Mientras no exista la matriz real (results/w3_matrix.json), los tasks se prueban con este
grafo pequeño, diseñado para que tenga los casos difíciles que el enunciado pide manejar.

Grafo de transbordos (transfer_count en cada arista)
----------------------------------------------------
  Troncales (entre ellas):   L-A -> L-B 40    L-A -> L-C 20
                             L-B -> L-A 30    L-B -> L-C 25
                             L-C -> L-A 35    L-C -> L-B 10
  Alimentadora B-201:        B-201 -> L-A 30  B-201 -> L-B 10  B-201 -> B-202 10
  B-202:                     ninguna salida
  Ninguna arista llega a B-201, y ninguna sale de las troncales hacia B-201 o B-202.

Qué se espera de cada ruta (respuesta conocida, para verificar los tasks)
-------------------------------------------------------------------------
- {L-A, L-B, L-C}: clase CERRADA (nadie sale de ella) -> RECURRENTE. Tiene ciclos de largo 2
  (L-A -> L-B -> L-A) y 3 (L-A -> L-B -> L-C -> L-A); mcd(2, 3) = 1 -> APERIÓDICA.
- B-201: nadie hace transbordo HACIA ella (alimentadora sin transbordos desde las troncales),
  así que una vez que se sale no se vuelve -> TRANSITORIA. Es el caso que el W3 pide al generador.
- B-202: fila de ceros (ruta sin salida, "dangling"). La matriz no queda estocástica en esa fila:
  cada task debe decir cómo la trata (PageRank: reparte su masa uniformemente entre todas).

Pasajeros (taps por ruta) para comparar rankings
------------------------------------------------
Elegidos para que el ranking por PageRank NO coincida con el de pasajeros: B-201 tiene muchos
pasajeros pero nadie le llega por transbordo (debería bajar), y L-C tiene pocos pasajeros pero
recibe transbordos de las otras troncales (debería subir).

Uso
---
    from toy_matrix import toy_matrix, toy_edges, toy_ridership
    python w3/toy_matrix.py          # imprime la matriz y verifica las clases
"""

from __future__ import annotations

import sys

ROUTES = ["L-A", "L-B", "L-C", "B-201", "B-202"]

# (from_route, to_route, transfer_count, avg_fare_amount, transfer_frac)
EDGES = [
    ("L-A", "L-B", 40, 0.45, 0.82),
    ("L-A", "L-C", 20, 0.60, 0.78),
    ("L-B", "L-A", 30, 0.50, 0.80),
    ("L-B", "L-C", 25, 0.55, 0.80),
    ("L-C", "L-A", 35, 0.40, 0.86),
    ("L-C", "L-B", 10, 0.70, 0.70),
    ("B-201", "L-A", 30, 0.90, 0.67),
    ("B-201", "L-B", 10, 1.10, 0.60),
    ("B-201", "B-202", 10, 1.38, 0.50),
]

RIDERSHIP = {"L-A": 500, "L-B": 400, "B-201": 300, "L-C": 150, "B-202": 120}

# Respuesta conocida de las clases comunicantes (sirve para verificar el Task 1 y los demás).
EXPECTED_CLASSES = [
    {"routes": ["L-A", "L-B", "L-C"], "closed": True, "type": "recurrent", "period": 1},
    {"routes": ["B-201"], "closed": False, "type": "transient", "period": None},
    {"routes": ["B-202"], "closed": None, "type": "dangling (depende del tratamiento)", "period": None},
]


def toy_edges() -> list[dict]:
    """Aristas en el formato del enunciado (el mismo de results/w3_edges.json)."""
    return [
        {"edge_id": f"e{i:03d}", "from_route": a, "to_route": b, "transfer_count": c,
         "avg_fare_amount": fare, "transfer_frac": frac}
        for i, (a, b, c, fare, frac) in enumerate(EDGES, start=1)
    ]


def build_matrix(routes: list[str], edges: list[dict]) -> tuple[list[list[float]], list[str]]:
    """M[i][j] = transfer_count(i -> j) / total que sale de i. Las filas sin salida quedan en ceros."""
    idx = {r: i for i, r in enumerate(routes)}
    counts = [[0.0] * len(routes) for _ in routes]
    for e in edges:
        counts[idx[e["from_route"]]][idx[e["to_route"]]] += e["transfer_count"]
    m, dangling = [], []
    for r, row in zip(routes, counts):
        total = sum(row)
        if total == 0:
            dangling.append(r)
            m.append([0.0] * len(routes))  # se deja en ceros: cada task decide cómo tratarla
        else:
            m.append([c / total for c in row])
    return m, dangling


def toy_matrix() -> dict:
    """Matriz de juguete con el mismo formato propuesto para results/w3_matrix.json."""
    m, dangling = build_matrix(ROUTES, toy_edges())
    return {"routes": list(ROUTES), "M": m, "dangling": dangling, "classes": EXPECTED_CLASSES,
            "source": "toy_matrix.py (matriz de juguete)"}


def toy_ridership() -> dict[str, int]:
    """Pasajeros (taps) por ruta de juguete, para comparar con PageRank y con π."""
    return dict(RIDERSHIP)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = toy_matrix()
    routes, m = data["routes"], data["M"]
    print("Matriz de transición M (fila = desde, columna = hacia):\n")
    print(f"{'':>7}" + "".join(f"{r:>8}" for r in routes) + "   suma fila")
    for r, row in zip(routes, m):
        print(f"{r:>7}" + "".join(f"{v:>8.3f}" for v in row) + f"   {sum(row):.1f}")
    print(f"\nRutas sin salida (dangling): {data['dangling']}")

    # Verificación independiente de las clases con componentes fuertemente conexas (scipy).
    from scipy.sparse.csgraph import connected_components
    import numpy as np
    n, labels = connected_components(np.array(m), directed=True, connection="strong")
    groups: dict[int, list[str]] = {}
    for r, lab in zip(routes, labels):
        groups.setdefault(lab, []).append(r)
    print(f"Clases comunicantes (scipy, {n}): {sorted(groups.values(), key=len, reverse=True)}")
    print("Esperado: [L-A, L-B, L-C] recurrente y aperiódica; [B-201] transitoria; [B-202] dangling")


if __name__ == "__main__":
    main()
