"""W3 - Aristas de transbordo entre rutas (grafo del que salen la cadena de Markov y PageRank).

Autor original: Felipe (armar_aristas, dentro de su w3_task3_pagerank.py). Integrado el
2026-10-23 (ver Decisiones). El PageRank del equipo está en w3/w3_task3_pagerank.py.

Cada arista es un TRANSBORDO: una tarjeta hace tap en una ruta y su tap siguiente es en otra
ruta, dentro del mismo viaje. Se agrupan por par (from_route, to_route):
  - transfer_count: cuántos transbordos de esa ruta a esa otra;
  - avg_fare_amount: tarifa promedio del segundo tap (COP; 0 si fue transferencia);
  - transfer_frac: fracción de esos transbordos cuyo segundo tap tiene transfer_flag = true.

Decisiones (integración, 2026-10-23)
------------------------------------
1. Transbordo = dos taps SEGUIDOS de la misma tarjeta, en rutas DISTINTAS y a <= 90 min
   (decisión de Edwin). Sin el límite de tiempo, el último tap de un día y el primero del día
   siguiente contarían como transbordo (p. ej. L-A -> B-103) y ninguna alimentadora podría
   quedar transitoria. 90 min es lo que dura, de sobra, un viaje con cambio de ruta.
2. Taps seguidos en la MISMA ruta no son arista (no hubo cambio de ruta; incluye los ~2% de
   duplicados del generador v2), pero sí actualizan el último tap de la tarjeta.
3. Se lee el stream con w2/stream.py en UNA pasada (orden de stream_seq): por tarjeta solo se
   guarda su último tap (ruta y hora), así la memoria es O(tarjetas) y no O(taps).
4. Comprobación: la versión de Felipe con pandas (agrupar por card_id, ordenar por stream_seq,
   shift) más el filtro de 90 min debe dar exactamente las mismas aristas.
5. Salida en el formato del enunciado (lista de aristas, edge_id e001, e002, ...).

Uso:  python w3/build_edges.py        Salida: results/w3_edges.json
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "w2"))    # para importar w2/stream.py

from stream import describe, find_taps, stream_taps  # noqa: E402

RESULTS = os.path.join(RAIZ, "results")
TRANSBORDO_MAX = 90 * 60     # segundos (decisión 1)


def armar_aristas_stream(path):
    """Decisión 3: una pasada; por tarjeta solo su último tap (ruta, hora)."""
    ultimo = {}                                          # card_id -> (route_id, timestamp)
    conteo = defaultdict(lambda: [0, 0.0, 0])            # (desde, hacia) -> [n, suma tarifas, n transfer]
    taps = 0
    for tap in stream_taps(path):
        taps += 1
        previo = ultimo.get(tap.card_id)
        if previo is not None:
            ruta, hora = previo
            if ruta != tap.route_id and (tap.timestamp - hora).total_seconds() <= TRANSBORDO_MAX:
                c = conteo[(ruta, tap.route_id)]
                c[0] += 1
                c[1] += tap.fare_amount
                c[2] += tap.transfer_flag
        ultimo[tap.card_id] = (tap.route_id, tap.timestamp)   # decisión 2: siempre se actualiza

    aristas = []
    for i, ((desde, hacia), (n, tarifas, transfer)) in enumerate(sorted(conteo.items()), start=1):
        aristas.append({
            "edge_id": "e%03d" % i,
            "from_route": desde,
            "to_route": hacia,
            "transfer_count": n,
            "avg_fare_amount": round(tarifas / n, 2),
            "transfer_frac": round(transfer / n, 3),
        })
    return aristas, taps, len(ultimo)


def armar_aristas(stream):
    # versión original de Felipe (pandas) + el filtro de 90 min (decisión 4)
    s = stream.sort_values(["card_id", "stream_seq"])
    sig = s.groupby("card_id")[["route_id", "fare_amount", "transfer_flag", "timestamp"]].shift(-1)
    pares = pd.DataFrame({
        "from_route": s["route_id"],
        "to_route": sig["route_id"],
        "fare": sig["fare_amount"],
        "transfer": sig["transfer_flag"],
        "dt": (sig["timestamp"] - s["timestamp"]).dt.total_seconds(),
    }).dropna(subset=["to_route"])
    pares["transfer"] = pares["transfer"].astype(float)
    pares = pares[(pares["from_route"] != pares["to_route"]) & (pares["dt"] <= TRANSBORDO_MAX)]

    aristas = pares.groupby(["from_route", "to_route"]).agg(
        transfer_count=("fare", "size"),
        avg_fare_amount=("fare", "mean"),
        transfer_frac=("transfer", "mean"),
    ).reset_index()
    aristas["avg_fare_amount"] = aristas["avg_fare_amount"].round(2)
    aristas["transfer_frac"] = aristas["transfer_frac"].round(3)
    aristas.insert(0, "edge_id", ["e%03d" % i for i in range(1, len(aristas) + 1)])
    return aristas


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS, exist_ok=True)
    path = find_taps()
    aristas, taps, tarjetas = armar_aristas_stream(path)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    directo = armar_aristas(df).to_dict("records")
    iguales = directo == aristas

    rutas = sorted({a["from_route"] for a in aristas} | {a["to_route"] for a in aristas})
    con_entrada = {a["to_route"] for a in aristas}
    con_salida = {a["from_route"] for a in aristas}
    total = sum(a["transfer_count"] for a in aristas)
    print(f"Stream: {describe(path)}, una sola pasada, memoria = último tap de {tarjetas:,} tarjetas")
    print(f"Transbordos (rutas distintas, <= {TRANSBORDO_MAX // 60} min): {total:,} en {len(aristas)} aristas "
          f"de {len(rutas) * (len(rutas) - 1)} pares posibles ({len(rutas)} rutas)")
    print("Rutas a las que nadie transborda:", sorted(set(rutas) - con_entrada) or "ninguna")
    print("Rutas sin salida (dangling):", sorted(set(rutas) - con_salida) or "ninguna")
    print("Coincide con la versión pandas de Felipe (+ filtro de 90 min):", iguales)
    print("Ejemplo:", aristas[0])

    out = os.path.join(RESULTS, "w3_edges.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(aristas, f, indent=2, ensure_ascii=False)
    print("Escrito", out)


if __name__ == "__main__":
    main()
