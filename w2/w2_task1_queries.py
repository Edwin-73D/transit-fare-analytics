"""W2 Task 1 - Modelo del stream y diseño de consultas.

Autor original: Felipe (w2_task1_stream_model.py). Integrado el 2026-10-23 (ver Decisiones).

Standing query: taps por ruta en los últimos 5 minutos (ventana deslizante con una cola).
Ad-hoc query: tarjetas distintas que han tapeado hoy (set que se reinicia cada día).

Teoría
------
- Standing query = consulta permanente: se responde en todo momento mientras llega el stream.
  Se mantiene una cola (deque) con los taps de los últimos 5 min y un contador por ruta: cada
  tap nuevo entra por la derecha y por la izquierda salen los que ya quedaron fuera de la
  ventana. Memoria = taps dentro de la ventana, no toda la historia.
- Ad-hoc query = pregunta que llega en un momento cualquiera ("¿cuántas tarjetas distintas van
  hoy?"). Para responderla sin recorrer la historia se mantiene un set con las tarjetas del día
  (conteo exacto), que se vacía al cambiar de día. Memoria = tarjetas distintas del día.

Decisiones (integración, 2026-10-23)
------------------------------------
1. Los taps llegan UNO A LA VEZ por w2/stream.py (una sola pasada, como todo el W2). Las clases
   TapsPorRuta y DistintasHoy son las de Felipe, sin cambios.
2. Una sola cola para todas las rutas + un contador por ruta (equivale a una cola por ruta y
   usa la misma memoria: cada tap está en una sola cola).
3. Cada 5.000 taps se guarda una foto con el conteo de TODAS las rutas en la ventana.
4. La verificación contra un cálculo directo (pandas, con todo el CSV) se hace DESPUÉS de la
   pasada: es una prueba de que el algoritmo está bien, no parte del algoritmo de streaming.
5. Se reporta el tamaño máximo de la cola: es la memoria que de verdad usó la standing query.

Uso:  python w2/w2_task1_queries.py        Salida: results/w2_task1.json
"""
import json
import os
import sys
from collections import deque, defaultdict
from datetime import datetime

import pandas as pd

from stream import describe, find_taps, stream_taps

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(RAIZ, "results")

VENTANA = 300        # segundos
CADA = 5000          # cada cuántos taps se consulta
EPOCA = datetime(1970, 1, 1)


class TapsPorRuta:
    # cuenta los taps de cada ruta dentro de los últimos `ventana` segundos
    def __init__(self, ventana):
        self.ventana = ventana
        self.cola = deque()
        self.conteo = defaultdict(int)

    def agregar(self, sec, ruta):
        self.cola.append((sec, ruta))
        self.conteo[ruta] += 1
        # sacar lo que ya quedó fuera de la ventana
        while self.cola[0][0] <= sec - self.ventana:
            _, r = self.cola.popleft()
            self.conteo[r] -= 1

    def top(self, k=3):
        vivos = {r: c for r, c in self.conteo.items() if c > 0}
        return sorted(vivos.items(), key=lambda x: -x[1])[:k]


class DistintasHoy:
    # tarjetas distintas del día actual, se reinicia cuando cambia el día
    def __init__(self):
        self.dia = None
        self.tarjetas = set()

    def agregar(self, dia, card):
        if dia != self.dia:
            self.dia, self.tarjetas = dia, set()
        self.tarjetas.add(card)


def verificar(path, puntos):
    """Decisión 4: recalcula cada foto con pandas sobre el CSV completo y la compara."""
    df = pd.read_csv(path, usecols=["timestamp", "card_id", "route_id"])
    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    if "stream_seq" in pd.read_csv(path, nrows=0).columns:
        orden = pd.read_csv(path, usecols=["stream_seq"])["stream_seq"].argsort(kind="stable")
    else:
        orden = ts.argsort(kind="stable")
    df, ts = df.iloc[orden].reset_index(drop=True), ts.iloc[orden].reset_index(drop=True)
    secs = (ts - EPOCA).dt.total_seconds().astype(int).to_numpy()
    dias = ts.dt.date.to_numpy()
    todo_ok = True
    for p in puntos:
        i = p["stream_seq"]
        sec = secs[i - 1]
        en_ventana = (secs[:i] > sec - VENTANA) & (secs[:i] <= sec)
        directo = df["route_id"].iloc[:i][en_ventana].value_counts().to_dict()
        distintas = df["card_id"].iloc[:i][dias[:i] == dias[i - 1]].nunique()
        p["verificado"] = (directo == p["counts"]) and distintas == p["distintas_hoy"]
        todo_ok = todo_ok and p["verificado"]
    return todo_ok


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS, exist_ok=True)
    path = find_taps()

    rolling = TapsPorRuta(VENTANA)
    hoy = DistintasHoy()
    puntos, max_cola, ultimo = [], 0, None
    for tap in stream_taps(path):            # una sola pasada (decisión 1)
        sec = int((tap.timestamp - EPOCA).total_seconds())
        rolling.agregar(sec, tap.route_id)
        hoy.agregar(tap.timestamp.date(), tap.card_id)
        max_cola = max(max_cola, len(rolling.cola))
        ultimo = tap
        if tap.stream_seq % CADA:
            continue
        puntos.append({
            "stream_seq": tap.stream_seq,
            "timestamp": str(tap.timestamp),
            "counts": {r: c for r, c in sorted(rolling.conteo.items()) if c > 0},
            "taps_en_ventana": len(rolling.cola),
            "top3_rutas": rolling.top(3),
            "distintas_hoy": len(hoy.tarjetas),
        })

    todo_ok = verificar(path, puntos)
    print(f"Stream: {describe(path)}, una sola pasada, foto cada {CADA} taps")
    print(pd.DataFrame([{k: v for k, v in p.items() if k != "counts"} for p in puntos]).to_string(index=False))
    print("consultas verificadas:", todo_ok, "| cola máxima:", max_cola, "taps")
    print(f"ad-hoc al final del stream ({ultimo.timestamp}): {len(hoy.tarjetas)} tarjetas distintas hoy")

    salida = {
        "window_min": VENTANA // 60,
        "snapshots": puntos,
        "adhoc": {"timestamp": str(ultimo.timestamp), "exact_distinct_cards": len(hoy.tarjetas)},
        "verified": bool(todo_ok),
        "max_queue_len": max_cola,
        "meta": {
            "task": "W2 Task 1 - standing query (taps por ruta en 5 min) y ad-hoc (tarjetas distintas hoy)",
            "source": describe(path),
            "standing_query": "taps por ruta en los últimos %d s (deque)" % VENTANA,
            "ad_hoc_query": "tarjetas distintas en el día actual (set exacto, reinicio diario)",
        },
    }
    with open(os.path.join(RESULTS, "w2_task1.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print("Escrito", os.path.join(RESULTS, "w2_task1.json"))


if __name__ == "__main__":
    main()
