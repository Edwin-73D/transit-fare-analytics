"""W2 Task 1 - Modelo del stream y diseño de consultas.

Standing query: taps por ruta en los últimos 5 minutos (ventana deslizante con una cola).
Ad-hoc query: tarjetas distintas que han tapeado hoy (set que se reinicia cada día).
Cada 5000 taps se consultan las dos y se comparan con un cálculo directo en pandas.
Guarda results/w2_task1.json.
"""
import json
import os
from collections import deque, defaultdict
import pandas as pd
from generador import cargar, armar_stream, RESULTS

VENTANA = 300        # segundos
CADA = 5000          # cada cuántos taps se consulta


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


def main():
    os.makedirs(RESULTS, exist_ok=True)
    stream = armar_stream(cargar("100k"))
    print(stream.drop(columns="sec").head(1).to_json(orient="records", date_format="iso", force_ascii=False))

    rolling = TapsPorRuta(VENTANA)
    hoy = DistintasHoy()
    secs = stream["sec"].to_numpy()
    rutas = stream["route_id"].to_numpy()
    dias = secs // 86400

    puntos = []
    todo_ok = True
    for i, (sec, ruta, card) in enumerate(zip(secs, rutas, stream["card_id"]), start=1):
        rolling.agregar(sec, ruta)
        hoy.agregar(sec // 86400, card)
        if i % CADA:
            continue

        # cálculo directo para comparar
        en_ventana = (secs[:i] > sec - VENTANA) & (secs[:i] <= sec)
        directo = pd.Series(rutas[:i][en_ventana]).value_counts()
        distintas = stream["card_id"].iloc[:i][dias[:i] == sec // 86400].nunique()
        ok = all(rolling.conteo[r] == c for r, c in directo.items()) \
            and sum(rolling.conteo.values()) == en_ventana.sum() \
            and len(hoy.tarjetas) == distintas
        todo_ok = todo_ok and ok

        puntos.append({
            "stream_seq": i,
            "timestamp": str(stream["timestamp"].iloc[i - 1]),
            "taps_en_ventana": int(en_ventana.sum()),
            "top3_rutas": rolling.top(3),
            "distintas_hoy": len(hoy.tarjetas),
            "elementos_en_cola": len(rolling.cola),
            "verificado": bool(ok),
        })

    print(pd.DataFrame(puntos).to_string(index=False))
    print("consultas verificadas:", todo_ok)

    salida = {"w2_task1_stream_model": {
        "standing_query": "taps por ruta en los últimos %d s (deque)" % VENTANA,
        "ad_hoc_query": "tarjetas distintas en el día actual (set exacto, reinicio diario)",
        "verificado": bool(todo_ok),
        "puntos": puntos,
    }}
    with open(os.path.join(RESULTS, "w2_task1.json"), "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
