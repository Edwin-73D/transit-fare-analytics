"""W2 Task 2 - Bloom filter para detectar taps duplicados dentro de un viaje.

Autor original: Felipe (w2_task2_bloom_filter.py). Integrado el 2026-10-23 (ver Decisiones).

Un VIAJE es un vehículo (vehicle_id) durante una ventana de 30 minutos. Cada vehículo tiene su
propio Bloom filter con las tarjetas que ya hicieron tap en el viaje actual; al empezar una
ventana nueva el filtro se vacía. Si la tarjeta ya está en el filtro, el tap se marca como
posible duplicado (doble cobro).

Teoría
------
Un Bloom filter es un arreglo de m bits y k funciones hash. Insertar = poner en 1 los k bits
h_1(x)..h_k(x); consultar = ver si los k bits están en 1.
  - Nunca hay falsos negativos: si x se insertó, sus k bits están en 1.
  - Puede haber falsos positivos: los k bits de una tarjeta nueva ya estaban en 1 por otras.
    Con n elementos insertados:  FPR = (1 - (1 - 1/m)^(k n))^k  ≈  (1 - e^(-k n / m))^k.
  - El k que minimiza el FPR es k = (m/n) ln 2; con él, FPR ≈ 0,6185^(m/n).
Memoria: m bits por vehículo, en vez de guardar las tarjetas (texto) del viaje.

Decisiones (integración, 2026-10-23)
------------------------------------
1. Viaje = vehicle_id + ventana de 30 min (lo acordado en el equipo; la versión original de
   Felipe usaba una ventana global de 10 min para todo el sistema).
2. Los taps llegan por w2/stream.py en UNA pasada; todas las configuraciones (m, k, semilla)
   se actualizan en esa misma pasada.
3. n de diseño = máximo de tarjetas distintas en un viaje (se calcula antes con pandas: es un
   parámetro de diseño, como la capacidad del vehículo). m = bits por elemento x n con 4, 8,
   12 y 16 bits por elemento (los de Felipe) y k = round((m/n) ln 2).
4. Dos mediciones del FPR (de Felipe):
   - en el stream: cada tarjeta nueva de un viaje se consulta ANTES de insertarla; como los
     filtros casi nunca se llenan (pocos taps por viaje), el FPR del stream es bajo;
   - con el filtro lleno (n tarjetas insertadas y 50.000 tarjetas que nunca se insertaron):
     es la tasa de diseño, la que se compara con la fórmula.
5. La "verdad" (un dict exacto tarjeta -> último tap, por viaje) solo se usa para medir; no es
   parte del algoritmo.
6. Duplicados inyectados = misma tarjeta y vehículo a <= 60 s (la regla del validador). Los
   que caen en la ventana siguiente (el original al final de una ventana y la copia al
   inicio de la otra) no se pueden detectar con esta definición de viaje: se reportan aparte.
   El filtro también marca repeticiones de la misma tarjeta en el viaje a más de 60 s (volvió a
   subir al mismo vehículo); se cuentan aparte como "otras repeticiones".
7. Familia de hash: src/hashing.py (la de Felipe). La gráfica de Felipe se reemplaza por el
   Panel A del W2 en dashboard.py.

Uso:  python w2/w2_task2_bloom.py        Salida: results/w2_panel_a.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))   # para importar src/hashing.py

from hashing import llave, hash_universal       # noqa: E402
from stream import describe, find_taps, stream_taps  # noqa: E402

RESULTS = os.path.join(RAIZ, "results")
SEED = 42
VIAJE = 1800                     # segundos (decisión 1)
BITS_POR_ELEMENTO = [4, 8, 12, 16]
SEMILLAS = 3
PRUEBAS_LLENO = 50_000           # tarjetas nunca insertadas para medir con el filtro lleno
DUP_S = 60                       # regla de duplicado (decisión 6)
SERIE_BITS = 8                   # configuración de la serie del Panel A
CADA = 1000                      # una foto de la serie cada CADA taps


class Bloom:
    def __init__(self, m, k, gen):
        self.m = m
        self.bits = bytearray(m)
        self.hashes = [hash_universal(m, gen) for _ in range(k)]

    def agregar(self, x):
        for h in self.hashes:
            self.bits[h(x)] = 1

    def __contains__(self, x):
        return all(self.bits[h(x)] for h in self.hashes)

    def vaciar(self):
        self.bits = bytearray(self.m)


def fpr_teorico(m, k, n):
    return (1 - (1 - 1 / m) ** (k * n)) ** k


def fpr_aprox(m, k, n):
    """La fórmula del enunciado: (1 - e^(-k n / m))^k."""
    return (1 - np.exp(-k * n / m)) ** k


def fpr_lleno(m, k, n, semilla):
    # inserta n tarjetas y prueba con tarjetas que no están (ids C010000 en adelante, no existen en el dataset)
    bloom = Bloom(m, k, np.random.default_rng(semilla))
    dentro = [llave("C%06d" % i) for i in range(1, n + 1)]
    for x in dentro:
        bloom.agregar(x)
    assert all(x in bloom for x in dentro)      # no hay falsos negativos
    fuera = [llave("C%06d" % i) for i in range(10_000, 10_000 + PRUEBAS_LLENO)]
    return sum(x in bloom for x in fuera) / len(fuera)


class Config:
    """Una configuración (m, k, semilla): un Bloom filter por vehículo y sus contadores."""

    def __init__(self, bits, m, k, semilla):
        self.bits, self.m, self.k = bits, m, k
        self.gen = np.random.default_rng(semilla)
        self.viajes = {}          # vehicle_id -> [ventana actual, Bloom, {tarjeta: último tap}]
        self.nuevas = self.falsos_pos = self.teorico = 0.0
        self.dup = self.dup_detectados = 0
        self.dup_rapidos = self.dup_rapidos_detectados = 0   # repetidos a <= DUP_S (los inyectados)

    def procesar(self, veh, ventana, sec, card, x):
        """Un tap: consulta el filtro del vehículo y luego inserta la tarjeta."""
        viaje = self.viajes.get(veh)
        if viaje is None:
            viaje = self.viajes[veh] = [ventana, Bloom(self.m, self.k, self.gen), {}]
        elif viaje[0] != ventana:                   # viaje nuevo: se vacía el filtro
            viaje[0] = ventana
            viaje[1].vaciar()
            viaje[2] = {}
        _, bloom, vistas = viaje
        marcado = x in bloom
        if card in vistas:                          # repetida de verdad en el viaje
            self.dup += 1
            self.dup_detectados += marcado
            if sec - vistas[card] <= DUP_S:         # doble tap rápido (como los inyectados)
                self.dup_rapidos += 1
                self.dup_rapidos_detectados += marcado
            vistas[card] = sec
        else:
            self.nuevas += 1
            self.teorico += fpr_teorico(self.m, self.k, len(vistas))
            self.falsos_pos += marcado              # nueva, pero el filtro dijo que ya estaba
            vistas[card] = sec
            bloom.agregar(x)

    def fpr(self):
        return self.falsos_pos / self.nuevas if self.nuevas else 0.0


def diseno(path):
    """Decisión 3 y 6: n de diseño y duplicados inyectados (con pandas, antes de la pasada)."""
    df = pd.read_csv(path, usecols=["timestamp", "card_id", "vehicle_id"])
    ts = pd.to_datetime(df["timestamp"], format="ISO8601")
    sec = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(int)
    df = df.assign(ts=ts, ventana=sec // VIAJE).sort_values("ts", kind="stable")
    n = int(df.groupby(["vehicle_id", "ventana"])["card_id"].nunique().max())
    gap = df.groupby(["card_id", "vehicle_id"])["ts"].diff().dt.total_seconds()
    prev_ventana = df.groupby(["card_id", "vehicle_id"])["ventana"].shift()
    es_dup = gap.between(0, DUP_S)
    return n, int(es_dup.sum()), int((es_dup & (prev_ventana != df["ventana"])).sum())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS, exist_ok=True)
    path = find_taps()
    n, dup_inyectados, dup_borde = diseno(path)
    print(f"Stream: {describe(path)}")
    print(f"n de diseño (máx. tarjetas distintas en un viaje de {VIAJE // 60} min): {n}")

    configs = []
    for bits in BITS_POR_ELEMENTO:
        m = bits * n
        k = max(1, round(bits * np.log(2)))
        configs += [Config(bits, m, k, SEED + s) for s in range(SEMILLAS)]
    serie_cfg = next(c for c in configs if c.bits == SERIE_BITS)

    serie = []
    for tap in stream_taps(path):               # una sola pasada (decisión 2)
        sec = int((tap.timestamp - pd.Timestamp("1970-01-01").to_pydatetime()).total_seconds())
        x = llave(tap.card_id)
        for c in configs:
            c.procesar(tap.vehicle_id, sec // VIAJE, sec, tap.card_id, x)
        if tap.stream_seq % CADA == 0:
            serie.append({"stream_seq": tap.stream_seq, "fpr_so_far": round(serie_cfg.fpr(), 6)})

    assert all(c.dup == c.dup_detectados for c in configs), "un Bloom filter no puede dejar pasar duplicados"
    runs = []
    for bits in BITS_POR_ELEMENTO:
        grupo = [c for c in configs if c.bits == bits]
        m, k = grupo[0].m, grupo[0].k
        runs.append({
            "bits_per_element": bits,
            "m": m,
            "k": k,
            "n": n,
            "fpr_empirical": round(float(np.mean([fpr_lleno(m, k, n, SEED + s) for s in range(SEMILLAS)])), 5),
            "fpr_theoretical": round(float(fpr_aprox(m, k, n)), 5),
            "fpr_theoretical_exact": round(fpr_teorico(m, k, n), 5),
            "fpr_stream_empirical": round(float(np.mean([c.fpr() for c in grupo])), 5),
            "fpr_stream_theoretical": round(float(np.mean([c.teorico / c.nuevas for c in grupo])), 5),
            "memory_bits_per_vehicle": m,
        })
    tabla = pd.DataFrame(runs)
    print(tabla.to_string(index=False))

    detectados = serie_cfg.dup_rapidos_detectados
    otros = serie_cfg.dup_detectados - detectados
    print(f"duplicados inyectados (<= {DUP_S} s): {dup_inyectados} | detectados: {detectados} | "
          f"no detectables (cruzan el borde de la ventana): {dup_borde}")
    print(f"otras repeticiones en el mismo viaje (> {DUP_S} s, también marcadas): {otros}")
    print(f"vehículos con filtro: {len(serie_cfg.viajes)}")

    salida = {
        "runs": runs,
        "series": serie,
        "duplicates_injected": dup_inyectados,
        "duplicates_detected": detectados,
        "duplicates_window_boundary": dup_borde,
        "other_repeats_in_trip": otros,
        "meta": {
            "task": "W2 Task 2 - Bloom filter de tarjetas por viaje (vehicle_id + 30 min)",
            "source": describe(path),
            "trip_window_s": VIAJE,
            "n_design": n,
            "seeds": SEMILLAS,
            "series_bits_per_element": SERIE_BITS,
            "vehicles": len(serie_cfg.viajes),
            "fpr_empirical": "con el filtro lleno (n tarjetas) y 50.000 tarjetas que nunca se insertaron",
            "fpr_theoretical": "(1 - e^(-k n / m))^k",
        },
    }
    with open(os.path.join(RESULTS, "w2_panel_a.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print("Escrito", os.path.join(RESULTS, "w2_panel_a.json"))


if __name__ == "__main__":
    main()
