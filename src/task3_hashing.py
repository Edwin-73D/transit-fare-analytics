"""W1 Task 3 - Hash universal sobre card_id (EXTENDIDO: una función vs power of two choices).

Autor original: Felipe (w1_task3_universal_hashing.py). Integrado el 2026-10-23: se cambió
el import y se agregó la extensión (power of two choices) que alimenta el Panel A.

Parte 1 (Felipe): mide las colisiones de h(k) = ((a*k + b) mod p) mod m con muchas funciones
al azar, las compara con la cota teórica C(n,2)/m y con un hash simple (k mod m), y arma un
índice por tarjeta con chaining.

Parte 2 (extensión): distribución de carga por bucket con m = n buckets.
  - Una función: cada tarjeta va al bucket h1(k).
  - Power of two choices (P2C): se calculan dos buckets h1(k) y h2(k) con dos funciones de la
    familia, y la tarjeta va al que tenga MENOS tarjetas en ese momento.
  Teoría (n llaves en n buckets, con alta probabilidad):
    carga máxima con una función  ~ ln n / ln ln n
    carga máxima con P2C          ~ ln ln n / ln 2 + O(1)
  Pasar de una opción a dos baja la carga máxima de forma exponencial (de ~ln n a ~ln ln n):
  es el "poder de dos opciones". Las fórmulas son de orden de crecimiento (asintóticas), así
  que se comparan como tendencia, no como valor exacto.

Decisiones (integración, 2026-10-23)
------------------------------------
1. La familia de hash viene de src/hashing.py (la misma de Felipe, sin cambios).
2. llave(card_id) = los bytes del texto como entero: es un hash polinomial en base 256
   (sum c_i * 256^(L-1-i)); el "mod p" no cambia nada porque la llave ya es menor que p.
3. Extensión con n = 3.000 tarjetas y m = n buckets (el caso clásico del teorema) y
   REPETICIONES funciones al azar (semillas distintas), para reportar promedios y no un caso
   suelto. El histograma que dibuja el Panel A es el de la primera corrida.
4. results/panel_a.json sigue el formato que lee el dashboard ("single_hash" y "p2c" con
   max_load, std, histogram); los resultados de la parte 1 van en claves extra.

Uso:  python src/task3_hashing.py        Salida: results/panel_a.json
"""
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from generator import cargar, RESULTS
from hashing import llave, hash_universal, P

TAMANOS = [1024, 4096, 16384]    # tamaños de tabla (m)
PRUEBAS = 200                    # funciones al azar por cada m
M_INDICE = 4096
REPETICIONES = 200               # corridas de la extensión (decisión 3)
SEED_P2C = 1000                  # semillas de la extensión: SEED_P2C + corrida


def cargas(llaves, h, m):
    # cuántas llaves caen en cada bucket
    return np.bincount([h(k) for k in llaves], minlength=m)


def pares(c):
    # pares de llaves en el mismo bucket
    return int((c * (c - 1) // 2).sum())


def histograma(c):
    # cuántos buckets tienen carga 0, 1, 2...
    return {str(i): int(v) for i, v in enumerate(np.bincount(c))}


def cargas_p2c(llaves, h1, h2, m):
    """Power of two choices: cada llave va al menos lleno de sus dos buckets (empate -> h1)."""
    c = np.zeros(m, dtype=int)
    for k in llaves:
        b1, b2 = h1(k), h2(k)
        c[b1 if c[b1] <= c[b2] else b2] += 1
    return c


def resumen_cargas(c, maximos):
    """Lo que lee el Panel A (de la primera corrida) + promedios sobre todas las corridas."""
    return {
        "max_load": int(c.max()),
        "std": round(float(c.std()), 4),
        "histogram": histograma(c),
        "max_load_mean": round(float(np.mean(maximos)), 3),
        "max_load_distribution": {str(v): int(n) for v, n in zip(*np.unique(maximos, return_counts=True))},
    }


def extension_p2c(llaves):
    """Una función vs power of two choices con m = n buckets, REPETICIONES veces."""
    n = m = len(llaves)
    max_una, max_p2c = [], []
    for r in range(REPETICIONES):
        gen = np.random.default_rng(SEED_P2C + r)
        h1, h2 = hash_universal(m, gen), hash_universal(m, gen)
        una, dos = cargas(llaves, h1, m), cargas_p2c(llaves, h1, h2, m)
        if r == 0:
            primera_una, primera_dos = una, dos
        max_una.append(int(una.max()))
        max_p2c.append(int(dos.max()))
    ln_n = math.log(n)
    return {
        "n_keys": n,
        "n_buckets": m,
        "repetitions": REPETICIONES,
        "single_hash": resumen_cargas(primera_una, max_una),
        "p2c": resumen_cargas(primera_dos, max_p2c),
        "theory": {"single_ln_n_over_ln_ln_n": round(ln_n / math.log(ln_n), 3),
                   "p2c_ln_ln_n_over_ln_2": round(math.log(ln_n) / math.log(2), 3)},
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # tildes bien en la consola de Windows
    os.makedirs(RESULTS, exist_ok=True)
    datos = {"10k": cargar("10k"), "100k": cargar("100k")}

    tarjetas = sorted(datos["100k"]["card_id"].unique())
    assert tarjetas == sorted(datos["10k"]["card_id"].unique())
    llaves = [llave(c) for c in tarjetas]
    assert max(llaves) < P
    n = len(llaves)
    print("tarjetas distintas:", n)

    # colisiones: teoría vs empírico
    filas, resumen = [], []
    for m in TAMANOS:
        teoria = n * (n - 1) / 2 / m
        total, maximos = [], []
        for t in range(PRUEBAS):
            c = cargas(llaves, hash_universal(m), m)
            if t == 0:
                primera = c
            total.append(pares(c))
            maximos.append(int(c.max()))
        simple = cargas(llaves, lambda k: k % m, m)

        fila = {
            "m": m,
            "load_factor": round(n / m, 3),
            "teoria_pares": round(teoria, 1),
            "universal_pares_prom": round(float(np.mean(total)), 1),
            "universal_pares_mediana": float(np.median(total)),
            "universal_pares_max": int(np.max(total)),
            "universal_carga_max_prom": round(float(np.mean(maximos)), 2),
            "simple_pares": pares(simple),
            "simple_carga_max": int(simple.max()),
        }
        filas.append(fila)
        resumen.append({**fila, "histograma_universal": histograma(primera), "histograma_simple": histograma(simple)})
    print(pd.DataFrame(filas).to_string(index=False))

    # índice por tarjeta: cada bucket guarda [card_id, filas del dataset]
    h = hash_universal(M_INDICE)
    llave_de = {c: llave(c) for c in tarjetas}

    def armar(df):
        tabla = [[] for _ in range(M_INDICE)]
        for fila, card in enumerate(df["card_id"]):
            bucket = tabla[h(llave_de[card])]
            for entrada in bucket:
                if entrada[0] == card:
                    entrada[1].append(fila)
                    break
            else:
                bucket.append([card, [fila]])
        return tabla

    def buscar(tabla, card):
        for entrada in tabla[h(llave_de[card])]:
            if entrada[0] == card:
                return entrada[1]
        return []

    indice = {}
    for nombre, df in datos.items():
        tabla = armar(df)
        conteo = df["card_id"].value_counts()
        ok = all(len(buscar(tabla, c)) == conteo[c] for c in tarjetas)    # cada búsqueda trae todos sus taps
        indice[nombre] = {"taps": len(df), "tarjetas": n,
                          "bucket_mas_largo": max(len(b) for b in tabla), "busquedas_correctas": bool(ok)}
        print(nombre, indice[nombre])

    # extensión: una función vs power of two choices (Panel A)
    ext = extension_p2c(llaves)
    print(f"\nExtensión (n = m = {n}, {REPETICIONES} corridas): carga máxima promedio "
          f"una función {ext['single_hash']['max_load_mean']} vs P2C {ext['p2c']['max_load_mean']}")
    print(f"  distribución de la carga máxima: una función {ext['single_hash']['max_load_distribution']}, "
          f"P2C {ext['p2c']['max_load_distribution']}")
    print(f"  teoría (orden de crecimiento): ln n/ln ln n = {ext['theory']['single_ln_n_over_ln_ln_n']}, "
          f"ln ln n/ln 2 = {ext['theory']['p2c_ln_ln_n_over_ln_2']}")

    salida = {
        **ext,                                   # lo que lee el Panel A (decisión 4)
        "hash_family": "((a*k + b) mod p) mod m, p = 2^61 - 1",
        "collisions": {"n_trials": PRUEBAS, "por_tamano_de_tabla": resumen},
        "index": indice,
    }
    with open(os.path.join(RESULTS, "panel_a.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print("Escrito", os.path.join(RESULTS, "panel_a.json"))


if __name__ == "__main__":
    main()
