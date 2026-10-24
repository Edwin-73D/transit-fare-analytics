"""W1 Task 3 - Hash universal sobre card_id.

Mide las colisiones de h(k) = ((a*k + b) mod p) mod m con muchas funciones al azar,
las compara con la cota teórica C(n,2)/m y con un hash simple (k mod m), y arma un
índice por tarjeta con chaining. Guarda results/panel_a.json.
"""
import json
import os
import numpy as np
import pandas as pd
from generador import cargar, llave, hash_universal, RESULTS, P

TAMANOS = [1024, 4096, 16384]    # tamaños de tabla (m)
PRUEBAS = 200                    # funciones al azar por cada m
M_INDICE = 4096


def cargas(llaves, h, m):
    # cuántas llaves caen en cada bucket
    return np.bincount([h(k) for k in llaves], minlength=m)


def pares(c):
    # pares de llaves en el mismo bucket
    return int((c * (c - 1) // 2).sum())


def histograma(c):
    # cuántos buckets tienen carga 0, 1, 2...
    return {str(i): int(v) for i, v in enumerate(np.bincount(c))}


def main():
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

    salida = {"task3_universal_hashing": {
        "hash_family": "((a*k + b) mod p) mod m, p = 2^61 - 1",
        "n_keys": n,
        "n_trials": PRUEBAS,
        "por_tamano_de_tabla": resumen,
        "indice_por_tarjeta": indice,
    }}
    with open(os.path.join(RESULTS, "panel_a.json"), "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
