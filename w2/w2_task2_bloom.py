"""W2 Task 2 - Bloom filter para detectar taps duplicados.

Un "viaje" es una ventana de 10 minutos (VIAJE). Dentro de la ventana, si el card_id ya
está en el filtro el tap se marca como posible duplicado; al cambiar de ventana el filtro
se vacía. Se compara la tasa de falsos positivos medida con la teórica
(1 - (1 - 1/m)^(k*n))^k, para varios tamaños del filtro, de dos formas:
- en el stream: el filtro se llena poco a poco dentro de cada ventana, así que la tasa
  es un promedio sobre todos los niveles de llenado;
- con el filtro lleno (n tarjetas insertadas) y tarjetas que nunca se insertaron: es la
  tasa de diseño, la que se compara con la fórmula del libro.
Guarda results/w2_task2.json y una gráfica.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generador import cargar, armar_stream, llave, hash_universal, SEED, RESULTS

VIAJE = 600                      # segundos
BITS_POR_ELEMENTO = [4, 8, 12, 16]
SEMILLAS = 3
PRUEBAS_LLENO = 50_000           # tarjetas nunca insertadas para medir con el filtro lleno


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


def correr(m, k, semilla, ventanas, tarjetas):
    bloom = Bloom(m, k, np.random.default_rng(semilla))
    vistas, actual = set(), None
    nuevas = falsos_pos = dup = dup_detectados = teorico = 0

    for v, card in zip(ventanas, tarjetas):
        if v != actual:                 # ventana nueva
            bloom.vaciar()
            vistas, actual = set(), v
        x = llave(card)
        marcado = x in bloom

        if card in vistas:              # duplicado de verdad
            dup += 1
            dup_detectados += marcado
        else:
            nuevas += 1
            teorico += fpr_teorico(m, k, len(vistas))
            falsos_pos += marcado       # nueva pero el filtro dijo que ya estaba
            vistas.add(card)
            bloom.agregar(x)

    return falsos_pos / nuevas, teorico / nuevas, dup, dup_detectados


def fpr_lleno(m, k, n, semilla):
    # inserta n tarjetas y prueba con tarjetas que no están (ids C010000 en adelante, no existen en el dataset)
    bloom = Bloom(m, k, np.random.default_rng(semilla))
    dentro = [llave("C%06d" % i) for i in range(1, n + 1)]
    for x in dentro:
        bloom.agregar(x)
    assert all(x in bloom for x in dentro)      # no hay falsos negativos
    fuera = [llave("C%06d" % i) for i in range(10_000, 10_000 + PRUEBAS_LLENO)]
    return sum(x in bloom for x in fuera) / len(fuera)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    stream = armar_stream(cargar("100k"))
    ventanas = stream["sec"].to_numpy() // VIAJE
    tarjetas = stream["card_id"].tolist()

    # n de diseño: máximo de tarjetas distintas en una ventana
    n = int(stream.assign(v=ventanas).groupby("v")["card_id"].nunique().max())
    print("máximo de tarjetas distintas por ventana:", n)

    filas = []
    for bits in BITS_POR_ELEMENTO:
        m = bits * n
        k = max(1, round(bits * np.log(2)))
        corridas = [correr(m, k, SEED + s, ventanas, tarjetas) for s in range(SEMILLAS)]
        dup, detectados = corridas[0][2], corridas[0][3]
        assert all(c[2] == c[3] for c in corridas), "un Bloom filter no puede dejar pasar duplicados"
        filas.append({
            "bits_por_elemento": bits,
            "m": m,
            "k": k,
            "fpr_stream_empirica": round(float(np.mean([c[0] for c in corridas])), 5),
            "fpr_stream_teorica": round(float(np.mean([c[1] for c in corridas])), 5),
            "fpr_lleno_empirica": round(float(np.mean([fpr_lleno(m, k, n, SEED + s) for s in range(SEMILLAS)])), 5),
            "fpr_lleno_teorica": round(fpr_teorico(m, k, n), 5),
            "duplicados_reales": dup,
            "duplicados_detectados": detectados,
        })
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for eje, tipo, titulo in ((ejes[0], "lleno", "filtro lleno (n tarjetas)"), (ejes[1], "stream", "en el stream")):
        eje.plot(tabla["bits_por_elemento"], tabla["fpr_%s_empirica" % tipo], "o-", label="empírica")
        eje.plot(tabla["bits_por_elemento"], tabla["fpr_%s_teorica" % tipo], "s--", label="teórica")
        eje.set_yscale("log")
        eje.set_xlabel("bits por elemento (m / n)")
        eje.set_title(titulo)
        eje.legend()
    ejes[0].set_ylabel("falsos positivos")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "w2_task2_bloom_fpr.png"), dpi=150)
    plt.close()

    salida = {"w2_task2_bloom_filter": {"viaje_s": VIAJE, "n_diseno": n, "semillas": SEMILLAS,
                                        "resultados": tabla.to_dict("records")}}
    with open(os.path.join(RESULTS, "w2_task2.json"), "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
