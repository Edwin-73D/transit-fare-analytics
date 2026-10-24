"""
w2_task3_sampling.py - Task 3 (W2): muestreo estable por hash (EXTENDIDO: tarjetas distintas).
=============================================================================================

Task de Pedro (C). Como no se había subido, se escribió durante la integración (2026-10-23).

Preguntas:
  1. ¿Qué fracción de los taps son transferencias? (estimada con una muestra)
  2. EXTENDIDO: ¿cuántas tarjetas distintas hay? (estimada con una muestra)

Muestreo estable por hash
-------------------------
Una tarjeta entra a la muestra si  h(card_id) mod 100 < 100 * p,  con h de la familia universal
del W1 (src/hashing.py). Como h es una función, la MISMA tarjeta siempre entra o siempre queda
fuera: la muestra guarda TODOS los taps de las tarjetas elegidas (muestra "por tarjeta", no por
tap). Por eso sirve para preguntas sobre tarjetas, como cuántas distintas hay.

Estimadores y teoría
--------------------
- Fracción de transferencias: taps con transfer_flag en la muestra / taps en la muestra.
- Tarjetas distintas:  D_est = (tarjetas distintas en la muestra) / p.
  Si cada tarjeta entra con prob. p, las distintas de la muestra son Binomial(D, p):
    E[D_est] = D (insesgado)   y   error relativo típico = sqrt((1 - p) / (p * D)).
  Más p -> menos error, pero más memoria: se guardan ~p*D tarjetas en vez de D.
  Esta gráfica "error vs p" reemplaza a "error vs memoria" del HyperLogLog (Task 4 eliminado).

Decisiones (y por qué)
---------------------
1. p = 5%, 10% y 20% (los del enunciado). Con la misma función h las muestras quedan anidadas:
   la de 5% está dentro de la de 10%, y esa dentro de la de 20%.
2. R = 200 funciones hash distintas (semillas) para medir el error de verdad; una sola puede
   salir con suerte. La serie del Panel B usa la primera función.
   Hallazgo (2026-10-23): con R = 30 una sola función "mala" (estimó 4.510 en vez de 3.000 con
   p = 10%) dominaba el error. La familia universal es independiente POR PARES: eso garantiza
   que la varianza PROMEDIO sobre las funciones es la de la teoría, pero no evita que alguna
   función rara falle mucho con llaves muy ordenadas (C000001, C000002, ...), mientras la
   mayoría falla MENOS que el azar. Por eso se reportan media (RMSE), mediana y máximo.
3. Una sola pasada por w2/stream.py con todos los muestreadores a la vez. El conteo exacto
   (un set con todas las tarjetas) solo sirve de referencia para medir el error.
4. h se calcula en cada tap (no se guarda en un diccionario por tarjeta): así la memoria del
   algoritmo es solo la de la muestra.
5. Se cuentan los taps tal cual llegan, incluidos los ~2% de duplicados del generador v2.

Uso
---
    python w2/w2_task3_sampling.py
Salida: results/w2_panel_b.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))   # para importar src/hashing.py

from hashing import hash_universal, llave       # noqa: E402
from stream import describe, find_taps, stream_taps  # noqa: E402

RESULTS = os.path.join(RAIZ, "results")
PS = [0.05, 0.10, 0.20]          # decisión 1
R = 200                          # funciones hash distintas (decisión 2)
SEED = 42
CADA = 1000                      # una foto de la serie cada CADA taps
KEY = {0.05: "est_p05", 0.10: "est_p10", 0.20: "est_p20"}


class HashSampler:
    """Una función hash h y, para cada p, la muestra de tarjetas y sus contadores de taps."""

    def __init__(self, gen: np.random.Generator) -> None:
        self.h = hash_universal(100, gen)                    # h(k) en 0..99
        self.cards = {p: set() for p in PS}                  # tarjetas en la muestra
        self.taps = {p: 0 for p in PS}
        self.transfers = {p: 0 for p in PS}

    def add(self, key: int, card: str, transfer: bool) -> None:
        """Un tap: entra a las muestras cuyo umbral 100*p supera h(card) (decisión 1)."""
        b = self.h(key)
        for p in PS:
            if b < 100 * p:
                self.cards[p].add(card)
                self.taps[p] += 1
                self.transfers[p] += transfer

    def distinct_est(self, p: float) -> float:
        """Tarjetas distintas estimadas: distintas en la muestra / p."""
        return len(self.cards[p]) / p

    def transfer_frac(self, p: float) -> float:
        """Fracción de transferencias entre los taps de la muestra."""
        return self.transfers[p] / self.taps[p] if self.taps[p] else 0.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS, exist_ok=True)
    path = find_taps()
    gen = np.random.default_rng(SEED)
    samplers = [HashSampler(gen) for _ in range(R)]

    exact_cards: set[str] = set()       # referencia exacta (decisión 3)
    total_taps = total_transfers = 0
    series = []
    for tap in stream_taps(path):       # una sola pasada (decisión 3)
        key = llave(tap.card_id)        # h se calcula en cada tap (decisión 4)
        for s in samplers:
            s.add(key, tap.card_id, tap.transfer_flag)
        exact_cards.add(tap.card_id)
        total_taps += 1
        total_transfers += tap.transfer_flag
        if tap.stream_seq % CADA == 0:
            first = samplers[0]
            series.append({"stream_seq": tap.stream_seq, "true": len(exact_cards),
                           **{KEY[p]: round(first.distinct_est(p), 1) for p in PS}})

    d = len(exact_cards)
    true_frac = total_transfers / total_taps
    error_vs_p, by_p, frac_err = {}, {}, {}
    for p in PS:
        ests = [s.distinct_est(p) for s in samplers]
        rel = [(e - d) / d for e in ests]
        error_vs_p[f"{p:.2f}"] = {
            "mean_estimate": round(statistics.fmean(ests), 1),
            "mean_abs_rel_error": round(statistics.fmean(abs(x) for x in rel), 5),
            "rmse_rel": round(math.sqrt(statistics.fmean(x * x for x in rel)), 5),
            "median_abs_rel_error": round(statistics.median(abs(x) for x in rel), 5),
            "max_abs_rel_error": round(max(abs(x) for x in rel), 5),
            "share_worse_than_3x_theory": round(sum(abs(x) > 3 * math.sqrt((1 - p) / (p * d)) for x in rel) / R, 4),
            "theory_rel_error": round(math.sqrt((1 - p) / (p * d)), 5),
            "bias_rel": round(statistics.fmean(rel), 5),
            "memory_cards_mean": round(statistics.fmean(len(s.cards[p]) for s in samplers), 1),
        }
        fracs = [s.transfer_frac(p) for s in samplers]
        by_p[f"{p:.2f}"] = round(statistics.fmean(fracs), 5)
        frac_err[f"{p:.2f}"] = round(statistics.fmean(abs(f - true_frac) for f in fracs), 5)

    result = {
        "transfer_frac": {"true": round(true_frac, 5), "by_p": by_p, "mean_abs_error_by_p": frac_err},
        "distinct_series": series,
        "error_vs_p": error_vs_p,
        "true_distinct": d,
        "meta": {
            "task": "W2 Task 3 - muestreo estable por hash (extendido: tarjetas distintas)",
            "source": describe(path),
            "rule": "la tarjeta entra si h(card_id) mod 100 < 100 p (h de la familia universal)",
            "estimator": "distintas = distintas en la muestra / p",
            "theory": "error relativo ~ sqrt((1 - p) / (p D)), estimador insesgado",
            "hash_functions": R,
            "series_uses": "la primera función hash",
        },
    }
    out = os.path.join(RESULTS, "w2_panel_b.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Stream: {describe(path)}, {R} funciones hash, una sola pasada")
    print(f"Tarjetas distintas reales: {d:,} | fracción real de transferencias: {true_frac:.4f}\n")
    print(f"{'p':>5} {'D est. (media)':>15} {'RMSE rel.':>10} {'teoría':>8} {'mediana':>8} {'máximo':>8} "
          f"{'memoria (tarjetas)':>19} {'transfer est.':>14}")
    for p in PS:
        e = error_vs_p[f"{p:.2f}"]
        print(f"{p:>5.2f} {e['mean_estimate']:>15,.1f} {e['rmse_rel']:>10.4f} {e['theory_rel_error']:>8.4f} "
              f"{e['median_abs_rel_error']:>8.4f} {e['max_abs_rel_error']:>8.4f} "
              f"{e['memory_cards_mean']:>19,.1f} {by_p[f'{p:.2f}']:>14.4f}")
    print(f"\nEscrito {out}")


if __name__ == "__main__":
    main()
