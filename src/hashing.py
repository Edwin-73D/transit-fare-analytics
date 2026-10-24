"""
hashing.py - Familia de hash universal compartida por todo el proyecto.
======================================================================

Autor original: Felipe (estaba dentro de generador.py en su repo, Juan319-u/logica-3).
Se movió a un módulo propio (decisión del equipo, 2026-10-23) para que la usen los tasks de
src/, w2/ y w3/ sin depender del generador: hashing universal del W1 (Task 3), las k funciones
del Bloom filter y el muestreo por hash del W2. El código de las funciones no cambió.

Teoría
------
Familia de Carter-Wegman:  h(k) = ((a*k + b) mod p) mod m
  - p primo mayor que cualquier llave (aquí p = 2^61 - 1, primo de Mersenne);
  - a elegido al azar en {1, ..., p-1} y b en {0, ..., p-1};
  - para dos llaves distintas, P(h(x) = h(y)) <= 1/m: es universal.
Por eso las colisiones esperadas entre n llaves son a lo sumo C(n, 2)/m.

Llave numérica de un card_id: los bytes del texto leídos como un entero ("C000123" son 7
bytes -> menos de 2^56 < p), así que dos tarjetas distintas siempre dan llaves distintas.

Uso
---
    from hashing import llave, hash_universal
    h = hash_universal(1024)          # una función al azar de la familia, con m = 1024
    h(llave("C000123"))               # bucket en 0..1023
"""

from __future__ import annotations

import numpy as np

SEED = 42
P = 2**61 - 1                      # primo de Mersenne, mayor que cualquier llave

# generador global para los hash aleatorios (reproducible con la semilla del proyecto)
rng = np.random.default_rng(SEED)


def llave(card_id: str) -> int:
    """'C000123' -> entero, con los bytes del texto."""
    return int.from_bytes(card_id.encode(), "big")


def hash_universal(m: int, gen: np.random.Generator | None = None):
    """Función al azar de la familia h(k) = ((a*k + b) mod p) mod m, con a y b al azar."""
    gen = gen if gen is not None else rng
    a = int(gen.integers(1, P))
    b = int(gen.integers(0, P))
    return lambda k: ((a * k + b) % P) % m
