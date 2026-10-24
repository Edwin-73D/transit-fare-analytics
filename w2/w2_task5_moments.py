"""
w2_task5_moments.py - Task 5 (W2): momentos de frecuencia F0, F1, F2 sobre route_id.
====================================================================================

Pregunta: ¿los pasajeros están repartidos entre muchas rutas o unas pocas cargan casi todo?

Teoría
------
Sea m_i = número de taps de la ruta i. El momento k es  F_k = suma de m_i^k:
  F0 = número de rutas distintas (las que tienen al menos un tap)
  F1 = total de taps
  F2 = suma de los cuadrados -> mide concentración.
Al elevar al cuadrado, una ruta con el doble de pasajeros aporta 4 veces más a F2, así que
F2 crece cuando pocas rutas se llevan casi todo. Para leerlo se compara con los extremos
posibles, con los mismos F0 y F1:
  F2_uniform = F1^2 / F0   (todas las rutas con el mismo número de taps: mínimo posible)
  F2_max     = F1^2        (todos los taps en una sola ruta: máximo posible)
  effective_routes = F1^2 / F2: "es como si los taps se repartieran parejo entre esta
  cantidad de rutas". Vale F0 si el reparto es parejo y 1 si todo está en una ruta.

Cómo se calcula en stream (una pasada, memoria limitada)
--------------------------------------------------------
Se guarda un contador por ruta (memoria O(F0) = O(número de rutas)). Cuando llega un tap
de una ruta que llevaba c taps y pasa a c + 1:
  F1 += 1;   F0 += 1 si c == 0;   F2 += (c + 1)^2 - c^2 = 2c + 1
Cada tap cuesta O(1): no se recalcula la suma completa.

Decisiones (y por qué)
---------------------
1. Conteo exacto sobre route_id: con decenas de rutas un contador por ruta cabe de sobra.
   Sobre card_id (millones de tarjetas en el Metro real) la memoria crecería con el número
   de tarjetas; ahí se usarían estimadores aproximados de F2 (p. ej. AMS).
2. Los taps se cuentan TAL CUAL llegan por el stream, incluidos los duplicados. Dato: el
   generador v2 inyecta ~2% de taps duplicados (misma tarjeta, mismo vehículo, segundos
   después). Eso infla un poco F1 y F2; F0 no cambia, porque las rutas son las mismas.
3. Se guarda una foto de F0, F1 y F2 cada EVERY taps (y siempre la del último tap), para
   ver cómo evolucionan a lo largo del stream.
4. Al final se recalcula F2 sumando los cuadrados de los contadores, para comprobar que la
   fórmula incremental 2c + 1 dio exactamente lo mismo.
5. top_routes trae TODAS las rutas ordenadas de mayor a menor (con su % de F1 y de F2):
   sirve también para el Panel C (pasajeros por ruta).

Uso
---
    python w2/w2_task5_moments.py                     # data/taps.csv o data/taps_10k.csv
    python w2/w2_task5_moments.py --every 1000 --taps data/taps.csv
Salida: results/w2_task5.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from stream import describe, stream_taps

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "w2_task5.json"

EVERY = 500  # cada cuántos taps se guarda una foto de la serie (decisión 3)


class FrequencyMoments:
    """Contadores por ruta y F0, F1, F2 actualizados en O(1) por cada tap."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.f0 = 0
        self.f1 = 0
        self.f2 = 0

    def add(self, route: str) -> None:
        """Procesa un tap de la ruta: c -> c + 1 y actualiza los momentos sin recalcular todo."""
        c = self.counts.get(route, 0)
        self.counts[route] = c + 1
        self.f1 += 1
        if c == 0:
            self.f0 += 1          # ruta nueva
        self.f2 += 2 * c + 1      # (c + 1)^2 - c^2

    def effective_routes(self) -> float:
        """F1^2 / F2: número de rutas entre las que el reparto 'equivale' a uno parejo."""
        return self.f1 ** 2 / self.f2 if self.f2 else 0.0

    def snapshot(self, seq: int) -> dict:
        """Foto de los momentos en este punto del stream."""
        return {"stream_seq": seq, "F0": self.f0, "F1": self.f1, "F2": self.f2,
                "effective_routes": round(self.effective_routes(), 3)}


def run(taps_path: str | None, every: int) -> tuple[FrequencyMoments, list[dict]]:
    """Una sola pasada por el stream: actualiza los momentos y guarda la serie cada `every` taps."""
    fm = FrequencyMoments()
    series = []
    seq = 0
    for tap in stream_taps(taps_path):
        fm.add(tap.route_id)
        seq = tap.stream_seq
        if fm.f1 % every == 0:
            series.append(fm.snapshot(seq))
    if not series or series[-1]["F1"] != fm.f1:  # la foto final, si no cayó en un múltiplo
        series.append(fm.snapshot(seq))
    return fm, series


def top_routes(fm: FrequencyMoments) -> list[dict]:
    """Rutas de mayor a menor con su conteo y su participación en F1 (pasajeros) y en F2."""
    return [
        {"route_id": r, "count": c,
         "share_of_F1": round(c / fm.f1, 4),
         "share_of_F2": round(c * c / fm.f2, 4)}
        for r, c in sorted(fm.counts.items(), key=lambda kv: kv[1], reverse=True)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 5 (W2): F0, F1, F2 exactos sobre route_id en una pasada")
    ap.add_argument("--taps", help="CSV de taps (por defecto data/taps.csv o data/taps_10k.csv)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida (por defecto results/w2_task5.json)")
    ap.add_argument("--every", type=int, default=EVERY, help=f"foto de la serie cada N taps (por defecto {EVERY})")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    fm, series = run(args.taps, args.every)

    # Comprobación (decisión 4): la fórmula incremental debe coincidir con la suma directa.
    f2_direct = sum(c * c for c in fm.counts.values())
    assert f2_direct == fm.f2, f"F2 incremental {fm.f2} != F2 directo {f2_direct}"

    f2_uniform = fm.f1 ** 2 / fm.f0
    f2_max = fm.f1 ** 2
    routes = top_routes(fm)
    result = {
        "meta": {
            "task": "W2 Task 5 - momentos de frecuencia F0, F1, F2 sobre route_id",
            "source": describe(args.taps),
            "every": args.every,
            "duplicates": "se cuentan tal cual llegan (el generador v2 inyecta ~2% de duplicados)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "F0": fm.f0,
        "F1": fm.f1,
        "F2": fm.f2,
        "F2_uniform": round(f2_uniform, 3),
        "F2_max": f2_max,
        "F2_over_uniform": round(fm.f2 / f2_uniform, 3),   # 1 = reparto parejo
        "effective_routes": round(fm.effective_routes(), 3),
        "incremental_check": f2_direct == fm.f2,
        "top_routes": routes,
        "series": series,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Stream: {result['meta']['source']}, una sola pasada, foto cada {args.every} taps\n")
    print(f"F0 = {fm.f0} rutas | F1 = {fm.f1:,} taps | F2 = {fm.f2:,}")
    print(f"F2 parejo (F1²/F0) = {f2_uniform:,.0f} | F2 máximo (F1²) = {f2_max:,}")
    print(f"F2 / F2 parejo = {result['F2_over_uniform']}x | rutas efectivas (F1²/F2) = "
          f"{result['effective_routes']} de {fm.f0}")
    top5 = routes[:5]
    print("Top 5: " + ", ".join(f"{r['route_id']} {r['share_of_F2'] * 100:.1f}% de F2" for r in top5)
          + f" | juntas: {sum(r['share_of_F1'] for r in top5) * 100:.1f}% de los taps, "
            f"{sum(r['share_of_F2'] for r in top5) * 100:.1f}% de F2")
    print(f"F2 incremental = F2 directo: {result['incremental_check']}")
    print("\nSerie (algunas fotos):")
    picks = series[:: max(1, len(series) // 5)]
    if picks[-1] is not series[-1]:
        picks.append(series[-1])  # siempre mostrar la foto final
    for s in picks:
        print(f"  seq {s['stream_seq']:>6}: F0 {s['F0']:>3}  F1 {s['F1']:>7,}  F2 {s['F2']:>12,}  "
              f"rutas efectivas {s['effective_routes']}")
    print(f"\nEscrito {out_path}")


if __name__ == "__main__":
    main()
