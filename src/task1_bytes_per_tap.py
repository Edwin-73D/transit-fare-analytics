"""
task1_bytes_per_tap.py - Complemento del Task 1 (W1): cuánto pesa un tap de verdad.
==================================================================================

El Task 1 (src/task1_estimates.md) supone que un registro pesa 200 bytes (supuesto S5).
Este script lo MIDE con el dataset generado, de dos formas:
  - CSV:  tamaño del archivo / número de taps (formato compacto, sin nombres de campo);
  - JSON: largo promedio de un tap escrito como objeto JSON (con nombres de campo), que es
          como viajaría un evento por un stream.

Idea y método: Felipe (w1_task1_big_data_context.py en su repo). Por decisión del equipo
(2026-10-23) el Task 1 se queda con la estimación teórica de Edwin con cifras reales del
Metro, y de la versión de Felipe se toma esta medición para respaldar el supuesto S5.

Uso
---
    python src/task1_bytes_per_tap.py                      # data/taps_100k.csv (o taps.csv, taps_10k.csv)
    python src/task1_bytes_per_tap.py --taps otro.csv
Salida: results/task1_bytes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps_100k.csv", ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]
DEFAULT_OUT = ROOT / "results" / "task1_bytes.json"
SAMPLE = 2000          # taps que se escriben como JSON para medir el promedio
ASSUMED_BYTES = 200    # supuesto S5 del Task 1


def find_taps(arg: str | None) -> Path:
    """Ruta del CSV: la que se pasa con --taps o la primera de las rutas por defecto que exista."""
    if arg:
        return Path(arg)
    for path in DEFAULT_TAPS_CANDIDATES:
        if path.is_file():
            return path
    sys.exit("No encuentro el CSV de taps. Genera los datos primero (python src/generator.py).")


def bytes_per_tap(path: Path) -> dict:
    """Bytes por tap en CSV (archivo / filas) y en JSON (promedio sobre una muestra)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    csv_bytes = path.stat().st_size / len(df)
    sample = df.head(SAMPLE).to_dict("records")
    json_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode("utf-8")) for r in sample) / len(sample)
    return {"source": path.name, "n_taps": len(df), "csv": round(csv_bytes, 1),
            "json": round(json_bytes, 1), "assumed_S5": ASSUMED_BYTES,
            "json_vs_assumed": round(json_bytes / ASSUMED_BYTES, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Mide los bytes por tap del dataset (complemento del Task 1)")
    ap.add_argument("--taps", help="CSV de taps")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    result = bytes_per_tap(find_taps(args.taps))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{result['source']} ({result['n_taps']:,} taps): CSV {result['csv']} B/tap, "
          f"JSON {result['json']} B/tap (supuesto S5 = {ASSUMED_BYTES} B)")
    print(f"Escrito {out}")


if __name__ == "__main__":
    main()
