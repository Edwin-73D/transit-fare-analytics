"""
stream.py - Simulador del stream de taps (Workshop 2).
=====================================================

Hace el papel de los torniquetes y validadores: entrega los taps UNO A LA VEZ, en orden de
stream_seq, como llegarían en vivo. Todos los tasks del W2 leen los datos SOLO a través de
este módulo, así todos procesan exactamente el mismo stream en el mismo orden.

Modelo de stream
----------------
- Los datos llegan como una secuencia a_1, a_2, a_3, ... que no se puede rebobinar.
- Cada elemento se ve UNA vez (una sola pasada) y después se pierde, salvo lo que el
  algoritmo decida guardar en su resumen (memoria limitada, mucho menor que el stream).
- Por eso aquí se usa un generador (yield): el consumidor recibe un tap, lo procesa y pide
  el siguiente; nunca tiene la lista completa.

Cómo se usa (desde cualquier script de w2/)
-------------------------------------------
    from stream import stream_taps

    for tap in stream_taps():          # data/taps.csv o data/taps_10k.csv
        tap.stream_seq, tap.timestamp, tap.card_id, tap.route_id, tap.vehicle_id, ...

Decisiones (y por qué)
---------------------
1. Si el CSV trae stream_seq (generador v2), se lee línea por línea con el módulo csv:
   memoria O(1), un stream de verdad. Se verifica que stream_seq venga creciente.
2. Si NO trae stream_seq (CSV del W1, p. ej. taps_10k.csv), se ordena por timestamp y se
   numera 1, 2, 3, ... Ordenar exige cargar el archivo completo, pero eso ocurre en el
   simulador (la "fuente"), no en los tasks: la regla de memoria limitada es para quien
   consume el stream. Sirve para probar mientras llega el generador v2.
3. vehicle_id es None si el CSV no lo trae (CSV del W1).
4. Cada tap es un objeto Tap con los tipos ya convertidos (int, datetime, float, bool),
   para que ningún task tenga que convertirlos por su cuenta.
5. Mismo orden de búsqueda del CSV que el W1: data/taps.csv y si no, data/taps_10k.csv.

Uso desde la terminal (para ver qué entrega)
--------------------------------------------
    python w2/stream.py              # muestra los primeros 5 taps y cuenta el total
    python w2/stream.py --head 10 --taps data/taps.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]


@dataclass(frozen=True, slots=True)
class Tap:
    """Un tap del stream, con los tipos ya convertidos."""

    stream_seq: int
    tap_id: str
    timestamp: datetime
    card_id: str
    route_id: str
    station: str
    fare_amount: float
    transfer_flag: bool
    vehicle_id: str | None  # None en los CSV del W1, que no traen este campo


def find_taps(arg: str | Path | None = None) -> Path:
    """Ruta del CSV: la que se pasa o la primera de las rutas por defecto que exista."""
    if arg:
        path = Path(arg)
        if not path.is_file():
            sys.exit(f"No encuentro el archivo de taps: {path}")
        return path
    for path in DEFAULT_TAPS_CANDIDATES:
        if path.is_file():
            return path
    sys.exit("No encuentro data/taps.csv. Genera los datos primero (python src/generator.py).")


def to_tap(row: dict, seq: int) -> Tap:
    """Convierte una fila del CSV (todo texto) en un Tap con tipos."""
    return Tap(
        stream_seq=seq,
        tap_id=row["tap_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        card_id=row["card_id"],
        route_id=row["route_id"],
        station=row["station"],
        fare_amount=float(row["fare_amount"]),
        transfer_flag=str(row["transfer_flag"]).strip().lower() in ("true", "1"),
        vehicle_id=row.get("vehicle_id") or None,
    )


def _stream_with_seq(path: Path) -> Iterator[Tap]:
    """CSV con stream_seq: lectura línea por línea (memoria O(1)), verificando el orden."""
    last = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = int(row["stream_seq"])
            if seq <= last:
                raise ValueError(f"stream_seq no es creciente: {seq} después de {last} ({path.name})")
            last = seq
            yield to_tap(row, seq)


def _stream_by_timestamp(path: Path) -> Iterator[Tap]:
    """CSV sin stream_seq: ordena por timestamp y numera 1, 2, 3, ... (solo para pruebas)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    order = pd.to_datetime(df["timestamp"], format="ISO8601").argsort(kind="stable")
    for seq, row in enumerate(df.iloc[order].to_dict("records"), start=1):
        yield to_tap(row, seq)


def stream_taps(path: str | Path | None = None, limit: int | None = None) -> Iterator[Tap]:
    """Entrega los taps uno a la vez en orden de stream_seq. limit = parar tras n taps."""
    path = find_taps(path)
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    source = _stream_with_seq(path) if "stream_seq" in header else _stream_by_timestamp(path)
    for i, tap in enumerate(source, start=1):
        if limit is not None and i > limit:
            return
        yield tap


def describe(path: str | Path | None = None) -> str:
    """Una línea que dice de dónde sale el stream y cómo se ordena (para los mensajes de los tasks)."""
    path = find_taps(path)
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    how = "stream_seq del CSV" if "stream_seq" in header else "stream_seq asignado por timestamp"
    return f"{path.name} ({how})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulador del stream de taps (W2)")
    ap.add_argument("--taps", help="CSV de taps (por defecto data/taps.csv o data/taps_10k.csv)")
    ap.add_argument("--head", type=int, default=5, help="cuántos taps mostrar (por defecto 5)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    print(f"Stream: {describe(args.taps)}\n")
    n, last = 0, None
    for tap in stream_taps(args.taps):
        if n < args.head:
            print(tap)
        n += 1
        last = tap
    print(f"\n{n:,} taps entregados; último stream_seq = {last.stream_seq}, timestamp = {last.timestamp}")


if __name__ == "__main__":
    main()
