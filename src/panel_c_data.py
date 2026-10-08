"""
panel_c_data.py - agrega los taps por ruta y por hora para el Panel C del dashboard.

Respeta la "regla de oro" del equipo: el dashboard NO lee el CSV, solo results/*.json.
Este script lee data/taps.csv y escribe results/panel_c.json con:

  meta   : fuente, número de taps, periodo simulado, días, definición de "pasajeros"
  routes : por ruta -> pasajeros (taps), tarjetas distintas, % del total
  hourly : por ruta -> lista de 24 valores = taps totales de la ruta en esa hora del día
           (sumados sobre todo el periodo; el dashboard divide por `days` para el promedio diario)

"Pasajeros" = número de taps (cada tap es una validación/abordaje).

Uso:
    python src/panel_c_data.py
    python src/panel_c_data.py --taps data/taps.csv --out results/panel_c.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAPS_CANDIDATES = [ROOT / "data" / "taps.csv", ROOT / "data" / "taps_10k.csv"]
DEFAULT_OUT = ROOT / "results" / "panel_c.json"


def find_taps(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
        sys.exit(f"No encuentro el archivo de taps: {path}")
    for cand in DEFAULT_TAPS_CANDIDATES:
        if cand.exists():
            return cand
    sys.exit(
        "No encuentro data/taps.csv. Genera los datos primero (python src/generator.py) "
        "o pasa la ruta con --taps."
    )


def build(taps_path: Path) -> dict:
    taps = pd.read_csv(taps_path, usecols=["timestamp", "card_id", "route_id"])
    ts = pd.to_datetime(taps["timestamp"])
    taps["hour"] = ts.dt.hour
    days = int(ts.dt.date.nunique())

    by_route = taps.groupby("route_id").agg(passengers=("route_id", "size"), unique_cards=("card_id", "nunique"))
    total = int(by_route["passengers"].sum())
    by_route = by_route.sort_values("passengers", ascending=False)

    routes = [
        {
            "route_id": route_id,
            "passengers": int(row.passengers),
            "unique_cards": int(row.unique_cards),
            "share_pct": round(100.0 * int(row.passengers) / total, 2),
        }
        for route_id, row in by_route.iterrows()
    ]

    grid = taps.groupby(["route_id", "hour"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    hourly = {route_id: [int(v) for v in grid.loc[route_id]] for route_id in by_route.index}

    return {
        "meta": {
            "source": taps_path.name,
            "n_taps": int(len(taps)),
            "n_routes": int(len(routes)),
            "period_start": str(ts.min()),
            "period_end": str(ts.max()),
            "days": days,
            "metric": "pasajeros = número de taps (validaciones) de la ruta",
            "hourly_note": "hourly[ruta][h] = taps totales de la ruta en la hora h, sumados sobre todos los días",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "routes": routes,
        "hourly": hourly,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Agregados por ruta y hora para el Panel C")
    ap.add_argument("--taps", help="CSV de taps (por defecto data/taps.csv)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON de salida (por defecto results/panel_c.json)")
    args = ap.parse_args()

    result = build(find_taps(args.taps))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    m = result["meta"]
    top = result["routes"][0]
    print(f"{m['n_taps']:,} taps, {m['n_routes']} rutas, {m['days']} días ({m['period_start']} -> {m['period_end']})")
    print(f"Ruta más concurrida: {top['route_id']} con {top['passengers']:,} pasajeros ({top['share_pct']}%)")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
