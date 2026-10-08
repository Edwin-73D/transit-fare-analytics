"""
dashboard.py - Dashboard del Grupo 8 (Public Transit Smart Card Fare Analytics)
===============================================================================

Tres paneles. El dashboard SOLO lee results/*.json (regla de oro del equipo): cada
módulo escribe su JSON y aquí nadie toca el CSV ni importa código de los otros.

  Panel A  results/panel_a.json    (src/task3_hashing.py, compañero 1)
           Carga por bucket sobre card_id: una función hash vs power of two choices (P2C).
  Panel B  results/panel_b.json    (src/task2_quicksort.py, compañero 2)
           Tiempo y comparaciones: QuickSort aleatorizado vs determinista, según n.
  Panel C  results/panel_c.json    (src/panel_c_data.py, compañero 2)
           Pasajeros por ruta en el periodo simulado, con las más concurridas resaltadas.
           Opcional: results/task5_markov.json (Task 5) para dibujar la línea de capacidad.

Cómo correrlo
-------------
    streamlit run dashboard.py            # abre el dashboard en el navegador
    python dashboard.py --export          # guarda las gráficas como PNG en results/capturas/
    RESULTS_DIR=otra/carpeta streamlit run dashboard.py     # leer JSON de otra carpeta (pruebas)

Formato esperado de results/panel_a.json (lo escribe el compañero 1)
-------------------------------------------------------------------
    {
      "n_keys": 3000, "n_buckets": 3000,                              (opcionales)
      "single_hash": {"max_load": 9, "std": 1.7, "histogram": {"0": 1100, "1": 1090, ...}},
      "p2c":         {"max_load": 5, "std": 0.9, "histogram": {"0": 400,  "1": 1500, ...}}
    }
  "histogram" = cuántos buckets tienen exactamente k elementos (dict {k: buckets} o lista
  indexada por k). También se aceptan nombres alternativos (ver _ALIASES) y, en lugar del
  histograma, la lista "bucket_loads" con la carga de cada bucket. Si falta max_load o std,
  se calculan a partir del histograma.

Diseño de las gráficas
----------------------
Un solo color de énfasis por rol: azul = la técnica aleatorizada / P2C; naranja = la línea
base (determinista / una sola función). Paleta categórica validada (daltonismo) y marcas finas.
Cada panel tiene una tabla con los mismos datos (vista de tabla) para no depender del color.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: Streamlit muestra la figura, o se guarda como PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
CAPTURAS = RESULTS / "capturas"

TASK5_CANDIDATES = ("task5_markov.json", "task5.json", "panel_task5.json")

# --------------------------------------------------------------------------- #
# Estilo (superficie clara; paleta categórica validada con el script de dataviz)
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # BLUE = aleatorizado / P2C ; ORANGE = línea base
TOP5 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#c3c2b7"  # "las demás rutas" (énfasis: resaltar unas pocas, atenuar el resto)

RC = {
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "regular",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def themed(fn):
    """Dibuja la figura con el estilo del dashboard sin tocar la configuración global."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC):
            return fn(*args, **kwargs)

    return wrapper


class DataError(Exception):
    """Un JSON existe pero no tiene el formato esperado (se muestra como mensaje, no como traza)."""


def es(x: float, decimals: int = 0) -> str:
    """Formato numérico en español: 1.234,5"""
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def _title(fig, text: str, width: int = 100) -> None:
    """Título-conclusión de la figura (tight_layout ya le reserva el espacio)."""
    fig.suptitle("\n".join(textwrap.wrap(text, width)), x=0.01, ha="left", fontsize=12, color=INK)


# --------------------------------------------------------------------------- #
# Lectura de resultados
# --------------------------------------------------------------------------- #
def load_json(name: str):
    """None si el archivo no existe; DataError si no es JSON válido."""
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DataError(f"{name} no es un JSON válido ({e}).") from e


def load_task5():
    for name in TASK5_CANDIDATES:
        data = load_json(name)
        if data is not None:
            return data
    return None


# --- Panel A ----------------------------------------------------------------
_ALIASES = {
    "single": ("single_hash", "single", "one_hash", "one_function", "chaining", "baseline", "una_funcion"),
    "p2c": ("p2c", "power_of_two_choices", "two_choices", "two_hash", "dos_opciones"),
}
_HIST_KEYS = ("histogram", "hist", "histograma", "load_histogram")
_LOADS_KEYS = ("bucket_loads", "loads", "cargas")
_MAX_KEYS = ("max_load", "carga_maxima", "max")
_STD_KEYS = ("std", "stdev", "std_dev", "desviacion")


def _first(d: dict, names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def _method_stats(m: dict, label: str) -> dict:
    hist = _first(m, _HIST_KEYS)
    if hist is None:
        loads = _first(m, _LOADS_KEYS)
        if loads is None:
            raise DataError(
                f"panel_a.json: la sección '{label}' no trae un histograma "
                f"(busqué {', '.join(_HIST_KEYS)}) ni la lista de cargas por bucket ({', '.join(_LOADS_KEYS)})."
            )
        hist = np.bincount(np.asarray(loads, dtype=int)).tolist()
    items = hist.items() if isinstance(hist, dict) else enumerate(hist)
    counts = {int(k): int(v) for k, v in items}

    loads_expanded = np.repeat(list(counts.keys()), list(counts.values()))
    max_load = _first(m, _MAX_KEYS)
    std = _first(m, _STD_KEYS)
    return {
        "counts": counts,
        "max_load": int(max_load) if max_load is not None else int(max(k for k, v in counts.items() if v > 0)),
        "std": float(std) if std is not None else float(np.std(loads_expanded)),
        "n_buckets": int(sum(counts.values())),
    }


def parse_panel_a(data: dict) -> dict:
    scopes = [data] + [data[k] for k in ("results", "methods", "strategies") if isinstance(data.get(k), dict)]
    parsed = {}
    for key, names in _ALIASES.items():
        section = next((s[n] for s in scopes for n in names if isinstance(s.get(n), dict)), None)
        if section is None:
            raise DataError(f"panel_a.json: no encuentro la sección '{key}' (busqué las claves: {', '.join(names)}).")
        parsed[key] = _method_stats(section, key)
    meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
    parsed["n_keys"] = data.get("n_keys", meta.get("n_keys"))
    parsed["n_buckets"] = data.get("n_buckets", meta.get("n_buckets", parsed["single"]["n_buckets"]))
    return parsed


# --- Panel B ----------------------------------------------------------------
_B_COLUMNS = {"n", "input", "algorithm", "time_s_median", "comparisons_mean", "comparisons_theory"}


def parse_panel_b(data: dict) -> pd.DataFrame:
    if not isinstance(data, dict) or "results" not in data:
        raise DataError("panel_b.json: falta la lista 'results' (vuelve a correr src/task2_quicksort.py).")
    df = pd.DataFrame(data["results"])
    missing = _B_COLUMNS - set(df.columns)
    if missing:
        raise DataError(f"panel_b.json: a los resultados les faltan las columnas {sorted(missing)}.")
    return df.sort_values(["input", "algorithm", "n"]).reset_index(drop=True)


# --- Panel C ----------------------------------------------------------------
def parse_panel_c(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Devuelve (rutas ordenadas de mayor a menor, matriz ruta x hora con taps/día promedio, días)."""
    try:
        routes = pd.DataFrame(data["routes"]).sort_values("passengers", ascending=False).reset_index(drop=True)
        days = max(int(data["meta"]["days"]), 1)
        hourly = pd.DataFrame(data["hourly"]).T.reindex(columns=range(24), fill_value=0)
        hourly.columns = range(24)
        hourly = hourly.astype(float) / days
    except (KeyError, TypeError, ValueError) as e:
        raise DataError(f"panel_c.json no tiene el formato esperado ({e!r}); vuelve a correr src/panel_c_data.py.") from e
    return routes, hourly, days


def parse_capacity(task5, route_id: str):
    """Capacidad C (taps/hora) del Task 5 para dibujar la línea. None si no hay dato utilizable.

    Acepta  {"capacity": 120}  o  {"capacity": {"L-A": 120, ...}}  (y los alias 'capacidad', 'C').
    """
    if not isinstance(task5, dict):
        return None
    cap = _first(task5, ("capacity", "capacidad", "C", "capacities"))
    if isinstance(cap, dict):
        cap = cap.get(route_id)
    if isinstance(cap, (int, float)) and cap > 0:
        return float(cap)
    return None


# --------------------------------------------------------------------------- #
# Panel A - carga por bucket
# --------------------------------------------------------------------------- #
@themed
def fig_panel_a(pa: dict):
    single, p2c = pa["single"], pa["p2c"]
    top = max(single["max_load"], p2c["max_load"])
    loads = np.arange(0, top + 1)
    ys = np.array([single["counts"].get(int(k), 0) for k in loads])
    yp = np.array([p2c["counts"].get(int(k), 0) for k in loads])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.4
    ax.bar(loads - w / 2, ys, w, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, label="Una función hash")
    ax.bar(loads + w / 2, yp, w, color=BLUE, edgecolor=SURFACE, linewidth=1.2, label="Power of two choices")

    # Etiquetas directas selectivas: solo la carga máxima de cada método. Se colocan a una altura
    # fija (más alta para la que está junto a barras grandes) con una línea guía hasta su barra.
    ymax = float(max(ys.max(), yp.max()))
    for res, counts, dx, frac in ((single, ys, -w / 2, 0.24), (p2c, yp, w / 2, 0.40)):
        k = res["max_load"]
        ax.annotate(
            f"carga máx. = {k}",
            xy=(k + dx, counts[k]),
            xytext=(k + dx, ymax * frac),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1, "shrinkA": 1, "shrinkB": 1},
        )
    ax.set_ylim(0, ymax * 1.12)

    ax.set_xticks(loads)
    ax.set_xlim(-0.7, top + 0.9)
    ax.set_xlabel("Elementos (card_id) en un bucket")
    ax.set_ylabel("Número de buckets")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v)))
    ax.legend(loc="upper right")

    if p2c["max_load"] < single["max_load"]:
        title = (
            f"Power of two choices baja la carga máxima de {single['max_load']} a {p2c['max_load']} "
            f"elementos por bucket (desviación {es(single['std'], 2)} → {es(p2c['std'], 2)})"
        )
    else:
        title = "Carga por bucket: una función hash vs power of two choices"
    _title(fig, title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Panel B - aleatorizado vs determinista
# --------------------------------------------------------------------------- #
def _fmt_size(v, _=None):
    return f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}"


def _fmt_time(v, _=None):
    return f"{v * 1000:g} ms" if v < 1 else f"{v:g} s"


@themed
def fig_panel_b(df: pd.DataFrame, metric: str = "time"):
    """metric = 'time' (segundos, mediana) o 'comparisons' (media, con la teoría punteada)."""
    col = "time_s_median" if metric == "time" else "comparisons_mean"
    names = {"deterministic": "Determinista", "randomized": "Aleatorizado"}
    colors = {"deterministic": ORANGE, "randomized": BLUE}
    sizes = sorted(df["n"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    for ax, kind, label in zip(axes, ("random", "sorted"), ("Entrada aleatoria", "Entrada ya ordenada")):
        sub = df[df["input"] == kind]
        ends = []
        for alg in ("randomized", "deterministic"):
            s = sub[sub["algorithm"] == alg].sort_values("n")
            if s.empty:
                continue
            ax.plot(
                s["n"], s[col], color=colors[alg], linewidth=2, marker="o", markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=names[alg], zorder=3,
            )
            ends.append((float(s[col].iloc[-1]), s["n"].iloc[-1], alg))
            if metric == "comparisons":
                ax.plot(s["n"], s["comparisons_theory"], color=INK, linewidth=1.3, linestyle=(0, (1, 2)), zorder=4)

        # Etiqueta directa al final de cada línea; si los extremos casi coinciden, se separan.
        ends.sort(key=lambda e: e[0])
        close = len(ends) == 2 and ends[1][0] / max(ends[0][0], 1e-12) < 1.5
        for i, (y, n, alg) in enumerate(ends):
            dy = (-7 if i == 0 else 7) if close else 0
            ax.annotate(names[alg], (n, y), xytext=(8, dy), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2, annotation_clip=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_size))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 3.2)
        ax.set_xlabel("n (tamaño de la entrada)")
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if metric == "time":
        axes[0].yaxis.set_major_formatter(FuncFormatter(_fmt_time))
        axes[0].set_ylabel("Tiempo (mediana)")
    else:
        axes[0].set_ylabel("Comparaciones (media)")
        handles.append(plt.Line2D([], [], color=INK, linewidth=1.3, linestyle=(0, (1, 2))))  # la teoría va punteada
        labels.append("Teoría")
    axes[0].legend(handles, labels, loc="upper left")

    nmax = int(max(sizes))
    pick = lambda alg: df[(df["n"] == nmax) & (df["input"] == "sorted") & (df["algorithm"] == alg)]
    det, rnd = pick("deterministic"), pick("randomized")
    if metric == "time" and not det.empty and not rnd.empty:
        t_d, t_r = float(det["time_s_median"].iloc[0]), float(rnd["time_s_median"].iloc[0])
        title = (
            f"Con n = {es(nmax)} y entrada ordenada, el determinista tarda {es(t_d, 2)} s y el aleatorizado "
            f"{es(t_r, 2)} s ({es(t_d / t_r, 0)}× más rápido)"
        )
        rd = df[(df["n"] == nmax) & (df["input"] == "random")]
        if len(rd) == 2:  # con entrada aleatoria, ¿van parejos?
            tt = rd.set_index("algorithm")["time_s_median"]
            if 0.8 <= tt["deterministic"] / tt["randomized"] <= 1.25:
                title += "; con entrada aleatoria van parejos"
    elif metric == "comparisons":
        title = (
            "Las comparaciones medidas siguen la teoría: n(n−1)/2 del determinista con entrada ordenada "
            "y 2(n+1)Hₙ − 4n del aleatorizado"
        )
    else:
        title = "QuickSort aleatorizado vs determinista"
    _title(fig, title)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    return fig


def table_panel_b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["input"] = out["input"].map({"random": "aleatoria", "sorted": "ordenada"})
    out["algorithm"] = out["algorithm"].map({"deterministic": "determinista", "randomized": "aleatorizado"})
    cols = {
        "n": "n", "input": "entrada", "algorithm": "algoritmo", "time_s_median": "tiempo (s)",
        "comparisons_mean": "comparaciones", "comparisons_theory": "teoría", "comparisons_ratio": "medido/teoría",
        "correct": "correcto",
    }
    return out[[c for c in cols if c in out.columns]].rename(columns=cols)


# --------------------------------------------------------------------------- #
# Panel C - pasajeros por ruta
# --------------------------------------------------------------------------- #
@themed
def fig_panel_c_total(routes: pd.DataFrame, top_k: int = 5):
    k = min(top_k, len(routes))
    colors = [BLUE] * k + [NEUTRAL] * (len(routes) - k)
    fig, ax = plt.subplots(figsize=(6.6, 0.26 * len(routes) + 1.9))
    y = np.arange(len(routes))
    ax.barh(y, routes["passengers"], height=0.72, color=colors, edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(routes["route_id"], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    pad = routes["passengers"].max() * 0.015
    for i in range(k):  # número solo en las rutas resaltadas
        v = routes["passengers"].iloc[i]
        ax.text(v + pad, i, es(v), va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, routes["passengers"].max() * 1.14)
    ax.set_xlabel("Pasajeros (taps) en el periodo simulado")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v)))
    ax.legend(
        handles=[Patch(color=BLUE, label=f"{k} rutas más concurridas"), Patch(color=NEUTRAL, label="Demás rutas")],
        loc="lower right",
    )
    share = 100 * routes["passengers"].iloc[:k].sum() / routes["passengers"].sum()
    _title(fig, f"Las {k} rutas más concurridas concentran el {es(share, 1)}% de los pasajeros", width=58)
    fig.tight_layout()
    return fig


@themed
def fig_panel_c_hourly(routes: pd.DataFrame, hourly: pd.DataFrame, capacity=None, top_k: int = 5):
    top = list(routes["route_id"].iloc[: min(top_k, len(routes))])
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))

    for rid in hourly.index:  # contexto: las demás rutas, tenues y finas
        if rid not in top:
            ax.plot(hours, hourly.loc[rid], color=NEUTRAL, linewidth=0.9, alpha=0.8, zorder=1)
    for rid, color in zip(top, TOP5):
        ax.plot(hours, hourly.loc[rid], color=color, linewidth=2, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=rid, zorder=3)
    ax.plot([], [], color=NEUTRAL, linewidth=1.2, label="Demás rutas")

    ymax = float(hourly.loc[top].to_numpy().max())
    if capacity is not None:
        ax.axhline(capacity, color=INK, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
        ax.text(23.4, capacity, f"Capacidad C = {es(capacity, 0)} (Task 5)", ha="right", va="bottom",
                fontsize=9, color=INK)
        ymax = max(ymax, capacity)

    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)], fontsize=8.5)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Taps por hora (promedio diario)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)

    lead = top[0]
    peak_h = int(hourly.loc[lead].idxmax())
    _title(fig, f"{lead} llega a {es(hourly.loc[lead].max(), 1)} taps por hora a las {peak_h}:00; "
                f"las horas pico son 6–9 am y 5–7 pm", width=70)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Exportar capturas (PNG) sin necesidad de Streamlit
# --------------------------------------------------------------------------- #
def export_all(outdir: Path = CAPTURAS) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    def save(fig, name):
        path = outdir / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    data = load_json("panel_a.json")
    if data is None:
        skipped.append("panel_a.json (lo genera src/task3_hashing.py)")
    else:
        save(fig_panel_a(parse_panel_a(data)), "panel_a_carga_por_bucket.png")

    data = load_json("panel_b.json")
    if data is None:
        skipped.append("panel_b.json (lo genera src/task2_quicksort.py)")
    else:
        df = parse_panel_b(data)
        save(fig_panel_b(df, "time"), "panel_b_tiempo.png")
        save(fig_panel_b(df, "comparisons"), "panel_b_comparaciones.png")

    data = load_json("panel_c.json")
    if data is None:
        skipped.append("panel_c.json (lo genera src/panel_c_data.py)")
    else:
        routes, hourly, _ = parse_panel_c(data)
        save(fig_panel_c_total(routes), "panel_c_pasajeros_por_ruta.png")
        save(fig_panel_c_hourly(routes, hourly, parse_capacity(load_task5(), routes["route_id"].iloc[0])),
             "panel_c_taps_por_hora.png")

    for p in written:
        print(f"guardado: {p}")
    for s in skipped:
        print(f"omitido, falta {s}")
    return written


# --------------------------------------------------------------------------- #
# Streamlit (capa fina: solo muestra lo que construyen las funciones de arriba)
# --------------------------------------------------------------------------- #
PRODUCERS = {
    "panel_a.json": "python src/task3_hashing.py",
    "panel_b.json": "python src/task2_quicksort.py",
    "panel_c.json": "python src/panel_c_data.py",
}


def _load_or_explain(st, name: str):
    """Carga un JSON; si falta o está mal, lo explica en pantalla y devuelve None."""
    try:
        data = load_json(name)
    except DataError as e:
        st.error(str(e))
        return None
    if data is None:
        st.info(f"Falta `results/{name}`. Se genera con `{PRODUCERS[name]}`.")
    return data


def _show(st, fig):
    st.pyplot(fig)
    plt.close(fig)


def _panel_a(st):
    st.subheader("Panel A · Carga por bucket sobre card_id")
    st.caption("Cuántos buckets tienen k tarjetas: una sola función hash universal vs power of two choices "
               "(cada tarjeta va al bucket menos lleno entre dos candidatos).")
    data = _load_or_explain(st, "panel_a.json")
    if data is None:
        return
    try:
        pa = parse_panel_a(data)
    except DataError as e:
        st.error(str(e))
        return
    single, p2c = pa["single"], pa["p2c"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Carga máxima · una función", single["max_load"])
    c2.metric("Carga máxima · P2C", p2c["max_load"], delta=p2c["max_load"] - single["max_load"], delta_color="inverse")
    c3.metric("Desviación · una función", es(single["std"], 2))
    c4.metric("Desviación · P2C", es(p2c["std"], 2), delta=round(p2c["std"] - single["std"], 2), delta_color="inverse")
    if pa.get("n_keys"):
        st.caption(f"{es(pa['n_keys'])} card_id distintos en {es(pa['n_buckets'])} buckets.")
    _show(st, fig_panel_a(pa))
    with st.expander("Ver datos en tabla"):
        loads = sorted(set(single["counts"]) | set(p2c["counts"]))
        st.dataframe(
            pd.DataFrame({"elementos por bucket": loads,
                          "buckets · una función": [single["counts"].get(k, 0) for k in loads],
                          "buckets · P2C": [p2c["counts"].get(k, 0) for k in loads]}),
            hide_index=True,
        )


def _panel_b(st):
    st.subheader("Panel B · QuickSort aleatorizado vs determinista")
    st.caption("Ordenar rutas por pasajeros. Determinista = pivote fijo (primer elemento); aleatorizado = pivote al azar. "
               "Los dos son Las Vegas: siempre devuelven el orden correcto, solo cambia el tiempo.")
    data = _load_or_explain(st, "panel_b.json")
    if data is None:
        return
    try:
        df = parse_panel_b(data)
    except DataError as e:
        st.error(str(e))
        return
    choice = st.radio("Métrica", ["Tiempo", "Comparaciones"], horizontal=True)
    _show(st, fig_panel_b(df, "time" if choice == "Tiempo" else "comparisons"))
    with st.expander("Ver datos en tabla"):
        st.dataframe(table_panel_b(df), hide_index=True)
        rr = data.get("route_ranking")
        if rr:
            ok = "✅ coincide" if rr.get("verified_vs_exact") else "❌ NO coincide"
            st.markdown(f"**Ranking de rutas con QuickSort (Task 2b)** · {ok} con el orden exacto")
            st.dataframe(pd.DataFrame(rr["ranking"]), hide_index=True)


def _panel_c(st):
    st.subheader("Panel C · Pasajeros por ruta")
    st.caption("Pasajeros = taps de la ruta en el periodo simulado. Se resaltan las 5 rutas más concurridas.")
    data = _load_or_explain(st, "panel_c.json")
    if data is None:
        return
    try:
        routes, hourly, days = parse_panel_c(data)
        task5 = load_task5()
    except DataError as e:
        st.error(str(e))
        return
    meta = data.get("meta", {})
    st.caption(f"{es(meta.get('n_taps', routes['passengers'].sum()))} taps · {len(routes)} rutas · {days} días "
               f"({str(meta.get('period_start', ''))[:10]} a {str(meta.get('period_end', ''))[:10]})")

    capacity = parse_capacity(task5, routes["route_id"].iloc[0])
    if capacity is None:
        st.caption("Línea de capacidad no disponible: falta la capacidad C en results/task5_markov.json (Task 5).")

    rank = (load_json("panel_b.json") or {}).get("route_ranking", {}).get("ranking")
    if rank:
        same = {r["route_id"] for r in rank[:5]} == set(routes["route_id"].iloc[:5])
        st.caption("✅ Las 5 rutas resaltadas coinciden con el ranking del QuickSort aleatorizado (Task 2b)."
                   if same else "⚠️ Las 5 rutas resaltadas NO coinciden con el ranking de panel_b.json: "
                                "revisa que ambos JSON vengan del mismo CSV.")

    left, right = st.columns(2)
    with left:
        _show(st, fig_panel_c_total(routes))
    with right:
        _show(st, fig_panel_c_hourly(routes, hourly, capacity))
    with st.expander("Ver datos en tabla"):
        st.dataframe(routes.rename(columns={"route_id": "ruta", "passengers": "pasajeros",
                                            "unique_cards": "tarjetas distintas", "share_pct": "% del total"}),
                     hide_index=True)


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Smart Card Fare Analytics · Grupo 8", layout="wide")
    st.title("Smart Card Fare Analytics")
    st.caption("Grupo 8 · Public Transit Smart Card · el dashboard solo lee results/*.json")

    with st.sidebar:
        st.header("Resultados cargados")
        for name, cmd in PRODUCERS.items():
            ok = (RESULTS / name).exists()
            st.markdown(f"{'✅' if ok else '⚠️'} `{name}`" + ("" if ok else f" · falta: `{cmd}`"))
        st.caption(f"Carpeta: {RESULTS}")

    tab_a, tab_b, tab_c = st.tabs(["Panel A · Carga por bucket", "Panel B · Aleatorizado vs determinista",
                                   "Panel C · Pasajeros por ruta"])
    with tab_a:
        _panel_a(st)
    with tab_b:
        _panel_b(st)
    with tab_c:
        _panel_c(st)


if __name__ == "__main__":
    if "--export" in sys.argv:
        export_all()
    else:
        run_streamlit()
