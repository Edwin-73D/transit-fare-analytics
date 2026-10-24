"""W3 Task 3 - PageRank sobre la red de transferencias entre rutas.

Cada arista es el paso de una tarjeta de una ruta a otra (taps consecutivos de la misma
tarjeta en rutas distintas, ordenados por stream_seq). Con eso se arma la matriz de
transición y se calcula PageRank con d = 0.85, que se compara con el ridership por ruta.
Guarda results/edges.json, results/panel_d.json y la gráfica del Panel D.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generador import cargar, armar_stream, RESULTS

D = 0.85


def armar_aristas(stream):
    # para cada tap, el siguiente tap de la misma tarjeta
    s = stream.sort_values(["card_id", "stream_seq"])
    sig = s.groupby("card_id")[["route_id", "fare_amount", "transfer_flag"]].shift(-1)
    pares = pd.DataFrame({
        "from_route": s["route_id"],
        "to_route": sig["route_id"],
        "fare": sig["fare_amount"],
        "transfer": sig["transfer_flag"],
    }).dropna(subset=["to_route"])
    pares["transfer"] = pares["transfer"].astype(float)
    pares = pares[pares["from_route"] != pares["to_route"]]      # solo cambios de ruta

    aristas = pares.groupby(["from_route", "to_route"]).agg(
        transfer_count=("fare", "size"),
        avg_fare_amount=("fare", "mean"),
        transfer_frac=("transfer", "mean"),    # fracción con transfer_flag = true en el segundo tap
    ).reset_index()
    aristas["avg_fare_amount"] = aristas["avg_fare_amount"].round(2)
    aristas["transfer_frac"] = aristas["transfer_frac"].round(3)
    aristas.insert(0, "edge_id", ["e%03d" % i for i in range(1, len(aristas) + 1)])
    return aristas


def pagerank(M, d=D):
    n = len(M)
    sin_salida = M.sum(axis=1) == 0
    r = np.full(n, 1 / n)
    for it in range(1, 1001):
        # los nodos sin salida reparten parejo a todos
        nuevo = d * (r @ M + r[sin_salida].sum() / n) + (1 - d) / n
        if np.abs(nuevo - r).sum() < 1e-12:
            return nuevo, it
        r = nuevo
    return r, 1000


def main():
    os.makedirs(RESULTS, exist_ok=True)
    stream = armar_stream(cargar("100k"))

    aristas = armar_aristas(stream)
    aristas.to_json(os.path.join(RESULTS, "edges.json"), orient="records", indent=2)
    print("aristas:", len(aristas))
    print(aristas.iloc[0].to_dict())

    # matriz de transición: cada fila suma 1
    rutas = sorted(set(aristas["from_route"]) | set(aristas["to_route"]))
    pos = {r: i for i, r in enumerate(rutas)}
    n = len(rutas)
    W = np.zeros((n, n))
    for e in aristas.itertuples():
        W[pos[e.from_route], pos[e.to_route]] = e.transfer_count
    salida = W.sum(axis=1)
    M = np.divide(W, salida[:, None], out=np.zeros_like(W), where=salida[:, None] > 0)

    pr, iteraciones = pagerank(M)
    print("rutas:", n, "| iteraciones:", iteraciones, "| suma:", round(pr.sum(), 6))

    # comprobar contra la solución directa del sistema lineal
    Mf = M.copy()
    Mf[M.sum(axis=1) == 0] = 1 / n
    directa = np.linalg.solve(np.eye(n) - D * Mf.T, np.full(n, (1 - D) / n))
    print("coincide con la solución directa:", bool(np.allclose(pr, directa, atol=1e-9)))

    # comparar con el ridership
    ridership = stream["route_id"].value_counts()
    comp = pd.DataFrame({"route_id": rutas, "pagerank": pr, "ridership": [int(ridership[r]) for r in rutas]})
    comp["ridership_share"] = comp["ridership"] / comp["ridership"].sum()
    comp["rank_pagerank"] = comp["pagerank"].rank(ascending=False).astype(int)
    comp["rank_ridership"] = comp["ridership"].rank(ascending=False).astype(int)
    comp = comp.sort_values("rank_pagerank").reset_index(drop=True)

    rho = comp["pagerank"].corr(comp["ridership"], method="spearman")
    print("Spearman PageRank vs ridership:", round(float(rho), 4))
    print(comp.round(4).to_string(index=False))

    # Panel D
    orden = comp.sort_values("ridership_share", ascending=False)
    x = np.arange(len(orden))
    plt.figure(figsize=(12, 4))
    plt.bar(x - 0.2, orden["ridership_share"], width=0.4, label="ridership (fracción de taps)")
    plt.bar(x + 0.2, orden["pagerank"], width=0.4, label="PageRank (d = %s)" % D)
    plt.xticks(x, orden["route_id"], rotation=60)
    plt.title("Panel D: PageRank vs ridership por ruta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "panel_d_pagerank_vs_ridership.png"), dpi=150)
    plt.close()

    resultado = {"w3_task3_pagerank": {
        "damping": D,
        "n_rutas": n,
        "n_aristas": len(aristas),
        "iteraciones": iteraciones,
        "spearman_vs_ridership": round(float(rho), 4),
        "rutas": comp.to_dict("records"),
    }}
    with open(os.path.join(RESULTS, "panel_d.json"), "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
