"""W3 Task 1 - Cadena de Markov (DTMC) de transbordos: matriz, clases comunicantes y periodo.

Task de Felipe (B). Escrito durante la integración (2026-10-23) sobre sus aristas
(w3/build_edges.py).

La cadena
---------
Estados = rutas. Un pasajero que está en la ruta i hace su próximo transbordo a la ruta j con
probabilidad  M[i][j] = transfer_count(i -> j) / (transbordos que salen de i).
Cada fila de M suma 1 (matriz estocástica), salvo las rutas sin salida (fila de ceros).

Teoría que se usa
-----------------
- i -> j ("j es accesible desde i") si hay un camino de transbordos de i a j.
  i <-> j ("se comunican") si i -> j y j -> i. Las CLASES COMUNICANTES son los grupos de rutas
  que se comunican entre sí = componentes fuertemente conexas (SCC) del grafo de aristas.
- Una clase es CERRADA si ninguna arista sale de ella. En una cadena finita:
    cerrada -> todos sus estados son RECURRENTES (se vuelve a ellos con prob. 1);
    no cerrada -> TRANSITORIOS (con prob. 1 se sale y no se vuelve nunca).
- PERIODO de una clase = mcd de las longitudes de todos los ciclos que vuelven a un estado.
  Periodo 1 = APERIÓDICA. Se calcula con un BFS por niveles desde una ruta r de la clase:
  para cada arista u -> v dentro de la clase, nivel(u) + 1 - nivel(v) es la diferencia entre
  dos caminos de r a v, y el mcd de todas esas diferencias es el periodo.
  (Basta un ciclo de largo 2, A -> B -> A, y uno de largo 3 para tener mcd = 1.)

Decisiones (y por qué)
---------------------
1. SCC con el algoritmo de Kosaraju, escrito aquí (dos DFS: una en el grafo y otra en el grafo
   con las aristas al revés), en vez de una librería, para poder explicarlo paso a paso.
2. Filas de ceros (rutas sin salida, "dangling"): M no es estocástica en esa fila. Para
   CLASIFICAR se tratan como estado ABSORBENTE (lazo a sí misma, M[i][i] = 1): el pasajero que
   llega ahí se queda, y queda como clase cerrada de un solo estado (recurrente). En el JSON la
   fila se deja en ceros y se marca en "dangling": cada task decide su tratamiento (PageRank,
   por ejemplo, reparte esa masa uniformemente entre todas las rutas). Con los datos reales no
   hay rutas sin salida (decisión de Edwin); el caso se muestra con la matriz de juguete (--toy).
3. Con --toy se clasifica la matriz de juguete y se compara con su respuesta conocida, como
   prueba del algoritmo.

Uso:  python w3/w3_task1_dtmc.py [--toy]        Salida: results/w3_matrix.json
"""
import argparse
import json
import math
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(RAIZ, "results")
EDGES = os.path.join(RESULTS, "w3_edges.json")


def armar_matriz(aristas):
    """M[i][j] = transfer_count(i -> j) / total que sale de i. Filas sin salida en ceros."""
    rutas = sorted({a["from_route"] for a in aristas} | {a["to_route"] for a in aristas})
    pos = {r: i for i, r in enumerate(rutas)}
    conteo = [[0] * len(rutas) for _ in rutas]
    for a in aristas:
        conteo[pos[a["from_route"]]][pos[a["to_route"]]] += a["transfer_count"]
    M = [[c / sum(fila) if sum(fila) else 0.0 for c in fila] for fila in conteo]
    return rutas, M, conteo


def vecinos(M):
    """Lista de adyacencia: i -> j si M[i][j] > 0. Decisión 2: fila de ceros -> lazo i -> i."""
    return [[j for j, p in enumerate(fila) if p > 0] or [i] for i, fila in enumerate(M)]


def kosaraju(ady):
    """Componentes fuertemente conexas (decisión 1). Devuelve una lista de clases (índices)."""
    n = len(ady)
    # 1) DFS en el grafo: orden en que cada estado TERMINA (todos sus descendientes visitados)
    visto, orden = [False] * n, []
    for s in range(n):
        if visto[s]:
            continue
        visto[s] = True
        pila = [(s, iter(ady[s]))]
        while pila:                                   # DFS iterativo (sin recursión)
            u, it = pila[-1]
            v = next((v for v in it if not visto[v]), None)
            if v is None:
                pila.pop()
                orden.append(u)
            else:
                visto[v] = True
                pila.append((v, iter(ady[v])))
    # 2) DFS en el grafo al revés, empezando por el que terminó de último: cada árbol es una SCC
    inv = [[] for _ in range(n)]
    for u in range(n):
        for v in ady[u]:
            inv[v].append(u)
    clase, clases = [None] * n, []
    for s in reversed(orden):
        if clase[s] is not None:
            continue
        clase[s], pila, grupo = len(clases), [s], []
        while pila:
            u = pila.pop()
            grupo.append(u)
            for v in inv[u]:
                if clase[v] is None:
                    clase[v] = len(clases)
                    pila.append(v)
        clases.append(sorted(grupo))
    return clases


def periodo(grupo, ady):
    """mcd de nivel(u) + 1 - nivel(v) sobre las aristas u -> v de la clase (BFS por niveles)."""
    dentro = set(grupo)
    nivel, cola, d = {grupo[0]: 0}, [grupo[0]], 0
    for u in cola:                                    # BFS: la cola crece mientras se recorre
        for v in ady[u]:
            if v in dentro and v not in nivel:
                nivel[v] = nivel[u] + 1
                cola.append(v)
    for u in grupo:
        for v in ady[u]:
            if v in dentro:
                d = math.gcd(d, nivel[u] + 1 - nivel[v])
    return d or None                                  # None: un solo estado sin lazo (no hay ciclos)


def clasificar(rutas, M):
    """Clases comunicantes con su tipo (cerrada -> recurrente) y periodo."""
    ady = vecinos(M)
    resultado = []
    for grupo in kosaraju(ady):
        dentro = set(grupo)
        cerrada = all(v in dentro for u in grupo for v in ady[u])
        resultado.append({
            "routes": [rutas[i] for i in grupo],
            "closed": cerrada,
            "type": "recurrent" if cerrada else "transient",
            "period": periodo(grupo, ady) if cerrada else None,
            "absorbing": len(grupo) == 1 and sum(M[grupo[0]]) == 0,   # ruta sin salida (decisión 2)
        })
    # primero las recurrentes y las más grandes
    return sorted(resultado, key=lambda c: (c["type"] != "recurrent", -len(c["routes"]), c["routes"]))


def imprimir(rutas, M, clases, dangling):
    print(f"Estados (rutas): {len(rutas)} | filas de M que suman 1: "
          f"{sum(abs(sum(f) - 1) < 1e-9 for f in M)} | sin salida: {dangling or 'ninguna'}")
    print(f"Clases comunicantes: {len(clases)}")
    for c in clases:
        extra = " (absorbente: ruta sin salida)" if c["absorbing"] else ""
        per = f", periodo {c['period']}" + (" -> aperiódica" if c["period"] == 1 else "") if c["period"] else ""
        print(f"  {c['type']:>10}{per}{extra}: {', '.join(c['routes'])}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="clasificar la matriz de juguete (prueba)")
    args = ap.parse_args()

    if args.toy:                                      # decisión 3
        from toy_matrix import EXPECTED_CLASSES, toy_matrix
        t = toy_matrix()
        clases = clasificar(t["routes"], t["M"])
        imprimir(t["routes"], t["M"], clases, t["dangling"])
        print("Coincide con la respuesta conocida de toy_matrix.py:", clases == EXPECTED_CLASSES)
        return

    if not os.path.exists(EDGES):
        sys.exit("Falta results/w3_edges.json: corre primero python w3/build_edges.py")
    with open(EDGES, encoding="utf-8") as f:
        aristas = json.load(f)
    rutas, M, conteo = armar_matriz(aristas)
    dangling = [r for r, fila in zip(rutas, M) if sum(fila) == 0]
    clases = clasificar(rutas, M)
    imprimir(rutas, M, clases, dangling)

    salida = {
        "routes": rutas,
        "M": M,
        "counts": conteo,
        "dangling": dangling,
        "classes": clases,
        "meta": {
            "task": "W3 Task 1 - DTMC de transbordos: clases comunicantes, recurrencia y periodo",
            "source": "results/w3_edges.json",
            "M": "M[i][j] = transfer_count(i -> j) / transbordos que salen de i (fila = desde)",
            "scc": "Kosaraju (propio); cerrada -> recurrente, no cerrada -> transitoria",
            "period": "mcd de nivel(u) + 1 - nivel(v) en las aristas de la clase (BFS)",
            "dangling_treatment": "absorbente (lazo) para clasificar; la fila queda en ceros en M",
        },
    }
    out = os.path.join(RESULTS, "w3_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print("Escrito", out)


if __name__ == "__main__":
    main()
