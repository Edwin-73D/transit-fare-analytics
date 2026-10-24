"""Generador de taps del Grupo 8 (Smart Card Fare Analytics).

Autor original: Felipe (generador.py en su repo, Juan319-u/logica-3). Integrado al repo
principal el 2026-10-23 con los cambios marcados abajo como "Cambio (integración)".

Lo usan todos los scripts de las tasks. Si lo corres solo (python src/generator.py)
crea data/taps_10k.csv y data/taps_100k.csv.

Cambios (integración, 2026-10-23)
---------------------------------
1. Una tarjeta no puede hacer transferencia en su PRIMER tap: una transferencia es cambiar
   de ruta dentro de un viaje, así que necesita un tap anterior. Antes salían cientos de
   transferencias en el primer tap de una tarjeta (731 en 10k). Ahora el primer tap de cada
   tarjeta paga tarifa completa. Efecto: el % de transferencias baja un poco, sobre todo en
   10k, donde los primeros taps son el 30% del total.
2. La familia de hash (llave, hash_universal, P) se movió a src/hashing.py; se re-exporta
   aquí para que los scripts que hacían "from generador import llave" sigan funcionando.
3. data/ y results/ quedan en la raíz del repo (antes, junto al archivo).
4. Corrido como script, regenera los CSV aunque ya existan (antes solo si faltaban), para
   que nunca se use un CSV viejo de otra versión del generador. cargar() sigue igual:
   genera el CSV solo si no existe.
5. Corrido como script escribe también data/taps.csv = la versión de 100k, que es el dataset
   OFICIAL del proyecto (decisión del equipo, 2026-10-23). Todos los scripts buscan primero
   data/taps.csv, así que todos usan los mismos datos.
Lo demás (semilla, tarjetas, rutas Zipf, tarifas en COP, timestamps, % de transferencias
sorteado entre 5% y 40%) no cambió.
"""
import os
import shutil
import numpy as np
import pandas as pd

from hashing import P, llave, hash_universal  # noqa: F401  (Cambio 2: re-exportados)

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)                  # Cambio 3: raíz del repo
DATA = os.path.join(RAIZ, "data")
RESULTS = os.path.join(RAIZ, "results")

SEED = 42
N_TARJETAS = 3000
DIAS = 7
INICIO = pd.Timestamp("2025-10-05")
RUTAS = ["L-" + c for c in "ABCDE"] + ["B-%d" % i for i in range(101, 121)]   # 5 de metro y 20 de bus
ESTACIONES = ["Niquía", "Bello", "Madera", "Acevedo", "Tricentenario", "Caribe", "Universidad",
              "Hospital", "Prado", "Parque Berrío", "San Antonio", "Alpujarra", "Exposiciones",
              "Industriales", "Poblado", "Aguacatala", "Ayurá", "Envigado", "Itagüí", "Sabaneta"]
TARIFAS = np.array([3300.0] * 5 + [3000.0] * 20)    # COP, el metro cuesta más

# las rutas tienen popularidad Zipf (s = 1.1)
PESOS = 1 / np.arange(1, len(RUTAS) + 1) ** 1.1
PESOS = PESOS / PESOS.sum()


def generar(n, seed=SEED):
    g = np.random.default_rng(seed)

    # las 3000 tarjetas salen al menos una vez, el resto al azar
    tarjetas = np.concatenate([np.arange(1, N_TARJETAS + 1), g.integers(1, N_TARJETAS + 1, n - N_TARJETAS)])
    g.shuffle(tarjetas)

    segundos = np.sort(g.integers(0, DIAS * 86400, n))
    ruta = g.choice(len(RUTAS), n, p=PESOS)
    estacion = g.choice(ESTACIONES, n)
    transfer = g.random(n) < g.uniform(0.05, 0.40)      # % de transferencias entre 5% y 40%

    # Cambio 1: el primer tap de cada tarjeta no puede ser transferencia. Los taps ya están
    # en orden de tiempo (segundos ordenados), así que el primero de cada tarjeta es la
    # primera vez que aparece en el arreglo.
    primer_tap = ~pd.Series(tarjetas).duplicated().to_numpy()
    transfer &= ~primer_tap

    return pd.DataFrame({
        "tap_id": ["T%08d" % i for i in range(1, n + 1)],
        "timestamp": INICIO + pd.to_timedelta(segundos, unit="s"),
        "card_id": ["C%06d" % t for t in tarjetas],
        "route_id": [RUTAS[r] for r in ruta],
        "station": estacion,
        "fare_amount": np.where(transfer, 0.0, TARIFAS[ruta]),    # la transferencia no cobra
        "transfer_flag": transfer,
    })


def cargar(nombre):
    # nombre: "10k" o "100k". Si el csv no existe lo crea
    tamanos = {"10k": 10_000, "100k": 100_000}
    os.makedirs(DATA, exist_ok=True)
    ruta = os.path.join(DATA, "taps_%s.csv" % nombre)
    if not os.path.exists(ruta):
        generar(tamanos[nombre]).to_csv(ruta, index=False)
    return pd.read_csv(ruta, parse_dates=["timestamp"])


def armar_stream(df):
    # ordena por tiempo y agrega los campos nuevos de W2: stream_seq, vehicle_id
    # y sec (segundos desde el inicio, para las ventanas)
    g = np.random.default_rng(SEED + 1)
    s = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    s["stream_seq"] = np.arange(1, len(s) + 1)
    unidades = np.where(s["route_id"].str.startswith("L-"), 6, 10)
    s["vehicle_id"] = s["route_id"] + "-V" + pd.Series(g.integers(1, unidades + 1)).astype(str).str.zfill(2)
    s["sec"] = (s["timestamp"] - INICIO).dt.total_seconds().astype(int)
    return s


if __name__ == "__main__":
    # Cambio 4: siempre regenera, para no quedarse con un CSV de otra versión.
    os.makedirs(DATA, exist_ok=True)
    for nombre, n in (("10k", 10_000), ("100k", 100_000)):
        generar(n).to_csv(os.path.join(DATA, "taps_%s.csv" % nombre), index=False)
        print("listo data/taps_%s.csv" % nombre)
    # Cambio 5: el dataset oficial (100k) también como data/taps.csv.
    shutil.copyfile(os.path.join(DATA, "taps_100k.csv"), os.path.join(DATA, "taps.csv"))
    print("listo data/taps.csv (= taps_100k.csv, dataset oficial)")
