"""Generador de taps del Grupo 8 (Smart Card Fare Analytics).

Lo usan todos los scripts de las tasks. Si lo corres solo (python generador.py)
crea data/taps_10k.csv y data/taps_100k.csv.
"""
import os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "data")
RESULTS = os.path.join(AQUI, "results")

SEED = 42
P = 2**61 - 1                      # primo de Mersenne, mayor que cualquier llave
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

# generador global para los hash aleatorios
rng = np.random.default_rng(SEED)


def generar(n, seed=SEED):
    g = np.random.default_rng(seed)

    # las 3000 tarjetas salen al menos una vez, el resto al azar
    tarjetas = np.concatenate([np.arange(1, N_TARJETAS + 1), g.integers(1, N_TARJETAS + 1, n - N_TARJETAS)])
    g.shuffle(tarjetas)

    segundos = np.sort(g.integers(0, DIAS * 86400, n))
    ruta = g.choice(len(RUTAS), n, p=PESOS)
    estacion = g.choice(ESTACIONES, n)
    transfer = g.random(n) < g.uniform(0.05, 0.40)      # % de transferencias entre 5% y 40%

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


def llave(card_id):
    # "C000123" -> entero, con los bytes del texto
    return int.from_bytes(card_id.encode(), "big")


def hash_universal(m, gen=None):
    # h(k) = ((a*k + b) mod p) mod m con a y b al azar
    gen = gen if gen is not None else rng
    a = int(gen.integers(1, P))
    b = int(gen.integers(0, P))
    return lambda k: ((a * k + b) % P) % m


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
    for nombre in ("10k", "100k"):
        cargar(nombre)
        print("listo data/taps_%s.csv" % nombre)
