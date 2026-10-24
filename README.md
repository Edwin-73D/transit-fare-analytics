# transit-fare-analytics

**Public Transit Smart Card Fare Analytics — Grupo 8**
Lógica y Representación III, Universidad de Antioquia, 2026-2.

El sistema de transporte de una ciudad procesa los taps de tarjetas inteligentes en estaciones
y buses, y necesita consultas rápidas por tarjeta y por ruta para analizar las tarifas. Usamos
como referencia el Metro de Medellín. Los datos son simulados con un generador propio.

## Equipo

| Persona | Rol en el Workshop 1 |
|---|---|
| Edwin Martinez (A) | Task 1, Task 2a, Task 5, integración |
| Felipe Corrales (B) | Generador de datos, Task 3 (hashing universal), README |
| Pedro Duque (C) | Task 2b (QuickSort), dashboard (Paneles A, B y C) |

## Instalación

Requiere Python 3.11 o superior.

```bash
git clone git@github.com:Edwin-73D/transit-fare-analytics.git
cd transit-fare-analytics
pip install -r requirements.txt
```

## Cómo correr el Workshop 1

Correr en este orden desde la raíz del repo:

```bash
# 1. Generar los datos (data/taps_10k.csv, data/taps_100k.csv y data/taps.csv = 100k, el oficial)
python src/generator.py

# 2. Verificar que el CSV cumple el esquema (detecta además la hora pico y su fuerza)
python src/validate_taps.py

# 3. Tasks (cada uno escribe su resultado en results/)
python src/task1_bytes_per_tap.py    # Task 1: bytes por tap medidos (complementa task1_estimates.md)
python src/task2_sampling.py         # Task 2a: reservoir sampling, ~15 s
python src/task2_quicksort.py        # Task 2b: QuickSort aleatorizado vs determinista, ~20 s
python src/task3_hashing.py          # Task 3: hashing universal + power of two choices
python src/task5_markov.py           # Task 5: cotas de Markov y Chebyshev
python src/panel_c_data.py           # datos del Panel C (pasajeros por ruta y por hora)

# 4. Dashboard
streamlit run dashboard.py           # en el navegador
python dashboard.py --export         # o guardar las gráficas como PNG en results/capturas/
```

El Task 1 es una estimación teórica con cifras reales del Metro: está en
[`src/task1_estimates.md`](src/task1_estimates.md).

## Qué produce cada script

| Script | Task | Salida | Panel |
|---|---|---|---|
| `src/generator.py` | Datos | `data/taps*.csv` | — |
| `src/validate_taps.py` | Control de calidad | consola | — |
| `src/task1_bytes_per_tap.py` | 1 | `results/task1_bytes.json` | — |
| `src/task2_sampling.py` | 2a | `results/task2_sampling.json` | — |
| `src/task2_quicksort.py` | 2b | `results/panel_b.json` | B |
| `src/task3_hashing.py` | 3 | `results/panel_a.json` | A |
| `src/task5_markov.py` | 5 | `results/task5_markov.json` | C (línea de capacidad) |
| `src/panel_c_data.py` | 6 | `results/panel_c.json` | C |
| `dashboard.py` | 6 | `results/capturas/*.png` | A, B, C |

**Regla del equipo:** cada task escribe su propio JSON en `results/` y el dashboard solo lee
esos archivos. Nadie depende del código de otro task.

## Datos

`src/generator.py` genera 3.000 tarjetas, 25 rutas (5 de metro `L-A`…`L-E` y 20 de bus
`B-101`…`B-120`) con popularidad tipo Zipf, 7 días a partir del 2025-10-05 y semilla fija (42).

| Campo | Formato | Ejemplo |
|---|---|---|
| tap_id | "T" + 8 dígitos | T00000001 |
| timestamp | AAAA-MM-DD HH:MM:SS | 2025-10-05 07:42:13 |
| card_id | "C" + 6 dígitos | C004512 |
| route_id | código de ruta | L-A, B-120 |
| station | estación o parada | Universidad |
| fare_amount | pesos (COP); 0 si es transferencia | 3300.0 |
| transfer_flag | booleano | False |

- La hora pico no es fija: los scripts la detectan desde los datos y reportan qué tan marcada es.
- El % de transferencias lo sortea el generador por dataset (entre 5% y 40%); el primer tap de
  una tarjeta nunca es transferencia.
- `data/` y `results/*.json` no se suben al repo: se regeneran con los comandos de arriba.

## Estructura

```
transit-fare-analytics/
├── data/                 ← CSV generados (no se suben)
├── src/
│   ├── generator.py      ← generador de taps
│   ├── hashing.py        ← familia de hash universal compartida
│   ├── validate_taps.py  ← validador del esquema
│   ├── task1_estimates.md, task1_bytes_per_tap.py
│   ├── task2_sampling.py, task2_quicksort.py
│   ├── task3_hashing.py
│   ├── task5_markov.py
│   └── panel_c_data.py
├── results/              ← JSON de cada task y capturas del dashboard
├── w2/                   ← Workshop 2 (data streams), en desarrollo
├── w3/                   ← Workshop 3 (cadenas de Markov), en desarrollo
├── dashboard.py
├── requirements.txt
└── README.md
```
