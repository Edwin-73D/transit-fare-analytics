# Task 1 – Contexto Big Data

**Escenario:** Public Transit Smart Card Fare Analytics (Grupo 8)
**Objetivo:** estimar cuántos taps por segundo procesa el sistema en hora pico y cuánto almacenamiento necesita si los pasajeros crecen 10x.

## 1. Dato base

Tomamos como referencia el Metro de Medellín, que registró **310,2 millones de viajes en 2025** ([El Tiempo, 2025](https://www.eltiempo.com/amp/colombia/medellin/metro-de-medellin-2025-los-datos-que-demuestran-por-que-es-el-mejor-sistema-de-transporte-de-colombia-3555247)).

## 2. Supuestos

| # | Supuesto | Valor | Justificación |
|---|----------|-------|---------------|
| S1 | Factor día hábil vs promedio anual | 1,2 | Los fines de semana y festivos tienen menos viajes, por eso un día hábil está por encima del promedio |
| S2 | Taps por viaje | 1,3 | Algunos viajes incluyen transbordo a bus integrado o metrocable, lo que genera un tap extra |
| S3 | % de taps del día en la hora más cargada (7–8 am) | 12% | La demanda se concentra en las horas pico de la mañana y la tarde |
| S4 | Factor del minuto pico dentro de la hora pico | 2 | La llegada de pasajeros no es uniforme: hay ráfagas (llegada de trenes, cambio de turno) |
| S5 | Tamaño de un registro | 200 bytes | Un registro JSON de ejemplo pesa 172 bytes; redondeamos por nombres de estación más largos |
| S6 | Sobrecosto de índices y metadatos | +50% → 300 bytes/tap | Índice por card_id y route_id para las consultas rápidas |
| S7 | Réplicas | 3 | Tolerancia a fallos estándar en sistemas distribuidos |

Registro de ejemplo (172 bytes):

```json
{"tap_id": "T0000001", "timestamp": "2026-10-05T07:42:13", "card_id": "C004512", "route_id": "B-120", "station": "Universidad", "fare_amount": 2.75, "transfer_flag": false}
```

## 3. Cálculos

**Taps por día hábil**

```
Viajes/día promedio = 310.200.000 / 365        ≈   850.000
Viajes/día hábil    = 850.000 × 1,2 (S1)       ≈ 1.020.000
Taps/día hábil      = 1.020.000 × 1,3 (S2)     ≈ 1.330.000  (~1,3 M)
```

**Taps por segundo en hora pico**

```
Taps en hora pico   = 1.300.000 × 0,12 (S3)    = 156.000 taps/hora
Taps/seg promedio   = 156.000 / 3.600          ≈ 43 taps/seg
Taps/seg minuto pico= 43 × 2 (S4)              ≈ 90 taps/seg
```

**Almacenamiento anual**

```
Taps/año            = 310.200.000 × 1,3 (S2)   ≈ 403 M taps
Almacenamiento/año  = 403 M × 300 B (S6)       ≈ 121 GB
Con 3 réplicas (S7) = 121 GB × 3               ≈ 363 GB
```

**Crecimiento 10x:** todas las cantidades escalan linealmente con el número de pasajeros, así que se multiplican por 10.

## 4. Resultados

| Métrica | Hoy | 10x |
|---------|-----|-----|
| Taps/día hábil | 1,3 M | 13 M |
| Taps/seg (hora pico, promedio) | 43 | 430 |
| Taps/seg (minuto pico) | ~90 | ~870 |
| Almacenamiento/año (sin réplicas) | 121 GB | 1,2 TB |
| Almacenamiento/año (3 réplicas) | 363 GB | 3,6 TB |

## 5. Análisis de sensibilidad

- Si el registro pesa **400 B** en vez de 300 B, el almacenamiento a 10x sube a **≈ 1,6 TB/año** (sin réplicas). El tamaño del registro solo afecta el almacenamiento, no los taps/seg.
- Si la hora pico concentra el **15%** del día en vez del 12%, el minuto pico a 10x sube a **≈ 1.080 taps/seg**. El sistema debe dimensionarse para el peor minuto, no para el promedio.

## 6. Conclusión: por qué esto es un problema de Big Data

El sistema cumple las 3 V:

- **Volumen:** terabytes por año a 10x.
- **Velocidad:** cientos de eventos por segundo, con picos cercanos a 900 taps/seg.
- **Variedad:** taps de metro, bus integrado y metrocable.

Con este volumen no es viable recorrer todo el dataset para cada consulta. Por eso el resto del workshop usa:

- **Hashing universal** (Task 3) para buscar por card_id en tiempo esperado O(1).
- **Muestreo aleatorizado** (Task 2) para estimar el fare promedio por ruta sin leer todos los registros.
- **Cotas de probabilidad** (Task 5) para anticipar la saturación de una ruta en hora pico sin simular todos los escenarios.

## Fuentes

- [Metro de Medellín 2025: cifras récord, ahorros para usuarios y sostenibilidad – El Tiempo](https://www.eltiempo.com/amp/colombia/medellin/metro-de-medellin-2025-los-datos-que-demuestran-por-que-es-el-mejor-sistema-de-transporte-de-colombia-3555247)
