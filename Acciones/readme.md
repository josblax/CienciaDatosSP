# Autopráctica: Construcción de un portafolio de inversión con datos reales

**Curso:** Ciencia de Datos en el Sector Privado
**Modalidad:** Individual o en parejas
**Duración estimada:** 3 a 4 horas
**Herramientas:** Python, `yfinance`, `pandas`, `numpy`, `scipy`, `matplotlib`

---

## 1. Objetivo

Que el alumno investigue empresas reales, construya un portafolio diversificado y use optimización de media-varianza (Markowitz) para decidir:

1. **Qué empresas** incluir y **cuántas**.
2. **Qué porcentaje** del capital va a cada una.
3. **Cuántos títulos (acciones enteras)** comprar con el capital disponible.

Después valida si su decisión hubiera funcionado con datos que **no** usó para optimizar.

---

## 2. Contexto del caso

Una empresa familiar tiene **$28,000 USD** de excedente de tesorería y te contrata como analista de datos. El director financiero pide:

> "Quiero la mejor relación rendimiento-riesgo posible, pero no quiero depender de una sola empresa ni de un solo sector. Justifica cada decisión con datos."

---

## 3. Reglas del portafolio

| Restricción | Rango permitido |
|---|---|
| Número de empresas | **Entre 6 y 10** |
| Número de industrias distintas | **Entre 3 y 5** |
| Peso mínimo por empresa | **5 %** |
| Peso máximo por empresa | **30 %** |
| Peso máximo por industria | **40 %** |
| Ventas en corto | **No permitidas** (pesos ≥ 0) |
| Capital sin invertir (efectivo sobrante) | **Máximo 3 %** ($840) |
| Periodo de entrenamiento | 2025-01-01 a 2025-12-31 |
| Periodo de prueba | 2026-01-01 a la fecha de entrega |

> **¿Por qué estos rangos?** Con menos de 6 empresas el riesgo específico de cada una todavía pesa mucho. Arriba de 10, el beneficio de diversificar crece poco y el portafolio se vuelve difícil de explicar y de vigilar. El tope por industria evita que "diversificar" signifique tener 6 empresas de tecnología.

---

## 4. Industrias sugeridas

Elige **entre 3 y 5** industrias. Se recomienda combinar al menos una **defensiva** (baja volatilidad) con una de **crecimiento** (alta volatilidad). Los tickers son solo ejemplos: puedes y debes buscar otros.

| Industria | Perfil | Ejemplos (EE. UU.) | Ejemplos (BMV, sufijo `.MX`) |
|---|---|---|---|
| Tecnología / semiconductores | Crecimiento, alta volatilidad | AAPL, MSFT, NVDA, AMD, GOOGL | — |
| Consumo discrecional | Crecimiento, cíclico | AMZN, TSLA, NKE, MCD | — |
| Consumo básico | Defensivo | KO, PG, WMT, PEP | WALMEX.MX, FEMSAUBD.MX, BIMBOA.MX |
| Servicios financieros | Cíclico, sensible a tasas | JPM, V, MA, GS | GFNORTEO.MX |
| Salud / farmacéutica | Defensivo | JNJ, UNH, PFE, LLY | — |
| Energía | Cíclico, ligado a materias primas | XOM, CVX | — |
| Servicios públicos (utilities) | Muy defensivo | NEE, DUK | — |
| Telecomunicaciones / medios | Mixto | NFLX, T, VZ | AMXB.MX |
| Industrial / materiales | Cíclico | CAT, GE, HON | CEMEXCPO.MX, GMEXICOB.MX |

**Mezclas de referencia** (solo como punto de partida, no como respuesta):

| Perfil | Crecimiento | Cíclico | Defensivo |
|---|---|---|---|
| Conservador | 15–25 % | 25–35 % | 45–60 % |
| Balanceado | 30–40 % | 30–40 % | 25–35 % |
| Agresivo | 45–60 % | 25–35 % | 10–20 % |

> Si combinas acciones de EE. UU. con acciones de la BMV, recuerda que las `.MX` cotizan en **pesos mexicanos**. Antes de mezclarlas debes convertirlas a USD con el tipo de cambio (`MXN=X` en yfinance) o justificar por qué no lo haces.

---

## 5. Actividades

### Parte A. Investigación (sin código)

Para **cada empresa** que consideres, investiga y anota en una tabla:

- Industria y qué vende.
- Capitalización de mercado aproximada.
- Un riesgo concreto de la empresa (regulatorio, competencia, deuda, etc.).
- Por qué crees que aporta algo al portafolio.

Empieza con una **lista larga de 12 a 15 candidatas** y descarta hasta llegar a tu selección final de 6 a 10. Documenta por qué descartaste cada una.

### Parte B. Datos y estadísticos

1. Descarga los precios de cierre ajustados del periodo de **entrenamiento**.
2. **Reordena las columnas** para que coincidan con tu lista de tickers (yfinance las devuelve en orden alfabético).
3. Calcula los rendimientos diarios, el rendimiento anual esperado (μ) y la matriz de covarianza anualizada (Σ).
4. Genera un **mapa de calor de la matriz de correlación**. Identifica el par de empresas más correlacionado y el menos correlacionado.

### Parte C. Optimización

Usa `scipy.optimize.minimize` para encontrar:

1. El portafolio de **mínima volatilidad**.
2. El portafolio de **máxima razón de Sharpe** (usa como tasa libre de riesgo la tasa de los T-Bills a 3 meses vigente; investígala y cita la fuente).

Ambos deben respetar **todas** las restricciones de la sección 3, incluido el límite por industria.

Grafica la nube de portafolios (simulación Monte Carlo de al menos 10,000 portafolios aleatorios) en un plano **riesgo (X) vs. rendimiento (Y)** y marca los dos portafolios óptimos.

### Parte D. De porcentajes a acciones enteras

No se pueden comprar 3.7 acciones de una empresa (salvo con fracciones, que aquí **no** se permiten). Con el portafolio de máximo Sharpe:

1. Toma el **último precio** del periodo de entrenamiento.
2. Calcula cuántos títulos enteros comprar de cada empresa con $28,000.
3. Reporta el peso **real** resultante (que será distinto del óptimo) y el efectivo sobrante.
4. Si el sobrante supera 3 %, propón un ajuste: por ejemplo, redondear hacia arriba la empresa que más se alejó de su peso óptimo.

### Parte E. Validación fuera de muestra

Este es el paso que distingue a un analista de datos de alguien que solo ajusta un modelo.

1. Con los títulos enteros de la parte D, calcula cuánto valdría tu portafolio con los precios del **periodo de prueba**.
2. Compáralo contra tres alternativas con el mismo capital:
   - Pesos iguales (1/N) entre tus mismas empresas.
   - Tu portafolio de mínima volatilidad.
   - Un índice de referencia: `SPY` (S&P 500) o `^MXX` (IPC) según tu mezcla.
3. Grafica las cuatro curvas de valor en el tiempo.
4. Reporta para cada una: rendimiento total, volatilidad anualizada, Sharpe y **máxima caída (max drawdown)**.

---

## 6. Plantilla de código base

Completa las secciones marcadas con `TODO`.

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------- Configuración ----------
capital = 28_000.0
rf = 0.0          # TODO: tasa libre de riesgo anual investigada
w_min, w_max = 0.05, 0.30
tope_industria = 0.40

# TODO: tu selección final (6 a 10 empresas, 3 a 5 industrias)
industrias = {
    "AAPL": "Tecnología",
    "KO":   "Consumo básico",
    # ...
}
tickers = list(industrias)

# ---------- Datos ----------
precios = yf.download(tickers, start="2025-01-01", end="2026-01-01",
                      auto_adjust=True, progress=False)["Close"]
precios = precios[tickers].dropna()          # ¡reordenar!
rend = precios.pct_change().dropna()
mu = rend.mean().values * 252
cov = rend.cov().values * 252

# ---------- Funciones del portafolio ----------
def riesgo(w):
    return np.sqrt(w @ cov @ w)

def rendimiento(w):
    return w @ mu

def sharpe_negativo(w):
    return -(rendimiento(w) - rf) / riesgo(w)

# ---------- Restricciones ----------
n = len(tickers)
limites = [(w_min, w_max)] * n
restricciones = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

# TODO: agregar una restricción de desigualdad por industria
# Pista: para cada industria, construye una máscara booleana con los
# tickers que le pertenecen y exige  tope_industria - w[mascara].sum() >= 0

w0 = np.full(n, 1 / n)

# TODO: optimizar mínima volatilidad y máximo Sharpe
# res = minimize(sharpe_negativo, w0, method="SLSQP",
#                bounds=limites, constraints=restricciones)

# ---------- Monte Carlo ----------
# TODO: generar >= 10,000 vectores de pesos aleatorios (np.random.dirichlet)
# y graficar riesgo vs rendimiento coloreado por Sharpe

# ---------- Acciones enteras ----------
ultimo_precio = precios.iloc[-1].values
# TODO: titulos = np.floor(w_opt * capital / ultimo_precio)
#       calcular peso real y efectivo sobrante

# ---------- Validación fuera de muestra ----------
# TODO: descargar 2026-01-01 a hoy, valuar los títulos y comparar
#       contra 1/N, mínima volatilidad y SPY o ^MXX
```

---

## 7. Entregables

1. **Notebook** (`.ipynb`) ejecutado de principio a fin sin errores.
2. **Reporte ejecutivo** de máximo 2 páginas dirigido al director financiero. Debe incluir:
   - Tabla final: empresa, industria, títulos, precio de compra, monto invertido y peso real.
   - Gráfica de riesgo vs. rendimiento con los portafolios óptimos.
   - Gráfica de validación fuera de muestra.
   - Recomendación clara en lenguaje no técnico.
3. **Tabla de investigación** de la parte A, con las fuentes consultadas.

---

## 8. Preguntas de reflexión

Respóndelas en el reporte o al final del notebook.

1. ¿Tu portafolio de máximo Sharpe le ganó al de pesos iguales (1/N) en el periodo de prueba? ¿Por qué crees que pasó eso?
2. ¿Qué empresa recibió el peso máximo (30 %)? Si quitas ese tope, ¿cuánto peso le asignaría el optimizador? ¿Es razonable?
3. Usaste un año de historia para estimar μ. ¿Qué tan confiable es esa estimación? ¿Qué pasaría si usaras 5 años?
4. ¿Cuánto cambió el Sharpe al pasar de pesos óptimos a títulos enteros? ¿Cuándo importa más este problema: con capital chico o grande?
5. Menciona un riesgo que **no** aparece en la matriz de covarianza histórica y que podría afectar tu portafolio.

---

## 9. Rúbrica

| Criterio | Puntos |
|---|---|
| Investigación: lista larga, justificación de altas y descartes, fuentes | 15 |
| Datos y estadísticos correctos (orden de columnas, anualización, correlaciones) | 15 |
| Optimización con **todas** las restricciones, incluida la de industria | 20 |
| Conversión a títulos enteros y control del efectivo sobrante | 10 |
| Validación fuera de muestra contra las tres referencias | 20 |
| Reporte ejecutivo claro y orientado a la decisión | 10 |
| Preguntas de reflexión | 10 |
| **Total** | **100** |

**Penalizaciones:**
- Portafolio que viola alguna restricción de la sección 3: **−10**.
- Etiquetas de pesos que no corresponden al activo correcto (columnas sin reordenar): **−10**.
- Notebook que no corre de principio a fin: **−15**.

---

> **Aviso:** Esta práctica es un ejercicio académico. Los rendimientos pasados no garantizan rendimientos futuros, y nada de lo que se haga aquí constituye una recomendación de inversión.
