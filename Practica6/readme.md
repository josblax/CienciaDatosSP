# Práctica 6 
## Detección de Fraudes y Análisis de Riesgo en Transacciones

#### ==========================================================
### Práctica 6 — Detección de Fraudes y Análisis de Riesgo
### en Transacciones
### Autopráctica guiada

#### Datos (sintéticos): script anexo del practicario CDT015,
####   `practica6_datos_alumno.py` (semilla 42, 1,200 transacciones).
####   Colocar el script en la misma carpeta que este notebook.
#### ==========================================================

**Caso:** una institución bancaria quiere un sistema de monitoreo inteligente que detecte transacciones potencialmente fraudulentas, evalúe el riesgo de crédito de sus clientes y proponga estrategias de vigilancia automatizada.

**Cómo usar este notebook**

1. Lee las **instrucciones de Python** de cada parte: qué hace cada función y si transforma los datos o solo te da información.
2. Completa las celdas de código donde aparece `___` (guion bajo triple).
3. Si te atoras, abre la **Pista**. Si aún así no sale, compara con **Ver solución**.
4. Revisa la sección **Verifica tu resultado** para comprobar que vas bien.
5. Lee los **Hints** y escribe tu interpretación en la celda **Tu interpretación**. Esa es la parte más importante: el código solo es el medio.

> **Requisito previo:** ejecutar `practica6_datos_alumno.py` en la misma carpeta, que genera `datos_financieros.csv`.

---
## Parte 0. Preparación del entorno

#### Instrucciones de Python que usarás

* **`import pandas as pd`**
    * *Tipo:* Configuración.
    * *Explicación:* Carga la librería para manejar tablas (DataFrames). Por convención se abrevia `pd`.
* **`np.random.seed(42)`**
    * *Tipo:* Configuración.
    * *Explicación:* Fija la semilla aleatoria para que todos obtengan los mismos resultados al ejecutar el notebook.
* **`pd.read_csv("archivo.csv")`**
    * *Tipo:* Información.
    * *Explicación:* Lee un archivo CSV y lo convierte en un DataFrame.
* **`df.shape / df.head()`**
    * *Tipo:* Información.
    * *Explicación:* `shape` devuelve (filas, columnas); `head()` muestra los primeros 5 registros para revisar que la carga fue correcta.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
pd.set_option("display.max_columns", None)

df = pd.read_csv("datos_financieros.csv")
variables = ["monto_transaccion", "frecuencia_transacciones", "historial_crediticio"]

print(df.shape)
df.head()
```

**Verifica tu resultado:** `(1200, 7)`: 1,200 transacciones y 7 columnas.

---
## Parte 1. Análisis exploratorio de transacciones

Antes de construir cualquier modelo hay que entender cómo se comportan las transacciones: cuánto se mueve, con qué frecuencia y qué tan común es el fraude.

#### Instrucciones de Python que usarás

* **`df.isna().sum()`**
    * *Tipo:* Información.
    * *Explicación:* `isna()` marca los valores nulos y `sum()` los cuenta por columna. Te dice rápidamente si falta información.
* **`df.dtypes`**
    * *Tipo:* Información.
    * *Explicación:* Muestra el tipo de dato de cada columna (número entero, decimal o texto). Las columnas de texto tendrán que convertirse a números antes de modelar.
* **`df.describe()`**
    * *Tipo:* Información.
    * *Explicación:* Resumen estadístico: conteo, media, desviación estándar, mínimo, cuartiles y máximo.
* **`serie.median() / serie.skew()`**
    * *Tipo:* Información.
    * *Explicación:* `median()` es el valor central; `skew()` mide la asimetría. Un valor positivo indica una cola larga hacia la derecha (pocos valores muy grandes).
* **`df["fraude"].mean()`**
    * *Tipo:* Información.
    * *Explicación:* En una variable 0/1, el promedio es la proporción de unos; es decir, la **tasa de fraude**.
* **`serie.value_counts()`**
    * *Tipo:* Información.
    * *Explicación:* Cuenta cuántos registros hay en cada categoría.
* **`df.groupby("col")["fraude"].mean()`**
    * *Tipo:* Información.
    * *Explicación:* Agrupa las transacciones por categoría y calcula la tasa de fraude de cada grupo.
* **`sns.histplot(...) / sns.countplot(...)`**
    * *Tipo:* Información (visual).
    * *Explicación:* `histplot` dibuja la distribución de una variable numérica; `countplot` cuenta registros por categoría. Con `hue="fraude"` separa los colores por clase.

### 1.1 Revisar estructura y faltantes

Confirma los tipos de dato y que no haya valores nulos.

```python
print(df.___)
df.___().sum()
```

<details><summary><b>Ver solución</b></summary>

```python
print(df.dtypes)
df.isna().sum()
```

</details>

**Verifica tu resultado:** 0 nulos en todas las columnas. `tipo_transaccion` y `ubicacion` son de texto; el resto son numéricas.

### 1.2 Estadística descriptiva

Obtén el resumen de las variables numéricas y compara la media y la mediana del monto.

```python
print(df[variables].___().round(2))
print("Mediana del monto:", df["monto_transaccion"].___())
print("Asimetría del monto:", round(df["monto_transaccion"].___(), 2))
```

<details><summary><b>Pista</b></summary>

Las tres funciones aparecen en las instrucciones de arriba: resumen estadístico, valor central y asimetría.

</details>

<details><summary><b>Ver solución</b></summary>

```python
print(df[variables].describe().round(2))
print("Mediana del monto:", df["monto_transaccion"].median())
print("Asimetría del monto:", round(df["monto_transaccion"].skew(), 2))
```

</details>

**Verifica tu resultado:** monto promedio de $1,523.86, mediana de $1,089.90 y asimetría de 2.1. Score crediticio entre 300 y 849.

### 1.3 Tasa de fraude por categoría

Calcula la tasa de fraude general y por tipo de transacción y ubicación.

```python
print("Tasa de fraude:", round(df["fraude"].___() * 100, 1), "%")
print(df["tipo_transaccion"].___())
print(df.___("tipo_transaccion")["fraude"].mean().round(3))
print(df.___("ubicacion")["fraude"].mean().round(3))
```

<details><summary><b>Ver solución</b></summary>

```python
print("Tasa de fraude:", round(df["fraude"].mean() * 100, 1), "%")
print(df["tipo_transaccion"].value_counts())
print(df.groupby("tipo_transaccion")["fraude"].mean().round(3))
print(df.groupby("ubicacion")["fraude"].mean().round(3))
```

</details>

**Verifica tu resultado:** tasa de fraude de 68.0%. Por tipo y por ubicación, todas las tasas quedan entre 0.645 y 0.696.

### 1.4 Gráficas de distribución

Grafica la distribución del monto, la frecuencia y el tipo de transacción separado por clase.

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.___(df["monto_transaccion"], bins=___, kde=True, ax=axes[0])
axes[0].set_title("Monto de transacción")
sns.countplot(x="frecuencia_transacciones", data=df, ax=axes[1])
axes[1].set_title("Frecuencia de transacciones")
sns.___(x="tipo_transaccion", hue=___, data=df, ax=axes[2])
axes[2].set_title("Tipo de transacción por clase")
plt.tight_layout(); plt.show()
```

<details><summary><b>Pista</b></summary>

Para una variable continua usa un histograma (40 barras funciona bien); para una categórica, un conteo. `hue` recibe el nombre de la columna que define los colores.

</details>

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df["monto_transaccion"], bins=40, kde=True, ax=axes[0])
axes[0].set_title("Monto de transacción")
sns.countplot(x="frecuencia_transacciones", data=df, ax=axes[1])
axes[1].set_title("Frecuencia de transacciones")
sns.countplot(x="tipo_transaccion", hue="fraude", data=df, ax=axes[2])
axes[2].set_title("Tipo de transacción por clase")
plt.tight_layout(); plt.show()
```

</details>

**Verifica tu resultado:** el histograma del monto tiene una cola larga a la derecha; la frecuencia es casi uniforme entre 1 y 19; en los cuatro tipos de transacción la barra de fraude es más alta que la de normales.

#### Hints para interpretar tus resultados

* **Hint de distribución:** La media del monto ($1,524) es mucho mayor que la mediana ($1,090). *¿Qué te dice eso sobre unas cuantas transacciones muy grandes? ¿Cuál de las dos medidas describe mejor la transacción "típica"?*
* **Hint de tasa de fraude:** En este dataset el 68% de las transacciones son fraude. *En un banco real el fraude suele ser menos del 1%. ¿Qué tan realista es este escenario y qué cambiaría si el fraude fuera muy raro?*
* **Hint de categorías:** Las tasas de fraude por tipo y ubicación son muy parecidas (entre 65% y 70%). *¿Alcanza una diferencia de 3 o 4 puntos para decir que un canal es más riesgoso que otro, con solo ~300 transacciones por grupo?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 2. Identificación de patrones anormales

#### Instrucciones de Python que usarás

* **`df.groupby("fraude")[cols].mean()`**
    * *Tipo:* Información.
    * *Explicación:* Compara el promedio de cada variable entre transacciones normales (0) y fraudulentas (1).
* **`df[vars].corrwith(df["fraude"])`**
    * *Tipo:* Información.
    * *Explicación:* Correlación lineal de cada variable contra el fraude. Indica qué variables se relacionan más con él.
* **`serie.quantile([0.25, 0.75])`**
    * *Tipo:* Información.
    * *Explicación:* Devuelve el primer cuartil (Q1) y el tercer cuartil (Q3). La diferencia Q3 − Q1 es el **rango intercuartílico (IQR)**.
* **`Q3 + 1.5 * IQR`**
    * *Tipo:* Cálculo.
    * *Explicación:* Límite superior de la regla IQR. Los valores por encima se consideran **atípicos**.
* **`(condición).astype(int)`**
    * *Tipo:* Transformación.
    * *Explicación:* Convierte un resultado `True`/`False` en una bandera 0/1 que se puede guardar como columna.
* **`sns.scatterplot(...) / sns.boxplot(...)`**
    * *Tipo:* Información (visual).
    * *Explicación:* El diagrama de dispersión muestra la relación entre dos variables; el diagrama de caja muestra la mediana, los cuartiles y los atípicos de una variable por grupo.

### 2.1 Comparar transacciones normales y fraudulentas

```python
df.___("fraude")[variables].___().round(2)
```

<details><summary><b>Ver solución</b></summary>

```python
df.groupby("fraude")[variables].mean().round(2)
```

</details>

**Verifica tu resultado:** monto promedio de $1,021.14 en normales y de $1,760.43 en fraudes. La frecuencia y el score casi no cambian entre grupos.

### 2.2 Variables más relacionadas con el fraude

Ordena las variables numéricas según su correlación con el fraude.

```python
df[variables].___(df[___]).sort_values(ascending=___).round(3)
```

<details><summary><b>Pista</b></summary>

Necesitas una función que calcule la correlación de cada columna contra **otra** columna, no contra todas. Para ordenar de mayor a menor, `ascending` debe ser falso.

</details>

<details><summary><b>Ver solución</b></summary>

```python
df[variables].corrwith(df["fraude"]).sort_values(ascending=False).round(3)
```

</details>

**Verifica tu resultado:** monto 0.224, frecuencia 0.059 e historial crediticio 0.011.

### 2.3 Transacciones atípicas (regla IQR)

Calcula el límite de montos atípicos, marca las transacciones que lo superan y compara su tasa de fraude.

```python
q1, q3 = df["monto_transaccion"].___([0.25, 0.75])
iqr = ___ - ___
limite = q3 + ___ * iqr
df["atipico"] = (df["monto_transaccion"] > limite).astype(___)
print("Límite:", round(limite, 2), "| Atípicos:", df["atipico"].sum())
df.groupby("atipico")["fraude"].mean().round(3)
```

<details><summary><b>Pista</b></summary>

El IQR es la distancia entre el tercer y el primer cuartil. La regla clásica multiplica el IQR por 1.5.

</details>

<details><summary><b>Ver solución</b></summary>

```python
q1, q3 = df["monto_transaccion"].quantile([0.25, 0.75])
iqr = q3 - q1
limite = q3 + 1.5 * iqr
df["atipico"] = (df["monto_transaccion"] > limite).astype(int)
print("Límite:", round(limite, 2), "| Atípicos:", df["atipico"].sum())
df.groupby("atipico")["fraude"].mean().round(3)
```

</details>

**Verifica tu resultado:** límite de $4,691.42 y 54 transacciones atípicas. Tasa de fraude de 0.944 en atípicas contra 0.668 en el resto.

### 2.4 Visualizar los patrones

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.___(data=df, x="frecuencia_transacciones", y="monto_transaccion", hue="fraude", alpha=0.6, ax=axes[0])
axes[0].axhline(limite, color="red", ls="--", label="límite IQR"); axes[0].legend()
axes[0].set_title("Monto vs frecuencia")
sns.___(data=df, x="fraude", y="monto_transaccion", ax=axes[1])
axes[1].set_title("Monto por clase (0 = normal, 1 = fraude)")
plt.tight_layout(); plt.show()
```

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.scatterplot(data=df, x="frecuencia_transacciones", y="monto_transaccion", hue="fraude", alpha=0.6, ax=axes[0])
axes[0].axhline(limite, color="red", ls="--", label="límite IQR"); axes[0].legend()
axes[0].set_title("Monto vs frecuencia")
sns.boxplot(data=df, x="fraude", y="monto_transaccion", ax=axes[1])
axes[1].set_title("Monto por clase (0 = normal, 1 = fraude)")
plt.tight_layout(); plt.show()
```

</details>

**Verifica tu resultado:** casi todos los puntos por encima de la línea roja son fraudes; la caja de la clase 1 está más arriba que la de la clase 0, aunque ambas se traslapan mucho.

#### Hints para interpretar tus resultados

* **Hint de correlación:** El monto es la variable más relacionada con el fraude (0.224), pero es una correlación débil. *¿Qué tan fácil crees que será para un modelo separar ambas clases?*
* **Hint de atípicos:** El 94% de las transacciones atípicas son fraude. *¿Bloquearías automáticamente toda transacción mayor a $4,691? ¿Qué pasa con el 6% de clientes legítimos que hicieron una compra grande?*
* **Hint de score crediticio:** El historial crediticio casi no se relaciona con el fraude (0.011). *¿Por qué tiene sentido? Piensa en qué mide un score crediticio y qué es un fraude (por ejemplo, una tarjeta robada).*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 3. Desarrollo del modelo de detección de fraude

#### Instrucciones de Python que usarás

* **`pd.get_dummies(df[cols], dtype=int)`**
    * *Tipo:* Transformación.
    * *Explicación:* Convierte cada categoría de texto en una columna 0/1 (codificación *one-hot*). Por ejemplo, `ubicacion` se vuelve `ubicacion_online`, `ubicacion_cajero`, etc. Los modelos solo trabajan con números.
* **`train_test_split(X, y, test_size=0.2, stratify=y)`**
    * *Tipo:* Transformación.
    * *Explicación:* Separa los datos en entrenamiento (80%) y prueba (20%). `stratify=y` garantiza que ambos conjuntos tengan la misma tasa de fraude (68%).
* **`make_pipeline(StandardScaler(), LogisticRegression())`**
    * *Tipo:* Modelado.
    * *Explicación:* Primero estandariza las variables (media 0, desviación 1) y luego aplica la regresión logística. La logística necesita escalas comparables; los árboles no.
* **`DecisionTreeClassifier(max_depth=4)`**
    * *Tipo:* Modelado.
    * *Explicación:* Árbol de decisión limitado a 4 niveles de preguntas, para que no memorice los datos de entrenamiento (sobreajuste).
* **`RandomForestClassifier(n_estimators=200)`**
    * *Tipo:* Modelado.
    * *Explicación:* Conjunto de 200 árboles entrenados con muestras distintas; la predicción final es la votación de todos. Suele ser más estable que un solo árbol.
* **`class_weight="balanced"`**
    * *Tipo:* Modelado.
    * *Explicación:* Le da más peso a los errores en la clase menos frecuente, para que el modelo no se limite a predecir siempre la clase mayoritaria.
* **`modelo.fit(X, y)`**
    * *Tipo:* Modelado.
    * *Explicación:* Entrena el modelo: aprende la relación entre las variables `X` y el fraude `y`.

### 3.1 Codificar las variables categóricas

Construye la matriz de variables `X` (sin `id_cliente`, que es solo un identificador) y la variable objetivo `y`.

```python
X = pd.___(df[variables + ["tipo_transaccion", ___]], dtype=int)
y = df[___]
print(X.shape)
list(X.columns)
```

<details><summary><b>Ver solución</b></summary>

```python
X = pd.get_dummies(df[variables + ["tipo_transaccion", "ubicacion"]], dtype=int)
y = df["fraude"]
print(X.shape)
list(X.columns)
```

</details>

**Verifica tu resultado:** `(1200, 11)`: las 3 variables numéricas, 4 columnas de tipo de transacción y 4 de ubicación.

### 3.2 División 80/20 estratificada

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=___, random_state=42, stratify=___)
print(len(X_train), len(X_test))
print("Tasa de fraude train:", round(y_train.mean(), 3), "| test:", round(y_test.mean(), 3))
```

<details><summary><b>Ver solución</b></summary>

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(len(X_train), len(X_test))
print("Tasa de fraude train:", round(y_train.mean(), 3), "| test:", round(y_test.mean(), 3))
```

</details>

**Verifica tu resultado:** 960 transacciones para entrenamiento y 240 para prueba; tasa de fraude de 0.68 en ambos.

### 3.3 Entrenar los seis modelos

Entrena regresión logística, árbol de decisión y Random Forest, cada uno en versión estándar y balanceada.

```python
def crear_modelos(peso):
    return {
        "Regresion logistica": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000, class_weight=peso)),
        "Arbol de decision": DecisionTreeClassifier(max_depth=___, min_samples_leaf=20,
                                                    random_state=42, class_weight=peso),
        "Random Forest": RandomForestClassifier(n_estimators=___, max_depth=6, min_samples_leaf=10,
                                                random_state=42, class_weight=peso),
    }

entrenados = {}
for version, peso in [("estandar", None), ("balanceado", ___)]:
    for nombre, modelo in crear_modelos(peso).items():
        modelo.___(X_train, y_train)
        entrenados[(nombre, version)] = modelo
list(entrenados.keys())
```

<details><summary><b>Pista</b></summary>

La versión estándar no lleva pesos (`None`); la balanceada usa el texto `"balanced"`. Revisa las instrucciones para la profundidad del árbol y el número de árboles del bosque.

</details>

<details><summary><b>Ver solución</b></summary>

```python
def crear_modelos(peso):
    return {
        "Regresion logistica": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000, class_weight=peso)),
        "Arbol de decision": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                    random_state=42, class_weight=peso),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10,
                                                random_state=42, class_weight=peso),
    }

entrenados = {}
for version, peso in [("estandar", None), ("balanceado", "balanced")]:
    for nombre, modelo in crear_modelos(peso).items():
        modelo.fit(X_train, y_train)
        entrenados[(nombre, version)] = modelo
list(entrenados.keys())
```

</details>

**Verifica tu resultado:** 6 combinaciones de modelo y versión.

#### Hints para interpretar tus resultados

* **Hint de codificación:** *¿Por qué no basta con asignar números a las categorías (compra = 1, retiro = 2, transferencia = 3…)? ¿Qué relación de orden estarías inventando?*
* **Hint de identificadores:** *¿Qué pasaría si dejaras `id_cliente` como variable? ¿Un número de cliente más alto hace más probable el fraude?*
* **Hint de balanceo:** *En el modelo estándar, equivocarse con una transacción normal "cuesta" lo mismo que equivocarse con un fraude, pero hay el doble de fraudes. ¿Hacia qué clase crees que se va a inclinar el modelo?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 4. Evaluación del modelo

#### Instrucciones de Python que usarás

* **`modelo.predict(X)`**
    * *Tipo:* Información.
    * *Explicación:* Devuelve la predicción 0/1 para cada transacción.
* **`modelo.predict_proba(X)[:, 1]`**
    * *Tipo:* Información.
    * *Explicación:* Devuelve la probabilidad de que cada transacción sea fraude (columna 1). `predict` equivale a usar un umbral de 0.5 sobre esta probabilidad.
* **`confusion_matrix(y_real, y_pred).ravel()`**
    * *Tipo:* Información.
    * *Explicación:* Cuenta aciertos y errores. `ravel()` la aplana en el orden **tn, fp, fn, tp** (verdaderos negativos, falsos positivos, falsos negativos, verdaderos positivos).
* **`tp / (tp + fn)`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Sensibilidad (recall):** de todos los fraudes reales, qué proporción detectó el modelo.
* **`tn / (tn + fp)`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Especificidad:** de todas las transacciones normales, qué proporción reconoció correctamente como normales.
* **`(sensibilidad + especificidad) / 2`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Exactitud balanceada:** promedio de ambas. Un modelo que predice siempre la misma clase obtiene 0.5, sin importar qué tan desbalanceados estén los datos.
* **`accuracy_score(y_real, y_pred)`**
    * *Tipo:* Información.
    * *Explicación:* Proporción total de aciertos, sin distinguir el tipo de error.
* **`modelo.feature_importances_ / modelo[-1].coef_`**
    * *Tipo:* Información.
    * *Explicación:* En árboles y bosques, cuánto contribuye cada variable a las decisiones; en la logística (último paso del pipeline), el peso de cada variable.

### 4.1 Métricas de los seis modelos

Calcula accuracy, sensibilidad, especificidad, exactitud balanceada y errores de cada modelo en el conjunto de prueba.

```python
resultados = []
for (nombre, version), m in entrenados.items():
    pred = m.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    sens = ___ / (___ + ___)
    espec = ___ / (___ + ___)
    resultados.append({
        "modelo": nombre, "version": version,
        "accuracy": accuracy_score(y_test, pred),
        "sensibilidad": sens, "especificidad": espec,
        "exactitud_balanceada": (sens + espec) / ___,
        "falsos_negativos": fn, "falsos_positivos": fp,
    })
tabla = pd.DataFrame(resultados).round(3)
print("Línea base (predecir siempre fraude):", round(y_test.mean(), 3))
tabla
```

<details><summary><b>Pista</b></summary>

La sensibilidad mira solo a los fraudes reales (los que el modelo detectó más los que se le escaparon). La especificidad mira solo a las transacciones normales reales.

</details>

<details><summary><b>Ver solución</b></summary>

```python
resultados = []
for (nombre, version), m in entrenados.items():
    pred = m.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    sens = tp / (tp + fn)
    espec = tn / (tn + fp)
    resultados.append({
        "modelo": nombre, "version": version,
        "accuracy": accuracy_score(y_test, pred),
        "sensibilidad": sens, "especificidad": espec,
        "exactitud_balanceada": (sens + espec) / 2,
        "falsos_negativos": fn, "falsos_positivos": fp,
    })
tabla = pd.DataFrame(resultados).round(3)
print("Línea base (predecir siempre fraude):", round(y_test.mean(), 3))
tabla
```

</details>

**Verifica tu resultado:** línea base de 0.679.

| Modelo | Versión | Accuracy | Sensibilidad | Especificidad | Exactitud bal. | FN | FP |
|---|---|---|---|---|---|---|---|
| Logística | estándar | 0.692 | 1.000 | 0.039 | 0.519 | 0 | 74 |
| Árbol | estándar | 0.671 | 0.957 | 0.065 | 0.511 | 7 | 72 |
| Random Forest | estándar | 0.679 | 1.000 | 0.000 | 0.500 | 0 | 77 |
| Logística | balanceado | 0.617 | 0.564 | 0.727 | 0.646 | 71 | 21 |
| Árbol | balanceado | 0.483 | 0.344 | 0.779 | 0.561 | 107 | 17 |
| Random Forest | balanceado | 0.554 | 0.509 | 0.649 | 0.579 | 80 | 27 |

### 4.2 Matrices de confusión

Visualiza las seis matrices de confusión.

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, ((nombre, version), m) in zip(axes.ravel(), entrenados.items()):
    cm = ___(y_test, m.___(X_test))
    sns.heatmap(cm, annot=___, fmt=___, cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
    ax.set_title(f"{nombre}\n({version})", fontsize=9)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

<details><summary><b>Pista</b></summary>

La matriz se calcula comparando la clase real contra la predicción del modelo. En el mapa de calor, `fmt="d"` muestra los valores como enteros (son conteos de transacciones). La celda de abajo a la izquierda (real = fraude, predicho = normal) son los falsos negativos: fraudes que el banco dejaría pasar.

</details>

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, ((nombre, version), m) in zip(axes.ravel(), entrenados.items()):
    cm = confusion_matrix(y_test, m.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
    ax.set_title(f"{nombre}\n({version})", fontsize=9)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

</details>

**Verifica tu resultado:** en los tres modelos estándar casi toda la matriz se concentra en la columna "Fraude"; en los balanceados aparecen aciertos en ambas columnas.

### 4.3 Selección del modelo

Elige el modelo con mayor exactitud balanceada.

```python
mejor = tabla.sort_values(___, ascending=___).iloc[0]
print("Modelo seleccionado:", mejor["modelo"], "-", mejor["version"])
modelo_final = entrenados[(mejor["modelo"], mejor["version"])]
```

<details><summary><b>Ver solución</b></summary>

```python
mejor = tabla.sort_values("exactitud_balanceada", ascending=False).iloc[0]
print("Modelo seleccionado:", mejor["modelo"], "-", mejor["version"])
modelo_final = entrenados[(mejor["modelo"], mejor["version"])]
```

</details>

**Verifica tu resultado:** Regresión logística balanceada.

### 4.4 Variables más importantes

Compara la importancia de variables del Random Forest balanceado con los coeficientes de la logística balanceada.

```python
rf = entrenados[("Random Forest", "balanceado")]
print(pd.Series(rf.___, index=X.columns).sort_values(ascending=False).round(3).head(5))

lr = entrenados[("Regresion logistica", "balanceado")]
print(pd.Series(lr[-1].___[0], index=X.columns).sort_values(ascending=False).round(3).head(5))
```

<details><summary><b>Ver solución</b></summary>

```python
rf = entrenados[("Random Forest", "balanceado")]
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).round(3).head(5))

lr = entrenados[("Regresion logistica", "balanceado")]
print(pd.Series(lr[-1].coef_[0], index=X.columns).sort_values(ascending=False).round(3).head(5))
```

</details>

**Verifica tu resultado:** en ambos modelos el monto ocupa el primer lugar (importancia 0.491; coeficiente 0.655). El Random Forest pone al historial crediticio en segundo lugar (0.219), pero en la logística su peso es casi cero.

### 4.5 Ajuste del umbral de decisión

Observa cómo cambian los errores del modelo final al mover el umbral.

```python
prob = modelo_final.___(X_test)[:, ___]
for umbral in [0.4, 0.5, 0.6]:
    pred_u = (prob >= ___).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_u).ravel()
    print(f"Umbral {umbral}: fraudes detectados = {tp}, no detectados = {fn}, falsas alarmas = {fp}")
```

<details><summary><b>Ver solución</b></summary>

```python
prob = modelo_final.predict_proba(X_test)[:, 1]
for umbral in [0.4, 0.5, 0.6]:
    pred_u = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_u).ravel()
    print(f"Umbral {umbral}: fraudes detectados = {tp}, no detectados = {fn}, falsas alarmas = {fp}")
```

</details>

**Verifica tu resultado:**

| Umbral | Detectados (TP) | No detectados (FN) | Falsas alarmas (FP) |
|---|---|---|---|
| 0.4 | 144 | 19 | 55 |
| 0.5 | 92 | 71 | 21 |
| 0.6 | 47 | 116 | 9 |

#### Hints para interpretar tus resultados

* **Hint de accuracy:** El Random Forest estándar tiene 67.9% de accuracy, exactamente la línea base, y una especificidad de 0. *¿Qué está haciendo realmente ese modelo? ¿Lo pondrías en producción? ¿Es suficiente el accuracy para evaluar un detector de fraude?*
* **Hint de errores:** *Compara el costo de un falso negativo (fraude que pasa) contra el de un falso positivo (cliente legítimo bloqueado). ¿Cuál le cuesta más al banco? ¿Cuál le molesta más al cliente?*
* **Hint de umbral:** Bajar el umbral de 0.5 a 0.4 detecta 52 fraudes más, pero genera 34 falsas alarmas más. *¿Quién debería decidir el umbral: el científico de datos o la dirección de seguridad financiera?*
* **Hint de importancia:** *¿Por qué el Random Forest le da importancia al historial crediticio si su correlación con el fraude es casi cero? Pista: la importancia por impureza favorece a las variables con muchos valores distintos.*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 5. Análisis de riesgo crediticio

El riesgo de fraude y el riesgo de crédito son preguntas distintas: el primero pregunta si **esta transacción** es legítima; el segundo, si **este cliente** pagará lo que se le preste.

#### Instrucciones de Python que usarás

* **`df.groupby("id_cliente").agg(nueva=("col", "función"))`**
    * *Tipo:* Transformación.
    * *Explicación:* Resume todas las transacciones de cada cliente en una sola fila. Cada argumento define una nueva columna: de qué columna sale y qué función se aplica (`"mean"`, `"size"`, `"sum"`).
* **`pd.cut(serie, bins=cortes, labels=etiquetas, right=False)`**
    * *Tipo:* Transformación.
    * *Explicación:* Convierte una variable continua (el score) en categorías (niveles de riesgo). `right=False` hace que cada intervalo incluya su límite inferior.
* **`serie.value_counts()`**
    * *Tipo:* Información.
    * *Explicación:* Cuenta cuántos clientes hay en cada nivel.

### 5.1 Resumen por cliente

```python
clientes = df.___("id_cliente").agg(
    score_promedio=("historial_crediticio", ___),
    num_transacciones=("fraude", "size"),
    monto_promedio=("monto_transaccion", "mean"),
    tasa_fraude=("fraude", "mean"),
).reset_index()
print(clientes.shape)
clientes.head()
```

<details><summary><b>Pista</b></summary>

En este dataset el score de un mismo cliente puede cambiar entre transacciones, así que se usa su promedio.

</details>

<details><summary><b>Ver solución</b></summary>

```python
clientes = df.groupby("id_cliente").agg(
    score_promedio=("historial_crediticio", "mean"),
    num_transacciones=("fraude", "size"),
    monto_promedio=("monto_transaccion", "mean"),
    tasa_fraude=("fraude", "mean"),
).reset_index()
print(clientes.shape)
clientes.head()
```

</details>

**Verifica tu resultado:** `(681, 5)`: 681 clientes, con 1.76 transacciones en promedio (máximo 8).

### 5.2 Clasificación por nivel de riesgo

Usa rangos similares a los de las sociedades de información crediticia: alto riesgo por debajo de 580, riesgo medio de 580 a 669 y bajo riesgo de 670 en adelante.

```python
cortes = [0, ___, ___, 851]
niveles = ["Alto riesgo", "Riesgo medio", "Bajo riesgo"]
clientes["nivel_riesgo"] = pd.___(clientes["score_promedio"], bins=cortes, labels=niveles, right=False)
clientes["nivel_riesgo"].___()
```

<details><summary><b>Ver solución</b></summary>

```python
cortes = [0, 580, 670, 851]
niveles = ["Alto riesgo", "Riesgo medio", "Bajo riesgo"]
clientes["nivel_riesgo"] = pd.cut(clientes["score_promedio"], bins=cortes, labels=niveles, right=False)
clientes["nivel_riesgo"].value_counts()
```

</details>

**Verifica tu resultado:** 359 clientes de alto riesgo, 129 de riesgo medio y 193 de bajo riesgo.

### 5.3 Perfil de cada nivel

```python
clientes.groupby("nivel_riesgo", observed=True)[["score_promedio", "monto_promedio", "tasa_fraude"]].___().round(2)
```

<details><summary><b>Ver solución</b></summary>

```python
clientes.groupby("nivel_riesgo", observed=True)[["score_promedio", "monto_promedio", "tasa_fraude"]].mean().round(2)
```

</details>

**Verifica tu resultado:** tasa de fraude de 0.68, 0.69 y 0.67 en los niveles alto, medio y bajo; prácticamente igual en los tres.

#### Hints para interpretar tus resultados

* **Hint de riesgo vs. fraude:** La tasa de fraude es la misma en los tres niveles. *¿Significa que el score crediticio no sirve, o que mide otra cosa? ¿Por qué un banco necesita dos modelos distintos?*
* **Hint de concentración:** Más de la mitad de los clientes (53%) quedan en alto riesgo. *Con ese portafolio, ¿qué políticas de crédito propondrías para cada nivel (límites, tasas, garantías, aprobación automática)?*
* **Hint de datos:** El score de un mismo cliente cambia entre sus transacciones. *En un banco real, ¿de dónde vendría el score y cada cuánto se actualizaría?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte y tu propuesta de políticas de crédito para cada nivel.)*

---
## Parte 6. Propuesta de vigilancia automatizada

Aquí el modelo deja de ser un ejercicio técnico y se convierte en una decisión de operación: qué hacer con cada transacción en el momento en que ocurre.

#### Instrucciones de Python que usarás

* **`np.select([cond1, cond2], ["A", "B"], default="C")`**
    * *Tipo:* Transformación.
    * *Explicación:* Asigna `"A"` donde se cumple la primera condición, `"B"` donde se cumple la segunda (y no la primera) y `"C"` en el resto. Sirve para traducir probabilidades en acciones.
* **`(cond1) | (cond2)`**
    * *Tipo:* Transformación.
    * *Explicación:* Operador "o": es verdadero si se cumple al menos una condición. Cada condición va entre paréntesis.
* **`pd.crosstab(fila, columna)`**
    * *Tipo:* Información.
    * *Explicación:* Tabla de conteo cruzado. Aquí muestra cuántos fraudes y cuántas transacciones normales terminan en cada acción.

### 6.1 Reglas de decisión en tiempo real

Combina el modelo con la regla de montos atípicos para decidir una de tres acciones por transacción.

* **BLOQUEAR:** probabilidad de fraude ≥ 0.8 **o** monto atípico.
* **REVISAR** (autenticación adicional o analista): probabilidad ≥ 0.5.
* **APROBAR:** el resto.

```python
UMBRAL_REVISION = ___
UMBRAL_BLOQUEO = ___
monto_atipico = X_test["monto_transaccion"] > ___

accion = np.___(
    [(prob >= UMBRAL_BLOQUEO) ___ monto_atipico, prob >= UMBRAL_REVISION],
    ["BLOQUEAR", "REVISAR"], default=___)
pd.Series(accion).value_counts()
```

<details><summary><b>Pista</b></summary>

Reutiliza `prob` de 4.5 y `limite` de 2.3. El operador "o" en pandas y NumPy es `|`. El orden de las condiciones importa: la primera que se cumple gana.

</details>

<details><summary><b>Ver solución</b></summary>

```python
UMBRAL_REVISION = 0.5
UMBRAL_BLOQUEO = 0.8
monto_atipico = X_test["monto_transaccion"] > limite

accion = np.select(
    [(prob >= UMBRAL_BLOQUEO) | monto_atipico, prob >= UMBRAL_REVISION],
    ["BLOQUEAR", "REVISAR"], default="APROBAR")
pd.Series(accion).value_counts()
```

</details>

**Verifica tu resultado:** 127 transacciones aprobadas, 105 a revisión y 8 bloqueadas.

### 6.2 Impacto del esquema

```python
pd.___(pd.Series(accion, name="accion"), y_test.reset_index(drop=True).rename("fraude"))
```

<details><summary><b>Ver solución</b></summary>

```python
pd.crosstab(pd.Series(accion, name="accion"), y_test.reset_index(drop=True).rename("fraude"))
```

</details>

**Verifica tu resultado:**

| Acción | Normal (0) | Fraude (1) |
|---|---|---|
| APROBAR | 56 | 71 |
| BLOQUEAR | 0 | 8 |
| REVISAR | 21 | 84 |

#### Hints para interpretar tus resultados

* **Hint de bloqueo:** Las 8 transacciones bloqueadas son fraude y ningún cliente legítimo fue bloqueado. *¿Por qué conviene reservar el bloqueo automático solo para los casos más claros?*
* **Hint de revisión:** Cambia `UMBRAL_REVISION` a 0.4 y vuelve a ejecutar 6.1 y 6.2. Los fraudes aprobados bajan de 71 a 19, pero las revisiones suben de 105 a 191. *Si el equipo de analistas solo puede revisar 120 transacciones al día, ¿qué umbral elegirías? ¿Qué pasa si la revisión se hace con autenticación en la app en lugar de un analista?*
* **Hint de tiempo real:** *Este esquema se probó sobre datos históricos. ¿Qué se necesitaría para que funcione en milisegundos sobre cada transacción nueva? Piensa en arquitecturas de streaming (Kappa, Lambda).*
* **Hint de aprendizaje continuo:** *Los defraudadores cambian de táctica. ¿Cómo usarías los casos que los analistas confirman como fraude o falsa alarma para mantener el modelo actualizado?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte y tu esquema de monitoreo en tiempo real.)*

---
## Cierre de la práctica

Antes de redactar tu reporte técnico, responde:

* ¿Qué variables fueron más relevantes para detectar fraude?
* ¿Qué riesgos implica clasificar incorrectamente una transacción?
* ¿Por qué el accuracy puede ser engañoso cuando las clases están desbalanceadas?
* ¿Cómo puede la ciencia de datos mejorar la seguridad bancaria?
* ¿Qué ventajas ofrece el monitoreo automatizado de transacciones?

**Lista de evidencias:** código fuente (este notebook), gráficas de análisis exploratorio, resultados del modelo de detección de fraude, matrices de confusión, tabla comparativa de modelos, clasificación de riesgo crediticio, esquema de vigilancia y reporte técnico de 2–3 cuartillas (interpretación de resultados, análisis de datos y propuesta de mejora en seguridad financiera).
