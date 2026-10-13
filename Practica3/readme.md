# Práctica 3 — Autopráctica guiada
## Aplicación de la Ciencia de Datos en el Sector Privado de la Salud (CDT015)

**Caso:** una clínica privada quiere un sistema de apoyo a la decisión médica que identifique pacientes con alto riesgo de diabetes tipo 2 y use ese resultado para la telemedicina, un programa de adherencia y el abastecimiento de insumos.

**Cómo usar este notebook**

1. Lee las **instrucciones de Python** de cada parte: qué hace cada función y si transforma los datos o solo te da información.
2. Completa las celdas de código donde aparece `___` (guion bajo triple).
3. Si te atoras, abre la **Pista**. Si aún así no sale, compara con **Ver solución**.
4. Revisa la sección **Verifica tu resultado** para comprobar que vas bien.
5. Lee los **Hints** y escribe tu interpretación en la celda **Tu interpretación**. Esa es la parte más importante: el código solo es el medio.

> **Requisito previo:** ejecutar `practica3_datos_alumno.py` en la misma carpeta, que genera `datos_clinicos_practica3.csv` y `consumo_insumos.csv`.

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
    * *Explicación:* Lee un archivo CSV y lo convierte en un DataFrame. Con `parse_dates` indicas qué columnas son fechas.
* **`df.shape / df.head()`**
    * *Tipo:* Información.
    * *Explicación:* `shape` devuelve (filas, columnas); `head()` muestra los primeros 5 registros para revisar que la carga fue correcta.

```python
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
pd.set_option("display.max_columns", None)

df = pd.read_csv("datos_clinicos_practica3.csv")
consumo = pd.read_csv("consumo_insumos.csv", parse_dates=["mes"])
sintomas = ["refiere sed excesiva", "orina frecuente", "vision borrosa",
            "cansancio constante", "heridas que tardan en sanar"]

print(df.shape)
df.head()
```

**Verifica tu resultado:** `(768, 12)`: 768 pacientes y 12 columnas.

---
## Parte 1. Análisis exploratorio

Antes de modelar hay que conocer los datos y, sobre todo, **desconfiar de ellos**. En datos clínicos reales siempre hay errores de captura.

#### Instrucciones de Python que usarás

* **`df.replace(0, np.nan)`**
    * *Tipo:* Transformación.
    * *Explicación:* Reemplaza los ceros por `np.nan` (valor nulo). En datos médicos un 0 en glucosa, presión o IMC es un error de captura, no un valor real; convertirlo a nulo evita que sesgue los promedios hacia abajo.
* **`df.isna().sum()`**
    * *Tipo:* Información.
    * *Explicación:* `isna()` marca los valores nulos y `sum()` los cuenta por columna. Te dice rápidamente dónde falta información.
* **`df.drop(columns=[...])`**
    * *Tipo:* Transformación.
    * *Explicación:* Elimina columnas completas. Se usa cuando una variable tiene tantos faltantes que imputarla inventaría más datos de los que aporta.
* **`df[col].fillna(df[col].median())`**
    * *Tipo:* Transformación.
    * *Explicación:* Rellena (imputa) los nulos con la mediana de la columna.
* **`df.describe()`**
    * *Tipo:* Información.
    * *Explicación:* Resumen estadístico: conteo, media, desviación estándar, mínimo, cuartiles y máximo.
* **`df["diagnostico"].mean()`**
    * *Tipo:* Información.
    * *Explicación:* En una variable 0/1, el promedio es la proporción de unos; es decir, la **prevalencia** de la enfermedad.
* **`sns.heatmap(df.corr(), annot=True)`**
    * *Tipo:* Información (visual).
    * *Explicación:* Dibuja la matriz de correlación como mapa de calor, con el valor escrito en cada celda.
* **`df[vars].corrwith(df["diagnostico"])`**
    * *Tipo:* Información.
    * *Explicación:* Correlación lineal de cada variable contra el diagnóstico. Indica qué variables se relacionan más con la enfermedad.
* **`df["nota_clinica"].str.contains(texto)`**
    * *Tipo:* Transformación.
    * *Explicación:* Busca un texto dentro de cada nota y devuelve `True`/`False`. Con `.astype(int)` se vuelve una bandera 0/1: así se convierte texto libre en una variable numérica (minería de textos básica).
* **`df.groupby("diagnostico")[cols].mean()`**
    * *Tipo:* Información.
    * *Explicación:* Agrupa a los pacientes por diagnóstico y calcula el promedio de cada columna en cada grupo, para compararlos.

### 1.1 Detectar faltantes ocultos

Convierte los ceros clínicamente imposibles en nulos y cuenta cuántos hay por variable.

```python
cols_cero = ["glucosa", "presion_diastolica", "pliegue_cutaneo", "insulina", "imc"]
df[cols_cero] = df[cols_cero].replace(___, ___)
df[cols_cero].___().sum()
```

<details><summary><b>Pista</b></summary>

`replace` recibe primero el valor a buscar y luego el valor nuevo. El nulo en NumPy es `np.nan`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
cols_cero = ["glucosa", "presion_diastolica", "pliegue_cutaneo", "insulina", "imc"]
df[cols_cero] = df[cols_cero].replace(0, np.nan)
df[cols_cero].isna().sum()
```

</details>

**Verifica tu resultado:** glucosa 5, presión 35, pliegue 227, insulina 374, IMC 11.

### 1.2 Limpiar e imputar

Elimina `insulina` y `pliegue_cutaneo` (tienen demasiados faltantes) e imputa con la mediana las otras tres variables.

```python
df = df.drop(columns=[___, ___])
for c in ["glucosa", "presion_diastolica", "imc"]:
    df[c] = df[c].fillna(df[c].___())
df.isna().sum()
```

<details><summary><b>Pista</b></summary>

La única columna que debe conservar nulos es `adherencia`: solo existe para pacientes diagnosticados, así que ahí el nulo **sí** tiene significado.

</details>

<details><summary><b>Ver solución</b></summary>

```python
df = df.drop(columns=["insulina", "pliegue_cutaneo"])
for c in ["glucosa", "presion_diastolica", "imc"]:
    df[c] = df[c].fillna(df[c].median())
df.isna().sum()
```

</details>

**Verifica tu resultado:** 0 nulos en todas las columnas excepto `adherencia` (500, los pacientes sanos).

### 1.3 Estadística descriptiva y prevalencia

Obtén el resumen estadístico de las variables clínicas y la prevalencia del diagnóstico.

```python
variables = ["embarazos", "glucosa", "presion_diastolica", "imc", "historial_familiar", "edad"]
print(df[variables].___().round(2))
print("Prevalencia:", round(df["diagnostico"].___() * 100, 1), "%")
```

<details><summary><b>Ver solución</b></summary>

```python
variables = ["embarazos", "glucosa", "presion_diastolica", "imc", "historial_familiar", "edad"]
print(df[variables].describe().round(2))
print("Prevalencia:", round(df["diagnostico"].mean() * 100, 1), "%")
```

</details>

**Verifica tu resultado:** Prevalencia de 34.9%; edad mediana de 29 años; IMC mediano de 32.3.

### 1.4 Distribuciones y correlación


Grafica la distribución de cada variable y el mapa de calor de correlaciones.

```python
df[variables + ["diagnostico"]].___(figsize=(12, 8), bins=___)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
sns.___(df[variables + ["diagnostico"]].___(), annot=___, fmt=".2f", cmap="Blues")
plt.show()
```

<details><summary><b>Pista</b></summary>

`hist()` dibuja un histograma por cada columna y `bins` define el número de barras (usa 20). Para el mapa de calor, primero calcula la matriz de correlación con `.corr()` y usa `annot=True` para que escriba el valor dentro de cada celda.

</details>

<details><summary><b>Ver solución</b></summary>

```python
df[variables + ["diagnostico"]].hist(figsize=(12, 8), bins=20)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[variables + ["diagnostico"]].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.show()
```

</details>

**Verifica tu resultado:** 7 histogramas (6 variables y el diagnóstico) y un mapa de calor de 7 × 7 con valores entre -1 y 1.

### 1.5 Variables más influyentes

Ordena las variables según su correlación con el diagnóstico.

```python
df[variables].___(df[___]).sort_values(ascending=___).round(3)
```

<details><summary><b>Pista</b></summary>

Necesitas una función que calcule la correlación de cada columna contra **otra** columna, no contra todas. Para ordenar de mayor a menor, `ascending` debe ser falso.

</details>

<details><summary><b>Ver solución</b></summary>

```python
df[variables].corrwith(df["diagnostico"]).sort_values(ascending=False).round(3)
```

</details>

**Verifica tu resultado:** glucosa 0.493, IMC 0.312, edad 0.238 en los tres primeros lugares.

### 1.6 Minería de textos clínicos

Crea una bandera 0/1 por cada síntoma mencionado en la nota clínica y compara su frecuencia entre sanos y enfermos.

```python
for s in sintomas:
    col = "txt_" + s.split()[-1]
    df[col] = df["nota_clinica"].str.___(s, case=False).astype(int)
vars_texto = [c for c in df.columns if c.startswith("txt_")]
df.___("diagnostico")[vars_texto].mean().round(2)
```

<details><summary><b>Pista</b></summary>

`s.split()[-1]` toma la última palabra del síntoma para nombrar la columna (por ejemplo, `txt_borrosa`).

</details>

<details><summary><b>Ver solución</b></summary>

```python
for s in sintomas:
    col = "txt_" + s.split()[-1]
    df[col] = df["nota_clinica"].str.contains(s, case=False).astype(int)
vars_texto = [c for c in df.columns if c.startswith("txt_")]
df.groupby("diagnostico")[vars_texto].mean().round(2)
```

</details>

**Verifica tu resultado:** cada síntoma aparece en ~13–17% de las notas de pacientes sanos y en ~30–37% de las de enfermos.

#### Hints para interpretar tus resultados

* **Hint de limpieza:** ¿Por qué imputar con la mediana y no con el promedio? *Piensa en un paciente con glucosa extremadamente alta (un valor atípico): ¿cómo mueve al promedio y cómo a la mediana?*
* **Hint de calidad de datos:** La insulina tenía 374 faltantes de 768 (casi la mitad). *Si la hubieras imputado, ¿qué proporción de esa columna serían datos reales y qué proporción inventados?*
* **Hint de correlación:** La glucosa es la variable más correlacionada (0.49). *¿Significa eso que la glucosa causa la diabetes, o que la diabetes se manifiesta en la glucosa? Correlación no es causalidad.*
* **Hint de texto:** Los síntomas aparecen el doble de veces en enfermos, pero también aparecen en sanos. *¿Qué te dice eso sobre lo que vale una sola palabra en una nota médica para diagnosticar?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 2. Desarrollo del modelo

#### Instrucciones de Python que usarás

* **`train_test_split(..., test_size=0.2, stratify=y)`**
    * *Tipo:* Transformación.
    * *Explicación:* Separa los datos en entrenamiento (80%) y prueba (20%). `stratify=y` garantiza que ambos conjuntos tengan la misma proporción de enfermos (34.9%).
* **`make_pipeline(StandardScaler(), LogisticRegression())`**
    * *Tipo:* Modelado.
    * *Explicación:* Encadena dos pasos: primero estandariza las variables (media 0, desviación 1) y luego aplica la regresión logística. La logística necesita escalas comparables; el árbol no.
* **`DecisionTreeClassifier(max_depth=4)`**
    * *Tipo:* Modelado.
    * *Explicación:* Árbol de decisión limitado a 4 niveles de preguntas, para que no memorice los datos de entrenamiento (sobreajuste).
* **`clone(modelo)`**
    * *Tipo:* Modelado.
    * *Explicación:* Crea una copia nueva y sin entrenar del modelo. Permite entrenar el mismo tipo de modelo con distintos conjuntos de variables sin que se pisen.
* **`modelo.fit(X, y)`**
    * *Tipo:* Modelado.
    * *Explicación:* Entrena el modelo: aprende la relación entre las variables `X` y el diagnóstico `y`.

### 2.1 División 80/20 estratificada

Define los dos conjuntos de variables (sin texto y con texto) y divide los índices de los pacientes.

```python
X_num = df[variables]
X_txt = df[variables + vars_texto]
y = df["diagnostico"]
idx_train, idx_test = train_test_split(df.index, test_size=___, random_state=42, stratify=___)
print(len(idx_train), len(idx_test))
```

<details><summary><b>Ver solución</b></summary>

```python
X_num = df[variables]
X_txt = df[variables + vars_texto]
y = df["diagnostico"]
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
print(len(idx_train), len(idx_test))
```

</details>

**Verifica tu resultado:** 614 pacientes para entrenamiento y 154 para prueba.

### 2.2 Entrenar los cuatro modelos

Entrena regresión logística y árbol de decisión, cada uno sin texto y con texto.

```python
modelos = {
    "Regresion logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Arbol de decision": DecisionTreeClassifier(max_depth=___, random_state=42),
}
entrenados = {}
for nombre, modelo in modelos.items():
    for etiqueta, X in [("sin texto", X_num), ("con texto", X_txt)]:
        m = clone(modelo)
        m.___(X.loc[idx_train], y.loc[idx_train])
        entrenados[(nombre, etiqueta)] = (m, X)
list(entrenados.keys())
```

<details><summary><b>Ver solución</b></summary>

```python
modelos = {
    "Regresion logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Arbol de decision": DecisionTreeClassifier(max_depth=4, random_state=42),
}
entrenados = {}
for nombre, modelo in modelos.items():
    for etiqueta, X in [("sin texto", X_num), ("con texto", X_txt)]:
        m = clone(modelo)
        m.fit(X.loc[idx_train], y.loc[idx_train])
        entrenados[(nombre, etiqueta)] = (m, X)
list(entrenados.keys())
```

</details>

**Verifica tu resultado:** 4 combinaciones de modelo y conjunto de variables.

#### Hints para interpretar tus resultados

* **Hint de estratificación:** *Si no estratificaras y por azar el conjunto de prueba quedara con solo 20% de enfermos, ¿qué pasaría con la confiabilidad de la sensibilidad que vas a medir?*
* **Hint de sobreajuste:** *Prueba cambiar `max_depth` a `None` (sin límite) y compara después en la Parte 3. ¿El árbol mejora o empeora en el conjunto de prueba?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 3. Evaluación clínica

#### Instrucciones de Python que usarás

* **`modelo.predict(X)`**
    * *Tipo:* Información.
    * *Explicación:* Devuelve la predicción 0/1 para cada paciente.
* **`confusion_matrix(y_real, y_pred).ravel()`**
    * *Tipo:* Información.
    * *Explicación:* Cuenta aciertos y errores. `ravel()` la aplana en el orden **tn, fp, fn, tp** (verdaderos negativos, falsos positivos, falsos negativos, verdaderos positivos).
* **`tp / (tp + fn)`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Sensibilidad:** de todos los enfermos reales, qué proporción detectó el modelo.
* **`tn / (tn + fp)`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Especificidad:** de todos los sanos reales, qué proporción reconoció correctamente como sanos.
* **`accuracy_score(y_real, y_pred)`**
    * *Tipo:* Información.
    * *Explicación:* Proporción total de aciertos, sin distinguir el tipo de error.
* **`df.sort_values([...], ascending=False)`**
    * *Tipo:* Transformación.
    * *Explicación:* Ordena la tabla de mayor a menor según una o varias columnas.

### 3.1 Métricas de los cuatro modelos

Calcula accuracy, sensibilidad, especificidad y falsos negativos para cada modelo en el conjunto de prueba.

```python
resultados = []
for (nombre, etiqueta), (m, X) in entrenados.items():
    pred = m.predict(X.loc[idx_test])
    tn, fp, fn, tp = confusion_matrix(y.loc[idx_test], pred).ravel()
    resultados.append({
        "modelo": nombre, "variables": etiqueta,
        "accuracy": accuracy_score(y.loc[idx_test], pred),
        "sensibilidad": ___ / (___ + ___),
        "especificidad": ___ / (___ + ___),
        "falsos_negativos": fn,
    })
tabla = pd.DataFrame(resultados).round(3)
tabla
```

<details><summary><b>Pista</b></summary>

La sensibilidad mira solo a los enfermos reales (los que el modelo detectó más los que se le escaparon). La especificidad mira solo a los sanos reales.

</details>

<details><summary><b>Ver solución</b></summary>

```python
resultados = []
for (nombre, etiqueta), (m, X) in entrenados.items():
    pred = m.predict(X.loc[idx_test])
    tn, fp, fn, tp = confusion_matrix(y.loc[idx_test], pred).ravel()
    resultados.append({
        "modelo": nombre, "variables": etiqueta,
        "accuracy": accuracy_score(y.loc[idx_test], pred),
        "sensibilidad": tp / (tp + fn),
        "especificidad": tn / (tn + fp),
        "falsos_negativos": fn,
    })
tabla = pd.DataFrame(resultados).round(3)
tabla
```

</details>

**Verifica tu resultado:** 

| Modelo | Variables | Accuracy | Sensibilidad | Especificidad | FN |
|---|---|---|---|---|---|
| Logística | sin texto | 0.701 | 0.481 | 0.82 | 28 |
| Logística | con texto | 0.779 | 0.648 | 0.85 | 19 |
| Árbol | sin texto | 0.792 | 0.704 | 0.84 | 16 |
| Árbol | con texto | 0.727 | 0.481 | 0.86 | 28 |

### 3.2 Matrices de confusión

Visualiza las cuatro matrices de confusión.

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, ((nombre, etiqueta), (m, X)) in zip(axes, entrenados.items()):
    cm = ___(y.loc[idx_test], m.___(X.loc[idx_test]))
    sns.heatmap(cm, annot=___, fmt=___, cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{nombre}\n({etiqueta})", fontsize=9)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

<details><summary><b>Pista</b></summary>

La matriz se calcula comparando el diagnóstico real contra la predicción del modelo. En el mapa de calor, `fmt="d"` muestra los valores como números enteros (son conteos de pacientes). En cada matriz, la celda de abajo a la izquierda (real = 1, predicho = 0) son los falsos negativos: los enfermos que el modelo mandaría a casa.

</details>

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, ((nombre, etiqueta), (m, X)) in zip(axes, entrenados.items()):
    cm = confusion_matrix(y.loc[idx_test], m.predict(X.loc[idx_test]))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{nombre}\n({etiqueta})", fontsize=9)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

</details>

### 3.3 Selección desde la perspectiva médica

Elige el modelo con mayor sensibilidad y, en caso de empate, mayor especificidad.

```python
mejor = tabla.sort_values([___, ___], ascending=False).iloc[0]
print("Modelo seleccionado:", mejor["modelo"], "-", mejor["variables"])
modelo_final, X_final = entrenados[(mejor["modelo"], mejor["variables"])]
```

<details><summary><b>Ver solución</b></summary>

```python
mejor = tabla.sort_values(["sensibilidad", "especificidad"], ascending=False).iloc[0]
print("Modelo seleccionado:", mejor["modelo"], "-", mejor["variables"])
modelo_final, X_final = entrenados[(mejor["modelo"], mejor["variables"])]
```

</details>

**Verifica tu resultado:** Árbol de decisión sin texto.

#### Hints para interpretar tus resultados

* **Hint de accuracy:** La logística sin texto tiene 70% de accuracy, pero detecta menos de la mitad de los enfermos (48%). *¿Es suficiente el accuracy en contextos médicos? ¿Qué número le reportarías a la dirección médica?*
* **Hint de evaluación clínica:** El modelo elegido tiene 70.4% de sensibilidad: detecta a ~7 de cada 10 enfermos. *Analiza el impacto clínico del 29.6% restante (los falsos negativos). En una clínica real, ¿qué medidas de mitigación o seguimiento propondrías para ellos?*
* **Hint de texto:** Agregar las notas clínicas mejoró a la logística (de 48% a 65% de sensibilidad), pero empeoró al árbol. *¿Más variables siempre significan un mejor modelo? ¿Qué te dice esto sobre probar varias combinaciones antes de decidir?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 4. Análisis estratégico

Aquí el modelo deja de ser un ejercicio técnico y se convierte en decisiones de operación: a quién atender y cómo (telemedicina), a quién incluir en un programa (medicamentos) y cuánto comprar (cadena de suministro).

#### Instrucciones de Python que usarás

* **`cross_val_predict(modelo, X, y, cv=5, method="predict_proba")`**
    * *Tipo:* Información.
    * *Explicación:* Calcula la probabilidad de riesgo de cada paciente usando validación cruzada: cada paciente recibe la predicción de un modelo que **no** lo vio al entrenar. `[:, 1]` toma la probabilidad de la clase 1 (enfermo).
* **`pd.cut(serie, bins=umbrales, labels=acciones)`**
    * *Tipo:* Transformación.
    * *Explicación:* Convierte una variable continua (la probabilidad de riesgo) en categorías (acciones de triaje). Traduce la salida analítica a decisiones médicas.
* **`serie.value_counts()`**
    * *Tipo:* Información.
    * *Explicación:* Cuenta cuántos registros hay en cada categoría.
* **`df[(cond1) & (cond2)]`**
    * *Tipo:* Transformación.
    * *Explicación:* Filtra los pacientes que cumplen ambas condiciones a la vez. Cada condición va entre paréntesis.
* **`(beneficio - inversion) / inversion`**
    * *Tipo:* Cálculo.
    * *Explicación:* **ROI:** cuánto se gana por cada peso invertido.
* **`serie.rolling(3).mean()`**
    * *Tipo:* Transformación.
    * *Explicación:* Promedio móvil de 3 meses: suaviza la variación mensual para estimar la demanda típica.
* **`z * desviacion * np.sqrt(tiempo_entrega)`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Inventario de seguridad:** colchón de producto para cubrir la variabilidad de la demanda mientras llega el pedido. `z = 1.65` equivale a un nivel de servicio del 95%.

### 4.1 Probabilidad de riesgo para toda la población

Calcula la probabilidad de riesgo de cada paciente con el modelo seleccionado.

```python
df["prob_riesgo"] = cross_val_predict(___, ___, y, cv=___, method=___)[:, ___]
df["prob_riesgo"].describe().round(3)
```

<details><summary><b>Pista</b></summary>

Usa el modelo y el conjunto de variables que seleccionaste en 3.3, con 5 particiones. El método que devuelve probabilidades es `"predict_proba"`, que entrega dos columnas: la 0 (sano) y la 1 (enfermo). Necesitas la de enfermo.

</details>

<details><summary><b>Ver solución</b></summary>

```python
df["prob_riesgo"] = cross_val_predict(modelo_final, X_final, y, cv=5, method="predict_proba")[:, 1]
df["prob_riesgo"].describe().round(3)
```

</details>

**Verifica tu resultado:** valores entre 0 y 1; el promedio de la probabilidad debe quedar cerca de la prevalencia (~0.35).

### 4.2 Telemedicina: reglas de triaje

Clasifica a los pacientes en tres acciones según su probabilidad de riesgo.

```python
umbrales = [0, 0.2, 0.5, 1.01]
acciones = ["Recordatorio en app", "Teleconsulta de seguimiento", "Cita presencial + laboratorio"]
df["triaje"] = pd.___(df["prob_riesgo"], bins=umbrales, labels=acciones, right=False)
df["triaje"].___()
```

<details><summary><b>Ver solución</b></summary>

```python
umbrales = [0, 0.2, 0.5, 1.01]
acciones = ["Recordatorio en app", "Teleconsulta de seguimiento", "Cita presencial + laboratorio"]
df["triaje"] = pd.cut(df["prob_riesgo"], bins=umbrales, labels=acciones, right=False)
df["triaje"].value_counts()
```

</details>

**Verifica tu resultado:** 369 recordatorios en app, 99 teleconsultas y 300 citas presenciales.

### 4.3 Programa de adherencia a medicamentos

Identifica a los pacientes diagnosticados con adherencia menor a 60% y calcula el ROI del programa.

```python
objetivo = df[(df["diagnostico"] == ___) & (df["adherencia"] < ___)]
costo_por_paciente = 1500        # supuesto, MXN anuales
beneficio_por_paciente = 4000    # supuesto, MXN anuales en complicaciones evitadas
tasa_exito = 0.40                # supuesto
inversion = len(objetivo) * costo_por_paciente
beneficio = len(objetivo) * tasa_exito * beneficio_por_paciente
roi = (___ - ___) / ___ * 100
print("Pacientes objetivo:", len(objetivo), "| ROI:", round(roi, 1), "%")
```

<details><summary><b>Ver solución</b></summary>

```python
objetivo = df[(df["diagnostico"] == 1) & (df["adherencia"] < 0.6)]
costo_por_paciente = 1500        # supuesto, MXN anuales
beneficio_por_paciente = 4000    # supuesto, MXN anuales en complicaciones evitadas
tasa_exito = 0.40                # supuesto
inversion = len(objetivo) * costo_por_paciente
beneficio = len(objetivo) * tasa_exito * beneficio_por_paciente
roi = (beneficio - inversion) / inversion * 100
print("Pacientes objetivo:", len(objetivo), "| ROI:", round(roi, 1), "%")
```

</details>

**Verifica tu resultado:** 94 pacientes objetivo y ROI de 6.7%.

### 4.4 Cadena de suministro: punto de reorden

Proyecta la demanda mensual de tiras reactivas y calcula el punto de reorden.

```python
consumo["promedio_movil_3m"] = consumo["tiras_reactivas"].___(3).mean()
tiras_por_paciente_mes = 60                                           # supuesto
nuevos_alto_riesgo = (df["triaje"] == acciones[-1]).sum() * 0.10      # supuesto: 10% inicia monitoreo
demanda_mensual = consumo["promedio_movil_3m"].iloc[-1] + nuevos_alto_riesgo * tiras_por_paciente_mes
tiempo_entrega_meses = 0.5                                            # supuesto
z = 1.65                                                              # nivel de servicio 95%
inventario_seguridad = z * consumo["tiras_reactivas"].___() * np.sqrt(tiempo_entrega_meses)
punto_reorden = demanda_mensual * tiempo_entrega_meses + inventario_seguridad
print("Demanda mensual:", round(demanda_mensual), "| Punto de reorden:", round(punto_reorden))
```

<details><summary><b>Pista</b></summary>

La desviación estándar en pandas es `.std()`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
consumo["promedio_movil_3m"] = consumo["tiras_reactivas"].rolling(3).mean()
tiras_por_paciente_mes = 60                                           # supuesto
nuevos_alto_riesgo = (df["triaje"] == acciones[-1]).sum() * 0.10      # supuesto: 10% inicia monitoreo
demanda_mensual = consumo["promedio_movil_3m"].iloc[-1] + nuevos_alto_riesgo * tiras_por_paciente_mes
tiempo_entrega_meses = 0.5                                            # supuesto
z = 1.65                                                              # nivel de servicio 95%
inventario_seguridad = z * consumo["tiras_reactivas"].std() * np.sqrt(tiempo_entrega_meses)
punto_reorden = demanda_mensual * tiempo_entrega_meses + inventario_seguridad
print("Demanda mensual:", round(demanda_mensual), "| Punto de reorden:", round(punto_reorden))
```

</details>

**Verifica tu resultado:** demanda mensual de ~7,442 tiras y punto de reorden de ~4,308.

#### Hints para interpretar tus resultados

* **Hint de validación:** *¿Por qué usar `cross_val_predict` y no simplemente `modelo_final.predict_proba()` sobre todos los pacientes? Piensa en qué pasa cuando un modelo evalúa pacientes que ya vio al entrenar.*
* **Hint de triaje:** 300 pacientes quedarían en cita presencial. *Si la clínica solo puede atender 150 citas al mes, ¿moverías el umbral de 0.5 hacia arriba o hacia abajo? ¿Qué arriesgas con cada decisión?*
* **Hint estratégico (ROI):** Obtuviste un ROI positivo de 6.7%. *Cambia `tasa_exito` a 0.10 y vuelve a ejecutar. ¿Qué pasa con el ROI? La analítica en salud busca el equilibrio entre el costo de la intervención y el ahorro por prevención a largo plazo.*
* **Hint de supuestos:** *Observa que el ROI no cambia si hay 94 o 500 pacientes objetivo. ¿Por qué? ¿Qué supuesto del modelo económico lo provoca y qué tan realista es?*
* **Hint de abastecimiento:** *Si las tiras se cambiaran por insulina, que requiere cadena de frío, ¿qué otros datos necesitarías además de la demanda?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 5. Ética y gobernanza

#### Instrucciones de Python que usarás

* **`df.loc[idx_test][condicion]`**
    * *Tipo:* Información.
    * *Explicación:* Selecciona del conjunto de prueba solo los pacientes que cumplen una condición (por ejemplo, enfermos que el modelo no detectó).
* **`np.where(cond, "A", "B")`**
    * *Tipo:* Transformación.
    * *Explicación:* Asigna `"A"` donde la condición se cumple y `"B"` donde no. Aquí sirve para crear grupos de edad.
* **`confusion_matrix(..., labels=[0, 1])`**
    * *Tipo:* Información.
    * *Explicación:* Igual que antes, pero `labels` asegura las cuatro celdas aunque un subgrupo no tenga algún tipo de caso.
* **`hashlib.sha256((sal + id).encode()).hexdigest()`**
    * *Tipo:* Transformación.
    * *Explicación:* Convierte el identificador en un código irreversible (hash). Con una **sal** secreta nadie puede reconstruir el ID original. Es una técnica de **seudonimización**.

### 5.1 Riesgo de falsos negativos

Identifica a los pacientes enfermos que el modelo no detectó en el conjunto de prueba.

```python
pred_test = modelo_final.predict(X_final.loc[idx_test])
fn_ids = df.loc[idx_test][(y.loc[idx_test] == ___) & (pred_test == ___)]["paciente_id"]
print("Pacientes enfermos no detectados:", len(fn_ids))
```

<details><summary><b>Ver solución</b></summary>

```python
pred_test = modelo_final.predict(X_final.loc[idx_test])
fn_ids = df.loc[idx_test][(y.loc[idx_test] == 1) & (pred_test == 0)]["paciente_id"]
print("Pacientes enfermos no detectados:", len(fn_ids))
```

</details>

**Verifica tu resultado:** 16 pacientes.

### 5.2 Sesgo: desempeño por grupo de edad

Compara la sensibilidad del modelo en menores y mayores de 40 años.

```python
grupo = np.where(df.loc[idx_test, "edad"] < ___, "<40", ">=40")
for g in ["<40", ">=40"]:
    mask = grupo == g
    tn, fp, fn, tp = confusion_matrix(y.loc[idx_test][mask], pred_test[mask], labels=[0, 1]).ravel()
    print(f"Edad {g}: sensibilidad = {tp / (tp + fn):.2f} (n = {mask.sum()})")
```

<details><summary><b>Ver solución</b></summary>

```python
grupo = np.where(df.loc[idx_test, "edad"] < 40, "<40", ">=40")
for g in ["<40", ">=40"]:
    mask = grupo == g
    tn, fp, fn, tp = confusion_matrix(y.loc[idx_test][mask], pred_test[mask], labels=[0, 1]).ravel()
    print(f"Edad {g}: sensibilidad = {tp / (tp + fn):.2f} (n = {mask.sum()})")
```

</details>

**Verifica tu resultado:** sensibilidad de 0.61 en menores de 40 (n = 115) y de 0.89 en mayores de 40 (n = 39).

### 5.3 Protección de datos clínicos

Seudonimiza el identificador de cada paciente y guarda un archivo sin las notas clínicas.

```python
df["paciente_id"] = df["paciente_id"].astype(str).apply(
    lambda x: hashlib.___(("sal_clinica" + x).encode()).hexdigest()[:12])
df.drop(columns=[___]).to_csv("datos_clinicos_seudonimizados.csv", index=False)
df[["paciente_id", "edad", "diagnostico"]].head()
```

<details><summary><b>Ver solución</b></summary>

```python
df["paciente_id"] = df["paciente_id"].astype(str).apply(
    lambda x: hashlib.sha256(("sal_clinica" + x).encode()).hexdigest()[:12])
df.drop(columns=["nota_clinica"]).to_csv("datos_clinicos_seudonimizados.csv", index=False)
df[["paciente_id", "edad", "diagnostico"]].head()
```

</details>

**Verifica tu resultado:** el `paciente_id` aparece como un código de 12 caracteres (por ejemplo, `a3f9…`) en lugar de un número consecutivo.

#### Hints para interpretar tus resultados

* **Hint de ética algorítmica:** El modelo detecta peor a los menores de 40 años (61% contra 89%). *Concluye sobre el peligro de usar la IA como un "sustituto ciego" del médico en lugar de un "apoyo a la decisión". ¿Debería el triaje automático aplicar los mismos umbrales para todas las edades?*
* **Hint de tamaño de muestra:** El grupo de mayores de 40 solo tiene 39 pacientes de prueba. *¿Qué tan seguro puedes estar de ese 89%? ¿Cambiaría tu conclusión con más datos?*
* **Hint de población:** Todas las pacientes del dataset son mujeres de 21 años o más de herencia indígena pima. *¿Usarías este modelo con hombres o con población mexicana sin volver a validarlo?*
* **Hint de privacidad:** Se eliminó la columna de notas antes de guardar. *¿Por qué el texto libre es un riesgo para la privacidad aunque el ID ya esté seudonimizado? Piensa en lo que un médico puede escribir en una nota.*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Cierre de la práctica

Antes de redactar tu reporte técnico, responde:

* ¿Es suficiente el accuracy en contextos médicos?
* ¿Cuál es el impacto de un falso negativo en salud?
* ¿Qué aporta la información no estructurada (notas clínicas) al modelo?
* ¿Cómo mejoraría la telemedicina con modelos predictivos?
* ¿Un modelo entrenado con una población específica puede aplicarse a cualquier paciente?

**Lista de evidencias:** código fuente (este notebook), gráficas exploratorias, matrices de confusión, tabla comparativa de modelos, resultados de triaje, ROI y punto de reorden, y reporte técnico de 2–3 cuartillas (resultados, interpretación clínica, propuesta estratégica y análisis ético).
