# Práctica 6-B (complementaria)
## Detección de Fraude en Transacciones de Dinero Móvil con PaySim

#### ==========================================================
### Práctica 6-B — Detección de Fraude en Transacciones
### de Dinero Móvil (complementaria de la Práctica 6)
### Autopráctica guiada

#### Datos: PaySim, "Synthetic Financial Datasets For Fraud Detection"
####   https://www.kaggle.com/datasets/ealaxi/paysim1
####   Simulador construido a partir de un mes de transacciones reales
####   de un servicio de dinero móvil en un país africano.
#### Muestra: "paysim_50k.csv" (proporcionada por el docente):
####   50,000 transacciones, 1,000 fraudes (2 %), semilla 42.
#### ==========================================================

**Caso:** una empresa de servicios financieros móviles quiere un sistema que detecte transacciones fraudulentas antes de autorizarlas. Hoy solo cuenta con una regla de negocio: marcar cualquier transferencia mayor a 200,000. La dirección de seguridad quiere saber cuánto fraude se le está escapando, si un modelo de ciencia de datos lo haría mejor y cuánto dinero protegería.

**¿En qué se diferencia de la Práctica 6?** Allí el fraude era el 68 % de las transacciones y las variables casi no lo explicaban. Aquí el fraude es poco frecuente, tiene un patrón de comportamiento real y existe una regla de negocio contra la cual comparar. La Parte 5 sustituye el riesgo crediticio (PaySim no tiene historial de crédito) por el **impacto económico del fraude**.

**Cómo usar este notebook**

1. Lee las **instrucciones de Python** de cada parte: qué hace cada función y si transforma los datos o solo te da información.
2. Completa las celdas de código donde aparece `___` (guion bajo triple).
3. Si te atoras, abre la **Pista**. Si aún así no sale, compara con **Ver solución**.
4. Revisa la sección **Verifica tu resultado** para comprobar que vas bien.
5. Lee los **Hints** y escribe tu interpretación en la celda **Tu interpretación**. Esa es la parte más importante: el código solo es el medio.

> **Requisito previo:** colocar `paysim_50k.csv` en la misma carpeta que este notebook.

### Diccionario de datos

| Columna | Descripción |
|---|---|
| `step` | Hora de la simulación (1 a 744, es decir, 30 días). |
| `type` | Tipo de transacción: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER. |
| `amount` | Monto de la transacción. |
| `nameOrig` / `nameDest` | Cuenta de origen y de destino (las que empiezan con M son comercios). |
| `oldbalanceOrg` / `newbalanceOrig` | Saldo de la cuenta de origen antes y después de la transacción. |
| `oldbalanceDest` / `newbalanceDest` | Saldo de la cuenta de destino antes y después de la transacción. |
| `isFraud` | 1 = fraude, 0 = normal (variable objetivo). |
| `isFlaggedFraud` | 1 = la regla del negocio la marcó (transferencia mayor a 200,000). |

---
## Parte 0. Preparación del entorno

#### Instrucciones de Python que usarás

* **`import pandas as pd`**
    * *Tipo:* Configuración.
    * *Explicación:* Carga la librería para manejar tablas (DataFrames). Por convención se abrevia `pd`.
* **`np.random.seed(42)`**
    * *Tipo:* Configuración.
    * *Explicación:* Fija la semilla aleatoria para que todos obtengan los mismos resultados.
* **`pd.read_csv("archivo.csv")`**
    * *Tipo:* Información.
    * *Explicación:* Lee un archivo CSV y lo convierte en un DataFrame.
* **`df.shape / df.head()`**
    * *Tipo:* Información.
    * *Explicación:* `shape` devuelve (filas, columnas); `head()` muestra los primeros 5 registros.

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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

np.random.seed(42)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

df = pd.read_csv("paysim_50k.csv")
print(df.shape)
df.head()
```

**Verifica tu resultado:** `(50000, 11)`: 50,000 transacciones y 11 columnas.

---
## Parte 1. Análisis exploratorio de transacciones

#### Instrucciones de Python que usarás

* **`df.dtypes / df.isna().sum()`**
    * *Tipo:* Información.
    * *Explicación:* Tipo de dato de cada columna y número de nulos por columna.
* **`df["col"].describe()`**
    * *Tipo:* Información.
    * *Explicación:* Resumen estadístico de una variable: conteo, media, desviación, mínimo, cuartiles y máximo.
* **`serie.median() / serie.skew()`**
    * *Tipo:* Información.
    * *Explicación:* Valor central y asimetría. Una asimetría muy positiva indica pocos valores enormes que jalan la media hacia arriba.
* **`pd.crosstab(fila, columna)`**
    * *Tipo:* Información.
    * *Explicación:* Tabla de conteo cruzado entre dos variables categóricas.
* **`sns.histplot(..., log_scale=True)`**
    * *Tipo:* Información (visual).
    * *Explicación:* Histograma con el eje en escala logarítmica. Es útil cuando los valores van de unos cuantos pesos a millones. Con `stat="density", common_norm=False` cada clase se normaliza por separado y se pueden comparar aunque una sea mucho más pequeña.

### 1.1 Revisar estructura y faltantes

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

**Verifica tu resultado:** 0 nulos. `type`, `nameOrig` y `nameDest` son texto; el resto son numéricas.

### 1.2 Estadística descriptiva del monto

```python
print(df["amount"].___())
print("Mediana:", round(df["amount"].___(), 2))
print("Asimetría:", round(df["amount"].___(), 2))
```

<details><summary><b>Ver solución</b></summary>

```python
print(df["amount"].describe())
print("Mediana:", round(df["amount"].median(), 2))
print("Asimetría:", round(df["amount"].skew(), 2))
```

</details>

**Verifica tu resultado:** la media es mucho mayor que la mediana y la asimetría es fuertemente positiva: unas cuantas transacciones de millones conviven con muchas pequeñas.

### 1.3 ¿Dónde ocurre el fraude?

Calcula la tasa de fraude y cruza el tipo de transacción contra la etiqueta de fraude.

```python
print("Tasa de fraude:", round(df["isFraud"].___() * 100, 2), "%")
pd.___(df["type"], df[___])
```

<details><summary><b>Ver solución</b></summary>

```python
print("Tasa de fraude:", round(df["isFraud"].mean() * 100, 2), "%")
pd.crosstab(df["type"], df["isFraud"])
```

</details>

**Verifica tu resultado:** tasa de fraude de 2.0 %. Los 1,000 fraudes están **solo** en TRANSFER y CASH_OUT; CASH_IN, DEBIT y PAYMENT tienen 0 fraudes.

### 1.4 Gráficas de distribución

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.___(x="type", hue="isFraud", data=df, ax=axes[0])
axes[0].set_title("Transacciones por tipo y clase")
sns.histplot(data=df[df["amount"] > 0], x="amount", hue="isFraud", log_scale=___,
             stat="density", common_norm=___, bins=50, ax=axes[1])
axes[1].set_title("Monto (escala logarítmica) por clase")
plt.tight_layout(); plt.show()
```

<details><summary><b>Pista</b></summary>

El primer gráfico cuenta registros por categoría. En el histograma, la escala logarítmica se activa con `True`, y `common_norm` debe ser falso para que cada clase se dibuje con su propia escala.

</details>

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.countplot(x="type", hue="isFraud", data=df, ax=axes[0])
axes[0].set_title("Transacciones por tipo y clase")
sns.histplot(data=df[df["amount"] > 0], x="amount", hue="isFraud", log_scale=True,
             stat="density", common_norm=False, bins=50, ax=axes[1])
axes[1].set_title("Monto (escala logarítmica) por clase")
plt.tight_layout(); plt.show()
```

</details>

**Verifica tu resultado:** en el conteo, la barra de fraude casi no se ve (2 %). En el histograma, la distribución de los fraudes está desplazada hacia montos más altos que la de las transacciones normales.

#### Hints para interpretar tus resultados

* **Hint de muestra:** En el dataset completo de PaySim el fraude es solo el 0.13 %; en esta muestra es el 2 % porque se conservaron menos transacciones normales. *¿Qué ventaja tiene eso para entrenar un modelo? ¿Qué riesgo tiene al reportar resultados a la dirección?*
* **Hint de comportamiento:** El fraude solo aparece en TRANSFER y CASH_OUT. *Piensa en cómo opera un defraudador que toma el control de una cuenta: ¿por qué transferiría el dinero y luego lo retiraría en efectivo?*
* **Hint de línea base:** *Si un modelo dijera "normal" a todas las transacciones, ¿qué accuracy tendría? Compáralo con lo que viste en la Práctica 6.*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 2. Identificación de patrones anormales

Como el fraude solo ocurre en TRANSFER y CASH_OUT, a partir de aquí el análisis se enfoca en esos dos tipos. Esta es una decisión de negocio: no tiene sentido vigilar con un modelo un canal donde no hay fraude.

#### Instrucciones de Python que usarás

* **`df[df["col"].isin([...])].copy()`**
    * *Tipo:* Transformación.
    * *Explicación:* Filtra las filas cuyo valor está en la lista. `copy()` crea una tabla independiente para poder agregarle columnas sin advertencias.
* **`df.groupby("isFraud")["amount"].agg(["mean", "median"])`**
    * *Tipo:* Información.
    * *Explicación:* Calcula varias estadísticas a la vez para cada grupo.
* **`serie % 24`**
    * *Tipo:* Transformación.
    * *Explicación:* Operador módulo: el residuo de dividir entre 24. Convierte el número de hora de la simulación (1 a 744) en la hora del día (0 a 23).
* **`(cond1) & (cond2)`**
    * *Tipo:* Transformación.
    * *Explicación:* Operador "y": verdadero solo si se cumplen ambas condiciones. Cada condición va entre paréntesis.
* **`sns.lineplot(...)`**
    * *Tipo:* Información (visual).
    * *Explicación:* Gráfica de línea; útil para ver cómo cambia una tasa a lo largo de las horas.

### 2.1 Filtrar los canales con fraude y comparar montos

```python
tc = df[df["type"].___(["TRANSFER", "CASH_OUT"])].copy()
print("Transacciones TRANSFER + CASH_OUT:", len(tc))
print("Tasa de fraude en estos canales:", round(tc["isFraud"].mean() * 100, 2), "%")
tc.groupby(___)["amount"].agg(["mean", ___])
```

<details><summary><b>Ver solución</b></summary>

```python
tc = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
print("Transacciones TRANSFER + CASH_OUT:", len(tc))
print("Tasa de fraude en estos canales:", round(tc["isFraud"].mean() * 100, 2), "%")
tc.groupby("isFraud")["amount"].agg(["mean", "median"])
```

</details>

**Verifica tu resultado:** quedan menos de la mitad de las 50,000 transacciones, y la tasa de fraude sube respecto al 2 % general. La mediana del monto de los fraudes es claramente mayor que la de las transacciones normales.

### 2.2 Patrón "vaciar la cuenta"

Marca las transacciones en las que se mueve todo el saldo de la cuenta de origen (o más) y compara su tasa de fraude.

```python
tc["vacia_cuenta"] = ((tc["amount"] >= tc[___]) & (tc["oldbalanceOrg"] > ___)).astype(int)
tc.groupby("vacia_cuenta")["isFraud"].agg(["count", "mean"])
```

<details><summary><b>Pista</b></summary>

Compara el monto contra el saldo **anterior** de la cuenta de origen. La segunda condición excluye las cuentas que ya estaban en cero.

</details>

<details><summary><b>Ver solución</b></summary>

```python
tc["vacia_cuenta"] = ((tc["amount"] >= tc["oldbalanceOrg"]) & (tc["oldbalanceOrg"] > 0)).astype(int)
tc.groupby("vacia_cuenta")["isFraud"].agg(["count", "mean"])
```

</details>

**Verifica tu resultado:** la tasa de fraude en el grupo `vacia_cuenta = 1` es muchas veces mayor que en el grupo 0.

### 2.3 Fraude por hora del día

```python
tc["hora"] = tc["step"] ___ 24
tasa_hora = tc.groupby("hora")["isFraud"].mean()
plt.figure(figsize=(10, 4))
sns.___(x=tasa_hora.index, y=tasa_hora.values, marker="o")
plt.title("Tasa de fraude por hora del día"); plt.xlabel("Hora"); plt.ylabel("Tasa de fraude")
plt.show()
```

<details><summary><b>Ver solución</b></summary>

```python
tc["hora"] = tc["step"] % 24
tasa_hora = tc.groupby("hora")["isFraud"].mean()
plt.figure(figsize=(10, 4))
sns.lineplot(x=tasa_hora.index, y=tasa_hora.values, marker="o")
plt.title("Tasa de fraude por hora del día"); plt.xlabel("Hora"); plt.ylabel("Tasa de fraude")
plt.show()
```

</details>

**Verifica tu resultado:** la tasa de fraude no es pareja a lo largo del día; observa en qué franja horaria se dispara.

### 2.4 ¿Qué tan buena es la regla actual del negocio?

Compara la regla `isFlaggedFraud` contra el fraude real en toda la muestra.

```python
regla = pd.crosstab(df[___], df["isFraud"])
print(regla)
detectados = regla.loc[1, 1]
print("La regla detecta", detectados, "de", df["isFraud"].___(), "fraudes:",
      round(detectados / df["isFraud"].sum() * 100, 1), "%")
```

<details><summary><b>Ver solución</b></summary>

```python
regla = pd.crosstab(df["isFlaggedFraud"], df["isFraud"])
print(regla)
detectados = regla.loc[1, 1]
print("La regla detecta", detectados, "de", df["isFraud"].sum(), "fraudes:",
      round(detectados / df["isFraud"].sum() * 100, 1), "%")
```

</details>

**Verifica tu resultado:** la regla marca 16 transacciones y las 16 son fraude (no se equivoca nunca), pero solo detecta 16 de 1,000 fraudes: 1.6 %.

#### Hints para interpretar tus resultados

* **Hint de la regla:** La regla tiene 100 % de precisión y 1.6 % de sensibilidad. *¿Es una buena regla? ¿Qué le dirías a la dirección que hoy confía en ella?*
* **Hint de comportamiento:** *Relaciona el patrón "vaciar la cuenta" con la forma de operar del defraudador. ¿Qué haría un defraudador más astuto para no ser detectado por esa señal?*
* **Hint de hora:** *¿Por qué el fraude pesa más en ciertas horas? Piensa en cuándo hacen transacciones los clientes legítimos y cuándo nadie está vigilando su cuenta.*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 3. Desarrollo del modelo de detección de fraude

#### Instrucciones de Python que usarás

* **`pd.get_dummies(df[cols], dtype=int)`**
    * *Tipo:* Transformación.
    * *Explicación:* Convierte las categorías de texto (aquí, `type`) en columnas 0/1.
* **`train_test_split(indices, test_size=0.2, stratify=y)`**
    * *Tipo:* Transformación.
    * *Explicación:* Separa los índices en entrenamiento (80 %) y prueba (20 %) con la misma tasa de fraude en ambos. Al dividir índices, puedes usar la misma partición con distintos conjuntos de variables.
* **`class_weight="balanced"`**
    * *Tipo:* Modelado.
    * *Explicación:* Da más peso a los errores en la clase minoritaria (aquí, el fraude). Sin esto, el modelo tiende a decir "normal" a todo.
* **`RandomForestClassifier(n_jobs=-1)`**
    * *Tipo:* Modelado.
    * *Explicación:* `n_jobs=-1` usa todos los núcleos del procesador para entrenar más rápido.
* **`modelo.fit(X, y)`**
    * *Tipo:* Modelado.
    * *Explicación:* Entrena el modelo.

> **Fuga de datos (data leakage).** Los autores de PaySim advierten que las transacciones fraudulentas se anulan, así que las columnas de saldos (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) **no deben usarse** para detectar fraude. Además, los saldos "nuevos" se conocen hasta *después* de la transacción, cuando ya es tarde para bloquearla. En esta parte vas a entrenar un modelo **sin saldos** (el correcto) y uno **con saldos** para ver qué pasa.

### 3.1 Preparar las variables

Crea una bandera de madrugada y arma los dos conjuntos de variables.

```python
tc["madrugada"] = (tc["hora"] < ___).astype(int)
saldos = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

X_sin = pd.get_dummies(tc[["amount", "hora", "madrugada", "type"]], dtype=int)
X_con = pd.get_dummies(tc[["amount", "hora", "madrugada", "type"] + ___], dtype=int)
y = tc[___]
print(X_sin.shape, X_con.shape)
list(X_sin.columns)
```

<details><summary><b>Pista</b></summary>

Considera madrugada de las 0 a las 5 horas (es decir, hora menor a 6). El conjunto "con saldos" agrega la lista `saldos` a las variables base.

</details>

<details><summary><b>Ver solución</b></summary>

```python
tc["madrugada"] = (tc["hora"] < 6).astype(int)
saldos = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

X_sin = pd.get_dummies(tc[["amount", "hora", "madrugada", "type"]], dtype=int)
X_con = pd.get_dummies(tc[["amount", "hora", "madrugada", "type"] + saldos], dtype=int)
y = tc["isFraud"]
print(X_sin.shape, X_con.shape)
list(X_sin.columns)
```

</details>

**Verifica tu resultado:** `X_sin` tiene 5 columnas (`amount`, `hora`, `madrugada`, `type_CASH_OUT`, `type_TRANSFER`) y `X_con` tiene 9.

### 3.2 División 80/20 estratificada

```python
idx_train, idx_test = train_test_split(tc.index, test_size=___, random_state=42, stratify=___)
print(len(idx_train), len(idx_test))
print("Fraudes en prueba:", y.loc[idx_test].sum())
```

<details><summary><b>Ver solución</b></summary>

```python
idx_train, idx_test = train_test_split(tc.index, test_size=0.2, random_state=42, stratify=y)
print(len(idx_train), len(idx_test))
print("Fraudes en prueba:", y.loc[idx_test].sum())
```

</details>

**Verifica tu resultado:** 200 fraudes en el conjunto de prueba (el 20 % de 1,000).

### 3.3 Entrenar los modelos

Entrena tres modelos balanceados **sin saldos** y un Random Forest **con saldos**.

```python
modelos = {
    "Regresion logistica": (make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight=___)), X_sin),
    "Arbol de decision": (DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                          random_state=42), X_sin),
    "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                      class_weight="balanced", random_state=42, n_jobs=-1), X_sin),
    "Random Forest CON SALDOS": (RandomForestClassifier(n_estimators=200, max_depth=8,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=42, n_jobs=-1), ___),
}
for nombre, (modelo, X) in modelos.items():
    modelo.___(X.loc[idx_train], y.loc[idx_train])
    print(nombre, "entrenado")
```

<details><summary><b>Ver solución</b></summary>

```python
modelos = {
    "Regresion logistica": (make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced")), X_sin),
    "Arbol de decision": (DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                          random_state=42), X_sin),
    "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                      class_weight="balanced", random_state=42, n_jobs=-1), X_sin),
    "Random Forest CON SALDOS": (RandomForestClassifier(n_estimators=200, max_depth=8,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=42, n_jobs=-1), X_con),
}
for nombre, (modelo, X) in modelos.items():
    modelo.fit(X.loc[idx_train], y.loc[idx_train])
    print(nombre, "entrenado")
```

</details>

**Verifica tu resultado:** se imprimen los 4 modelos como entrenados.

#### Hints para interpretar tus resultados

* **Hint de variables:** *¿Por qué se usa la hora y además una bandera de madrugada? Piensa en cómo "ve" la hora una regresión logística (como una línea recta de 0 a 23) y cómo la ve un árbol (puede hacer cortes donde quiera).*
* **Hint de fuga:** *Antes de ver resultados, predice: ¿el modelo con saldos será mejor o peor? ¿Eso lo haría un modelo más útil?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 4. Evaluación del modelo

#### Instrucciones de Python que usarás

* **`confusion_matrix(y_real, y_pred).ravel()`**
    * *Tipo:* Información.
    * *Explicación:* Devuelve **tn, fp, fn, tp** en ese orden.
* **`tp / (tp + fn)` · `tn / (tn + fp)` · `tp / (tp + fp)`**
    * *Tipo:* Cálculo.
    * *Explicación:* **Sensibilidad** (fraudes detectados de todos los fraudes), **especificidad** (normales reconocidas de todas las normales) y **precisión** (de las alertas, cuántas eran fraude de verdad).
* **`roc_auc_score(y_real, probabilidades)`**
    * *Tipo:* Información.
    * *Explicación:* Mide qué tan bien ordena el modelo las transacciones por riesgo, sin depender de un umbral. 0.5 es azar; 1.0 es perfecto.
* **`modelo.predict_proba(X)[:, 1]`**
    * *Tipo:* Información.
    * *Explicación:* Probabilidad de fraude de cada transacción.
* **`modelo[-1].coef_ / modelo.feature_importances_`**
    * *Tipo:* Información.
    * *Explicación:* Coeficientes de la logística (último paso del pipeline) e importancia de variables en árboles.

### 4.1 Métricas de los cuatro modelos

```python
resultados = []
for nombre, (m, X) in modelos.items():
    pred = m.predict(X.loc[idx_test])
    prob = m.predict_proba(X.loc[idx_test])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y.loc[idx_test], pred).ravel()
    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy_score(y.loc[idx_test], pred),
        "sensibilidad": ___ / (___ + ___),
        "especificidad": ___ / (___ + ___),
        "precision": ___ / (___ + ___),
        "auc": roc_auc_score(y.loc[idx_test], ___),
        "FN": fn, "FP": fp,
    })
tabla = pd.DataFrame(resultados).set_index("modelo").round(3)
print("Línea base (decir siempre 'normal'):", round(1 - y.loc[idx_test].mean(), 3))
tabla
```

<details><summary><b>Pista</b></summary>

La sensibilidad mira solo a los fraudes reales; la especificidad, solo a las normales reales; la precisión, solo a las transacciones que el modelo marcó como fraude. El AUC se calcula con las probabilidades, no con las predicciones 0/1.

</details>

<details><summary><b>Ver solución</b></summary>

```python
resultados = []
for nombre, (m, X) in modelos.items():
    pred = m.predict(X.loc[idx_test])
    prob = m.predict_proba(X.loc[idx_test])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y.loc[idx_test], pred).ravel()
    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy_score(y.loc[idx_test], pred),
        "sensibilidad": tp / (tp + fn),
        "especificidad": tn / (tn + fp),
        "precision": tp / (tp + fp),
        "auc": roc_auc_score(y.loc[idx_test], prob),
        "FN": fn, "FP": fp,
    })
tabla = pd.DataFrame(resultados).set_index("modelo").round(3)
print("Línea base (decir siempre 'normal'):", round(1 - y.loc[idx_test].mean(), 3))
tabla
```

</details>

**Verifica tu resultado:** los tres modelos sin saldos tienen AUC muy por encima de 0.5, y el Random Forest CON SALDOS obtiene métricas casi perfectas.

### 4.2 Matrices de confusión

```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (nombre, (m, X)) in zip(axes, modelos.items()):
    cm = ___(y.loc[idx_test], m.predict(X.loc[idx_test]))
    sns.heatmap(cm, annot=___, fmt=___, cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
    ax.set_title(nombre, fontsize=9); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (nombre, (m, X)) in zip(axes, modelos.items()):
    cm = confusion_matrix(y.loc[idx_test], m.predict(X.loc[idx_test]))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"])
    ax.set_title(nombre, fontsize=9); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

</details>

**Verifica tu resultado:** en cada matriz, la fila "Fraude" suma 200. La celda de abajo a la izquierda son los fraudes que se escaparon.

### 4.3 ¿De dónde sale la perfección? Importancia con saldos

```python
rf_con = modelos["Random Forest CON SALDOS"][0]
pd.Series(rf_con.___, index=X_con.columns).sort_values(ascending=___).round(3)
```

<details><summary><b>Ver solución</b></summary>

```python
rf_con = modelos["Random Forest CON SALDOS"][0]
pd.Series(rf_con.feature_importances_, index=X_con.columns).sort_values(ascending=False).round(3)
```

</details>

**Verifica tu resultado:** las variables de saldo ocupan los primeros lugares, por encima del monto.

### 4.4 Selección del modelo y ajuste del umbral

Elige, **entre los modelos sin saldos**, el de mayor AUC y analiza distintos umbrales.

```python
candidatos = tabla.drop(index=___)
mejor = candidatos["auc"].___()
modelo_final = modelos[mejor][0]
print("Modelo seleccionado:", mejor)

prob_final = modelo_final.predict_proba(X_sin.loc[idx_test])[:, 1]
y_test = y.loc[idx_test]
filas = []
for umbral in [0.3, 0.5, 0.7, 0.9]:
    pred_u = (prob_final >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_u).ravel()
    filas.append({"umbral": umbral, "detectados": tp, "escapados": fn, "falsas_alarmas": fp,
                  "sensibilidad": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0})
pd.DataFrame(filas).round(3)
```

<details><summary><b>Pista</b></summary>

`drop(index=...)` quita la fila del modelo con fuga. `idxmax()` devuelve el nombre (índice) de la fila con el valor más alto.

</details>

<details><summary><b>Ver solución</b></summary>

```python
candidatos = tabla.drop(index="Random Forest CON SALDOS")
mejor = candidatos["auc"].idxmax()
modelo_final = modelos[mejor][0]
print("Modelo seleccionado:", mejor)

prob_final = modelo_final.predict_proba(X_sin.loc[idx_test])[:, 1]
y_test = y.loc[idx_test]
filas = []
for umbral in [0.3, 0.5, 0.7, 0.9]:
    pred_u = (prob_final >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_u).ravel()
    filas.append({"umbral": umbral, "detectados": tp, "escapados": fn, "falsas_alarmas": fp,
                  "sensibilidad": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0})
pd.DataFrame(filas).round(3)
```

</details>

**Verifica tu resultado:** al subir el umbral bajan las falsas alarmas y también los fraudes detectados.

### 4.5 ¿Cuánto valen las alertas en producción?

La muestra tiene 2 % de fraude, pero en la operación real es 0.13 %. La precisión depende de la prevalencia, así que hay que corregirla.

```python
p_real = ___                     # prevalencia real de PaySim
pred_05 = (prob_final >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_05).ravel()
sens, espec = tp / (tp + fn), tn / (tn + fp)
prec_muestra = tp / (tp + fp)
prec_real = sens * p_real / (sens * p_real + (1 - ___) * (1 - p_real))
print(f"Precisión en la muestra: {prec_muestra:.3f}")
print(f"Precisión esperada en producción: {prec_real:.3f}")
print(f"De cada 100 alertas en producción, ~{prec_real * 100:.1f} serían fraude real")
```

<details><summary><b>Pista</b></summary>

0.13 % escrito como proporción es 0.0013. En el denominador, la tasa de falsas alarmas es 1 − especificidad.

</details>

<details><summary><b>Ver solución</b></summary>

```python
p_real = 0.0013                  # prevalencia real de PaySim
pred_05 = (prob_final >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_05).ravel()
sens, espec = tp / (tp + fn), tn / (tn + fp)
prec_muestra = tp / (tp + fp)
prec_real = sens * p_real / (sens * p_real + (1 - espec) * (1 - p_real))
print(f"Precisión en la muestra: {prec_muestra:.3f}")
print(f"Precisión esperada en producción: {prec_real:.3f}")
print(f"De cada 100 alertas en producción, ~{prec_real * 100:.1f} serían fraude real")
```

</details>

**Verifica tu resultado:** la precisión esperada en producción es mucho menor que la de la muestra.

#### Hints para interpretar tus resultados

* **Hint de accuracy:** La línea base ya tiene un accuracy altísimo sin detectar un solo fraude. *¿Qué métrica le reportarías a la dirección de seguridad y por qué?*
* **Hint de fuga de datos:** El modelo con saldos parece casi perfecto. *Si lo pusieras en producción, ¿tendrías los saldos "nuevos" en el momento de autorizar la transacción? ¿Qué pasa con un modelo que en el laboratorio es perfecto y en la operación no puede usarse?*
* **Hint de prevalencia:** *Con la precisión corregida, ¿cuántas alertas falsas tendría que revisar un analista por cada fraude real? ¿Cómo cambia eso el diseño del proceso de revisión?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 5. Impacto económico del fraude

Contar fraudes no es lo mismo que contar dinero. Un modelo que detecta pocos fraudes pero grandes puede proteger más que uno que detecta muchos pequeños.

#### Instrucciones de Python que usarás

* **`df.loc[condicion, "amount"].sum()`**
    * *Tipo:* Información.
    * *Explicación:* Suma el monto solo de las filas que cumplen la condición.
* **`costo_unitario * numero_de_alertas`**
    * *Tipo:* Cálculo.
    * *Explicación:* Costo operativo de revisar las alertas (tiempo de analistas, llamadas al cliente).
* **`(beneficio - costo) / costo`**
    * *Tipo:* Cálculo.
    * *Explicación:* **ROI**: cuánto se gana por cada peso invertido.

### 5.1 Monto protegido contra monto en riesgo

```python
prueba = tc.loc[idx_test, ["type", "amount", "isFraud", "isFlaggedFraud"]].copy()
prueba["prob"] = prob_final
prueba["alerta"] = (prueba["prob"] >= 0.5).astype(int)

monto_fraude = prueba.loc[prueba["isFraud"] == 1, "amount"].___()
monto_detectado = prueba.loc[(prueba["isFraud"] == 1) & (prueba[___] == 1), "amount"].sum()
print(f"Monto total de fraude en prueba: ${monto_fraude:,.0f}")
print(f"Monto detectado por el modelo:   ${monto_detectado:,.0f}")
print(f"% del dinero protegido: {monto_detectado / monto_fraude:.1%}")
print(f"% de fraudes detectados (conteo): {prueba.loc[prueba.isFraud == 1, 'alerta'].mean():.1%}")
```

<details><summary><b>Ver solución</b></summary>

```python
prueba = tc.loc[idx_test, ["type", "amount", "isFraud", "isFlaggedFraud"]].copy()
prueba["prob"] = prob_final
prueba["alerta"] = (prueba["prob"] >= 0.5).astype(int)

monto_fraude = prueba.loc[prueba["isFraud"] == 1, "amount"].sum()
monto_detectado = prueba.loc[(prueba["isFraud"] == 1) & (prueba["alerta"] == 1), "amount"].sum()
print(f"Monto total de fraude en prueba: ${monto_fraude:,.0f}")
print(f"Monto detectado por el modelo:   ${monto_detectado:,.0f}")
print(f"% del dinero protegido: {monto_detectado / monto_fraude:.1%}")
print(f"% de fraudes detectados (conteo): {prueba.loc[prueba.isFraud == 1, 'alerta'].mean():.1%}")
```

</details>

**Verifica tu resultado:** el porcentaje de dinero protegido es mayor que el porcentaje de fraudes detectados por conteo.

### 5.2 ROI del sistema de alertas

```python
costo_revision = 150             # supuesto: costo por alerta revisada
recuperacion = 0.80              # supuesto: % del monto que se logra frenar al detectar
costo = prueba["alerta"].sum() * ___
beneficio = monto_detectado * ___
roi = (___ - ___) / ___ * 100
print(f"Alertas: {prueba['alerta'].sum()} | Costo: ${costo:,.0f} | Beneficio: ${beneficio:,.0f} | ROI: {roi:,.0f} %")
```

<details><summary><b>Ver solución</b></summary>

```python
costo_revision = 150             # supuesto: costo por alerta revisada
recuperacion = 0.80              # supuesto: % del monto que se logra frenar al detectar
costo = prueba["alerta"].sum() * costo_revision
beneficio = monto_detectado * recuperacion
roi = (beneficio - costo) / costo * 100
print(f"Alertas: {prueba['alerta'].sum()} | Costo: ${costo:,.0f} | Beneficio: ${beneficio:,.0f} | ROI: {roi:,.0f} %")
```

</details>

**Verifica tu resultado:** ROI positivo y muy alto, porque los montos de los fraudes son grandes comparados con el costo de revisar una alerta.

#### Hints para interpretar tus resultados

* **Hint de dinero contra conteo:** *Si tuvieras que elegir entre un modelo que detecta 80 % de los fraudes pero solo 50 % del dinero, y otro que detecta 60 % de los fraudes y 90 % del dinero, ¿cuál elegirías? ¿Depende de quién sea la víctima?*
* **Hint de supuestos:** Cambia `costo_revision` a 1,500 y `recuperacion` a 0.3. *¿Sigue siendo rentable? ¿Qué datos reales necesitaría el banco para hacer este cálculo con confianza?*
* **Hint de prevalencia:** *Este ROI se calculó con 2 % de fraude. En producción (0.13 %) habría muchas más alertas falsas por cada fraude. ¿Cómo afectaría eso al costo?*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte. Te servirán para el reporte técnico.)*

---
## Parte 6. Propuesta de vigilancia automatizada

#### Instrucciones de Python que usarás

* **`np.select([cond1, cond2], ["A", "B"], default="C")`**
    * *Tipo:* Transformación.
    * *Explicación:* Asigna `"A"` donde se cumple la primera condición, `"B"` donde se cumple la segunda y `"C"` en el resto.
* **`(cond1) | (cond2)`**
    * *Tipo:* Transformación.
    * *Explicación:* Operador "o": verdadero si se cumple al menos una condición.
* **`df.groupby("col").agg(nueva=("col", "función"))`**
    * *Tipo:* Información.
    * *Explicación:* Resume varias columnas por grupo con nombres nuevos.

### 6.1 Regla actual contra esquema propuesto

Combina la regla del negocio con el modelo:

* **BLOQUEAR:** probabilidad ≥ 0.9 **o** marcada por la regla actual.
* **REVISAR** (autenticación adicional o analista): probabilidad ≥ 0.5.
* **APROBAR:** el resto.

```python
prueba["accion"] = np.___(
    [(prueba["prob"] >= ___) | (prueba["isFlaggedFraud"] == 1), prueba["prob"] >= ___],
    ["BLOQUEAR", "REVISAR"], default=___)

resumen = prueba.groupby("accion").agg(
    transacciones=("isFraud", "size"),
    fraudes=("isFraud", "sum"),
    monto_fraude=("amount", lambda s: s[prueba.loc[s.index, "isFraud"] == 1].sum()),
)
resumen
```

<details><summary><b>Ver solución</b></summary>

```python
prueba["accion"] = np.select(
    [(prueba["prob"] >= 0.9) | (prueba["isFlaggedFraud"] == 1), prueba["prob"] >= 0.5],
    ["BLOQUEAR", "REVISAR"], default="APROBAR")

resumen = prueba.groupby("accion").agg(
    transacciones=("isFraud", "size"),
    fraudes=("isFraud", "sum"),
    monto_fraude=("amount", lambda s: s[prueba.loc[s.index, "isFraud"] == 1].sum()),
)
resumen
```

</details>

**Verifica tu resultado:** la columna `fraudes` suma 200. La mayoría de las transacciones quedan en APROBAR, y la mayor parte del monto de fraude queda en BLOQUEAR y REVISAR.

### 6.2 Comparación final

```python
solo_regla = prueba.loc[prueba["isFraud"] == 1, "isFlaggedFraud"].___()
esquema = prueba.loc[prueba["isFraud"] == 1, "accion"].isin(["BLOQUEAR", ___]).mean()
print(f"Fraudes detenidos solo con la regla actual: {solo_regla:.1%}")
print(f"Fraudes detenidos con el esquema propuesto: {esquema:.1%}")
```

<details><summary><b>Ver solución</b></summary>

```python
solo_regla = prueba.loc[prueba["isFraud"] == 1, "isFlaggedFraud"].mean()
esquema = prueba.loc[prueba["isFraud"] == 1, "accion"].isin(["BLOQUEAR", "REVISAR"]).mean()
print(f"Fraudes detenidos solo con la regla actual: {solo_regla:.1%}")
print(f"Fraudes detenidos con el esquema propuesto: {esquema:.1%}")
```

</details>

**Verifica tu resultado:** el esquema propuesto detiene muchísimos más fraudes que la regla actual, que en el conjunto de prueba detiene muy pocos o ninguno.

#### Hints para interpretar tus resultados

* **Hint de regla contra modelo:** *La regla actual nunca se equivoca, pero casi nunca detecta. El modelo detecta mucho más, pero genera falsas alarmas. ¿Por qué conviene combinarlos en lugar de elegir uno?*
* **Hint de adaptación:** *Si los defraudadores aprenden que las transferencias de más de 200,000 se bloquean, ¿qué harán? ¿Qué ventaja tiene un modelo que se reentrena con los casos confirmados?*
* **Hint de tiempo real:** *¿Qué variables de este modelo están disponibles en el instante de la transacción y cuáles no? Relaciónalo con lo que aprendiste sobre fuga de datos.*

**Tu interpretación:**

*(Escribe aquí tus conclusiones de esta parte y tu esquema de monitoreo en tiempo real.)*

---
## Cierre de la práctica

Antes de redactar tu reporte técnico, responde:

* ¿Qué variables y patrones fueron más relevantes para detectar fraude?
* ¿Por qué un modelo perfecto en el laboratorio puede ser inútil en la operación?
* ¿Qué riesgos implica clasificar incorrectamente una transacción, en número de casos y en dinero?
* ¿Cómo cambian las conclusiones entre la Práctica 6 (datos sin señal) y esta práctica (datos con un patrón real)?
* ¿Qué ventajas ofrece el monitoreo automatizado frente a una regla fija del negocio?

**Lista de evidencias:** código fuente (este notebook), gráficas de análisis exploratorio, tabla comparativa de modelos, matrices de confusión, análisis de fuga de datos, cálculo de dinero protegido y ROI, esquema de vigilancia y reporte técnico de 2–3 cuartillas (interpretación de resultados, análisis de datos y propuesta de mejora en seguridad financiera).
