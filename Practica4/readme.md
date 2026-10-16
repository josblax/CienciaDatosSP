# Autopráctica 4. Predicción de fallas y optimización de la producción

**Ciencia de Datos en el Sector Privado (CDT015)** · Práctica 4 de 6 · Duración: 2 horas

## Cómo usar esta autopráctica

Cada ejercicio trae un objetivo, una plantilla de código con espacios `___` para completar, pistas, la solución plegada y el resultado esperado. Intenta resolver cada ejercicio antes de abrir la solución. Si tu resultado no coincide con el esperado, revisa la pista antes de copiar la solución.

Los ejercicios se ejecutan en orden: cada uno usa variables creadas en los anteriores.

**Requisitos:** Python con pandas, numpy, matplotlib, seaborn y scikit-learn.

## El caso

Una manufacturera dedicada a la producción de componentes electrónicos desea mejorar la eficiencia de su línea de producción. Cuenta con sensores industriales que registran, para cada máquina, la temperatura del equipo, la vibración del motor, la velocidad de producción, el número de piezas defectuosas y el tiempo de operación.

La empresa necesita un sistema de mantenimiento predictivo y control de calidad que permita:

1. Detectar posibles fallas antes de que ocurran.
2. Reducir productos defectuosos.
3. Optimizar la cadena de producción.

## Parte 0. Preparación

### Ejercicio 0.1 Generar los datos

Copia el siguiente script en un archivo `datos_p4.py` o en la primera celda de tu notebook y ejecútalo. Genera el archivo `datos_produccion.csv` con 1,000 registros.

> **Nota:** este script es una versión ajustada del que trae el practicario. En la versión original las fallas y los defectos casi no dependían de las condiciones de la máquina, y ningún modelo superaba al azar. Aquí la falla depende de la temperatura, la vibración y las horas de operación, y los defectos dependen de la temperatura, la vibración y la velocidad.

```python
import numpy as np
import pandas as pd

# Semilla para reproducibilidad
np.random.seed(42)

# Número de registros
n = 1000

# Generación de variables de producción
temperatura_maquina = np.random.normal(75, 10, n)      # °C
vibracion_motor = np.random.normal(0.6, 0.2, n)        # vibración relativa
velocidad_produccion = np.random.normal(120, 15, n)    # piezas por minuto
tiempo_operacion = np.random.randint(1, 10, n)         # horas de operación

# Defectos generados en producción
# (aumentan con la temperatura, la vibración y la velocidad de la línea)
tasa_defectos = np.exp(
    0.5
    + 0.03 * (temperatura_maquina - 75)
    + 1.0 * (vibracion_motor - 0.6)
    + 0.015 * (velocidad_produccion - 120)
)
defectos_producto = np.random.poisson(tasa_defectos)

# Probabilidad de falla basada en condiciones
# (depende de temperatura, vibración y horas de operación; no de la velocidad)
logit_falla = (
    -3.2
    + 0.25 * (temperatura_maquina - 75)
    + 12.0 * (vibracion_motor - 0.6)
    + 0.3 * (tiempo_operacion - 5)
    + np.random.normal(0, 0.3, n)
)

prob_falla = 1 / (1 + np.exp(-logit_falla))

# Generar variable objetivo
falla_maquina = (np.random.rand(n) < prob_falla).astype(int)

# Crear DataFrame
df = pd.DataFrame({
    "temperatura_maquina": np.round(temperatura_maquina, 2),
    "vibracion_motor": np.round(vibracion_motor, 3),
    "velocidad_produccion": np.round(velocidad_produccion, 0).astype(int),
    "defectos_producto": defectos_producto,
    "tiempo_operacion": tiempo_operacion,
    "falla_maquina": falla_maquina
})

# Guardar dataset
df.to_csv("datos_produccion.csv", index=False)
print("Dataset generado correctamente: datos_produccion.csv")
print("Total de registros:", len(df))
print(df.head())
```

**Resultado esperado:** el mensaje `Dataset generado correctamente: datos_produccion.csv`, `Total de registros: 1000` y las cinco primeras filas. La primera fila tiene temperatura 79.97, vibración 0.880 y `falla_maquina` igual a 1.

---

### Ejercicio 0.2 Cargar el dataset de producción (Paso 1)

Importa las librerías de toda la práctica, carga el archivo y revisa su tamaño, tipos de dato y datos faltantes.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ___
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import ___
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
df = pd.___('datos_produccion.csv')
print(df.___)
print(df.dtypes)
print(df.___().sum().sum())
```

**Pistas**

- La función que divide los datos en entrenamiento y prueba vive en `sklearn.model_selection`.
- El bosque de árboles para clasificar se llama `RandomForest...` y termina en `Classifier`.
- Para leer un archivo CSV con pandas se usa `read_csv`.
- El atributo que da (filas, columnas) no lleva paréntesis.
- El método que marca con `True` cada celda vacía es `isna`.

<details>
<summary>Ver solución</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
df = pd.read_csv('datos_produccion.csv')
print(df.shape)
print(df.dtypes)
print(df.isna().sum().sum())
```

</details>

**Resultado esperado:** `(1000, 6)`. Las columnas de medición son `float64` y las de conteo `int64`. Datos faltantes: `0`.

---

## Parte 1. Análisis exploratorio de datos

### Ejercicio 1.1 Estadísticas descriptivas (Pasos 2 y 3)

Calcula media, mediana, desviación estándar, mínimo y máximo de las tres variables principales, y la proporción de registros con falla.

```python
principales = ['temperatura_maquina', 'vibracion_motor', '___']
resumen = df[principales].agg(['___', 'median', '___', 'min', 'max']).round(2)
print(resumen)
print('Tasa de falla:', round(df['___'].mean(), 3))
```

**Pistas**

- La tercera variable principal es la velocidad de producción; copia el nombre exacto de la columna.
- Los nombres de las funciones de `agg` van en inglés: promedio y desviación estándar.
- El promedio de una columna de ceros y unos es la proporción de unos.

<details>
<summary>Ver solución</summary>

```python
principales = ['temperatura_maquina', 'vibracion_motor', 'velocidad_produccion']
resumen = df[principales].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
print(resumen)
print('Tasa de falla:', round(df['falla_maquina'].mean(), 3))
```

</details>

**Resultado esperado:** | | temperatura | vibración | velocidad |
|---|---|---|---|
| media | 75.19 | 0.61 | 120.07 |
| mediana | 75.26 | 0.61 | 120.00 |
| desv. estándar | 9.79 | 0.20 | 14.74 |
| mínimo | 42.59 | 0.01 | 75.00 |
| máximo | 113.53 | 1.24 | 179.00 |

Tasa de falla: **0.232** (232 de 1,000 registros).

**Interpreta:** La media y la mediana casi coinciden en las tres variables. ¿Qué te dice eso sobre la forma de sus distribuciones?

---

### Ejercicio 1.2 Gráficas de distribución (Paso 3)

Dibuja un histograma por cada variable principal.

```python
fig, axes = plt.subplots(1, ___, figsize=(14, 4))
for ax, col in zip(axes, ___):
    sns.___(df[col], bins=30, kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
```

**Pistas**

- Necesitas una gráfica por variable principal.
- El ciclo recorre la lista que definiste en el ejercicio anterior.
- La función de seaborn para histogramas es `histplot`.

<details>
<summary>Ver solución</summary>

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, principales):
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
```

</details>

**Resultado esperado:** tres histogramas con forma de campana, centrados cerca de 75 °C, 0.6 y 120 piezas por minuto.

---

### Ejercicio 1.3 Patrones iniciales: máquinas con y sin falla (Paso 3)

Compara el promedio de cada variable entre registros sin falla (0) y con falla (1), y grafica las dos variables que más se separan.

```python
comparacion = df.groupby('___')[principales + ['tiempo_operacion', 'defectos_producto']].___().round(2)
print(comparacion)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.___(data=df, x='falla_maquina', y='temperatura_maquina', ax=axes[0])
sns.boxplot(data=df, x='falla_maquina', y='___', ax=axes[1])
plt.tight_layout()
plt.show()
```

**Pistas**

- Agrupa por la columna que indica si la máquina falló.
- Calcula el promedio de cada grupo.
- La gráfica de caja de seaborn es `boxplot`.
- La segunda gráfica es para la vibración del motor.

<details>
<summary>Ver solución</summary>

```python
comparacion = df.groupby('falla_maquina')[principales + ['tiempo_operacion', 'defectos_producto']].mean().round(2)
print(comparacion)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x='falla_maquina', y='temperatura_maquina', ax=axes[0])
sns.boxplot(data=df, x='falla_maquina', y='vibracion_motor', ax=axes[1])
plt.tight_layout()
plt.show()
```

</details>

**Resultado esperado:** | falla | temperatura | vibración | velocidad | horas | defectos |
|---|---|---|---|---|---|
| 0 | 72.69 | 0.57 | 119.84 | 4.85 | 1.57 |
| 1 | 83.47 | 0.77 | 120.82 | 5.81 | 2.54 |

Las cajas de temperatura y vibración del grupo con falla quedan claramente más arriba.

**Interpreta:** ¿Qué variable casi no cambia entre los dos grupos? ¿Esperarías que sirva para predecir fallas?

---

## Parte 2. Identificación de defectos en producción

### Ejercicio 2.1 Relación entre temperatura, vibración y defectos (Paso 4)

Calcula la matriz de correlación y dibuja la dispersión de cada variable contra los defectos.

```python
correlacion = df[['temperatura_maquina', 'vibracion_motor', '___']].___().round(3)
print(correlacion)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.___(data=df, x='temperatura_maquina', y='defectos_producto', alpha=0.4, ax=axes[0])
sns.scatterplot(data=df, x='___', y='defectos_producto', alpha=0.4, ax=axes[1])
plt.tight_layout()
plt.show()
```

**Pistas**

- La tercera columna es el número de piezas defectuosas.
- El método de pandas que calcula correlaciones es `corr`.
- La gráfica de dispersión de seaborn es `scatterplot`.
- La segunda gráfica usa la vibración en el eje x.

<details>
<summary>Ver solución</summary>

```python
correlacion = df[['temperatura_maquina', 'vibracion_motor', 'defectos_producto']].corr().round(3)
print(correlacion)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.scatterplot(data=df, x='temperatura_maquina', y='defectos_producto', alpha=0.4, ax=axes[0])
sns.scatterplot(data=df, x='vibracion_motor', y='defectos_producto', alpha=0.4, ax=axes[1])
plt.tight_layout()
plt.show()
```

</details>

**Resultado esperado:** correlación de defectos con temperatura **0.344** y con vibración **0.215**. Entre temperatura y vibración casi no hay relación (−0.040). En las dispersiones la nube sube ligeramente hacia la derecha.

**Interpreta:** Una correlación de 0.344 es moderada. ¿Por qué los puntos se ven tan dispersos aunque la relación exista?

---

### Ejercicio 2.2 Defectos por rango de temperatura (Paso 4)

Agrupa la temperatura en cuatro rangos y calcula cuántos registros hay y cuántos defectos se generan en promedio en cada uno.

```python
df['rango_temp'] = pd.___(df['temperatura_maquina'], bins=[0, 65, 75, 85, 200],
                          labels=['<65', '65-75', '75-85', '>85'])
por_rango = df.groupby('___', observed=True)['defectos_producto'].agg(['___', 'mean']).round(2)
print(por_rango)
```

**Pistas**

- La función de pandas que convierte una variable numérica en rangos es `cut`.
- Agrupa por la columna de rangos que acabas de crear.
- La función que cuenta registros se llama `count`.

<details>
<summary>Ver solución</summary>

```python
df['rango_temp'] = pd.cut(df['temperatura_maquina'], bins=[0, 65, 75, 85, 200],
                          labels=['<65', '65-75', '75-85', '>85'])
por_rango = df.groupby('rango_temp', observed=True)['defectos_producto'].agg(['count', 'mean']).round(2)
print(por_rango)
```

</details>

**Resultado esperado:** | rango | registros | defectos promedio |
|---|---|---|
| <65 | 147 | 1.13 |
| 65-75 | 343 | 1.45 |
| 75-85 | 355 | 1.97 |
| >85 | 155 | 2.78 |

**Interpreta:** Por encima de 85 °C los defectos casi se duplican respecto a 65-75 °C. ¿Qué acción de control de calidad propondrías con este dato?

---

### Ejercicio 2.3 Variables que más influyen en los defectos (Paso 5)

Ordena las cuatro variables de operación según su correlación con los defectos.

```python
variables = ['temperatura_maquina', 'vibracion_motor', 'velocidad_produccion', '___']
influencia = df[___].corrwith(df['___']).sort_values(ascending=___).round(3)
print(influencia)
```

**Pistas**

- La cuarta variable son las horas de operación.
- `corrwith` se aplica a las columnas de la lista `variables`.
- Correlaciona contra la columna de defectos.
- Para ver la más fuerte primero, el orden no es ascendente.

<details>
<summary>Ver solución</summary>

```python
variables = ['temperatura_maquina', 'vibracion_motor', 'velocidad_produccion', 'tiempo_operacion']
influencia = df[variables].corrwith(df['defectos_producto']).sort_values(ascending=False).round(3)
print(influencia)
```

</details>

**Resultado esperado:** temperatura **0.344**, velocidad **0.294**, vibración **0.215** y horas de operación **0.018**.

**Interpreta:** La velocidad influye en los defectos, pero en el ejercicio 1.3 casi no cambiaba entre máquinas con y sin falla. ¿Qué significa que una variable explique los defectos pero no las fallas?

---

## Parte 3. Desarrollo del modelo predictivo

### Ejercicio 3.1 Dividir en entrenamiento y prueba (Paso 6)

Separa las variables de entrada de la variable objetivo y divide los datos en 80% entrenamiento y 20% prueba.

```python
X = df[['temperatura_maquina', 'vibracion_motor', 'velocidad_produccion',
        'defectos_producto', 'tiempo_operacion']]
y = df['___']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=___, random_state=42, stratify=___)
print(X_train.shape, X_test.shape)
print('Fallas en entrenamiento:', y_train.sum(), '| en prueba:', y_test.___())
```

**Pistas**

- La variable objetivo indica si la máquina falló.
- El 20% se escribe como proporción.
- `stratify` recibe la variable objetivo para que ambos conjuntos tengan la misma proporción de fallas.
- Cuenta las fallas de prueba igual que las de entrenamiento.

<details>
<summary>Ver solución</summary>

```python
X = df[['temperatura_maquina', 'vibracion_motor', 'velocidad_produccion',
        'defectos_producto', 'tiempo_operacion']]
y = df['falla_maquina']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)
print('Fallas en entrenamiento:', y_train.sum(), '| en prueba:', y_test.sum())
```

</details>

**Resultado esperado:** `(800, 5) (200, 5)`. Fallas en entrenamiento: **186**; en prueba: **46**.

---

### Ejercicio 3.2 Árbol de decisión (Pasos 7 y 8)

Entrena un árbol de decisión con un máximo de 4 niveles y predice el conjunto de prueba.

```python
arbol = ___(max_depth=___, random_state=42)
arbol.___(X_train, y_train)
pred_arbol = arbol.___(X_test)
print('Accuracy entrenamiento:', round(arbol.score(X_train, y_train), 3))
```

**Pistas**

- El clasificador de árbol ya está importado desde `sklearn.tree`.
- Limita la profundidad a 4 niveles.
- El método que entrena es `fit` y el que genera predicciones es `predict`.

<details>
<summary>Ver solución</summary>

```python
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol.fit(X_train, y_train)
pred_arbol = arbol.predict(X_test)
print('Accuracy entrenamiento:', round(arbol.score(X_train, y_train), 3))
```

</details>

**Resultado esperado:** accuracy en entrenamiento **0.901**.

---

### Ejercicio 3.3 Random Forest (Pasos 7 y 8)

Entrena un Random Forest de 200 árboles con los mismos datos.

```python
bosque = ___(n_estimators=___, random_state=42)
bosque.fit(___, ___)
pred_bosque = bosque.predict(X_test)
print('Accuracy entrenamiento:', round(bosque.score(X_train, y_train), 3))
```

**Pistas**

- El clasificador de bosque ya está importado desde `sklearn.ensemble`.
- Usa 200 árboles.
- Se entrena con los datos de entrenamiento, no con los de prueba.

<details>
<summary>Ver solución</summary>

```python
bosque = RandomForestClassifier(n_estimators=200, random_state=42)
bosque.fit(X_train, y_train)
pred_bosque = bosque.predict(X_test)
print('Accuracy entrenamiento:', round(bosque.score(X_train, y_train), 3))
```

</details>

**Resultado esperado:** accuracy en entrenamiento **1.0**. El bosque memoriza el entrenamiento; por eso la evaluación real se hace con prueba.

---

## Parte 4. Evaluación del modelo

### Ejercicio 4.1 Accuracy contra la línea base (Paso 9)

Compara el accuracy de los dos modelos con el de un modelo que siempre dice "sin falla".

```python
linea_base = 1 - y_test.___()
print('Línea base (predecir siempre "sin falla"):', round(linea_base, 3))
print('Árbol de decisión:', round(___(y_test, pred_arbol), 3))
print('Random Forest:', round(accuracy_score(___, ___), 3))
```

**Pistas**

- La proporción de fallas en prueba es el promedio de `y_test`.
- La función de accuracy ya está importada desde `sklearn.metrics`.
- Primero van los valores reales y después las predicciones.

<details>
<summary>Ver solución</summary>

```python
linea_base = 1 - y_test.mean()
print('Línea base (predecir siempre "sin falla"):', round(linea_base, 3))
print('Árbol de decisión:', round(accuracy_score(y_test, pred_arbol), 3))
print('Random Forest:', round(accuracy_score(y_test, pred_bosque), 3))
```

</details>

**Resultado esperado:** línea base **0.77**, árbol **0.875** y Random Forest **0.915**.

**Interpreta:** Un modelo que nunca detecta una falla tendría 77% de accuracy. ¿Por qué el accuracy solo no basta para evaluar este sistema?

---

### Ejercicio 4.2 Matriz de confusión (Paso 9)

Obtén la matriz de confusión de cada modelo y calcula qué porcentaje de las fallas reales detecta.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, nombre in [(axes[0], pred_arbol, 'Árbol'), (axes[1], ___, 'Random Forest')]:
    vn, fp, fn, vp = ___(y_test, pred).ravel()
    print(f'{nombre}: VN={vn} FP={fp} FN={fn} VP={vp} | fallas detectadas: {vp}/{vp + fn} = {vp / (___ + ___):.1%}')
    ConfusionMatrixDisplay.___(y_test, pred, display_labels=['Sin falla', 'Falla'], ax=ax, colorbar=False)
    ax.set_title(nombre)
plt.tight_layout()
plt.show()
```

**Pistas**

- La segunda pareja usa las predicciones del bosque.
- La función que arma la matriz ya está importada.
- Las fallas reales son las detectadas (VP) más las no detectadas (FN).
- El método de `ConfusionMatrixDisplay` que recibe valores reales y predicciones es `from_predictions`.

<details>
<summary>Ver solución</summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, nombre in [(axes[0], pred_arbol, 'Árbol'), (axes[1], pred_bosque, 'Random Forest')]:
    vn, fp, fn, vp = confusion_matrix(y_test, pred).ravel()
    print(f'{nombre}: VN={vn} FP={fp} FN={fn} VP={vp} | fallas detectadas: {vp}/{vp + fn} = {vp / (vp + fn):.1%}')
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['Sin falla', 'Falla'], ax=ax, colorbar=False)
    ax.set_title(nombre)
plt.tight_layout()
plt.show()
```

</details>

**Resultado esperado:** | modelo | VN | FP | FN | VP | fallas detectadas |
|---|---|---|---|---|---|
| Árbol | 144 | 10 | 15 | 31 | 31/46 = 67.4% |
| Random Forest | 147 | 7 | 10 | 36 | 36/46 = 78.3% |

**Interpreta:** Un falso negativo es una máquina que falla sin aviso; un falso positivo es una revisión que no hacía falta. ¿Cuál le cuesta más a la planta y por qué?

---

### Ejercicio 4.3 Importancia de variables (Paso 10)

Compara qué variables usa más cada modelo para decidir.

```python
importancias = pd.DataFrame({
    'arbol': arbol.___,
    'random_forest': ___.feature_importances_
}, index=X.___).sort_values('random_forest', ascending=False).round(3)
print(importancias)
```

**Pistas**

- La importancia de cada variable es un atributo que termina en guion bajo.
- La segunda columna es del Random Forest.
- El índice son los nombres de las columnas de `X`.

<details>
<summary>Ver solución</summary>

```python
importancias = pd.DataFrame({
    'arbol': arbol.feature_importances_,
    'random_forest': bosque.feature_importances_
}, index=X.columns).sort_values('random_forest', ascending=False).round(3)
print(importancias)
```

</details>

**Resultado esperado:** | variable | árbol | Random Forest |
|---|---|---|
| temperatura | 0.496 | 0.371 |
| vibración | 0.446 | 0.355 |
| velocidad | 0.014 | 0.119 |
| horas de operación | 0.044 | 0.089 |
| defectos | 0.000 | 0.067 |

**Interpreta:** El Random Forest le da 0.119 a la velocidad, aunque en el ejercicio 1.3 casi no cambiaba entre grupos. ¿Por qué un modelo puede repartir algo de importancia a una variable que no causa la falla?

---

### Ejercicio 4.4 Leer las reglas del árbol (Paso 10)

Dibuja los dos primeros niveles del árbol para ver qué preguntas hace.

```python
plt.figure(figsize=(14, 6))
___(arbol, max_depth=___, feature_names=list(X.___),
          class_names=['Sin falla', 'Falla'], filled=True, rounded=True, fontsize=9)
plt.show()
```

**Pistas**

- La función que dibuja el árbol se importó junto con el clasificador.
- Muestra solo dos niveles para que se lea bien.
- Los nombres de las variables salen de las columnas de `X`.

<details>
<summary>Ver solución</summary>

```python
plt.figure(figsize=(14, 6))
plot_tree(arbol, max_depth=2, feature_names=list(X.columns),
          class_names=['Sin falla', 'Falla'], filled=True, rounded=True, fontsize=9)
plt.show()
```

</details>

**Resultado esperado:** la primera pregunta es **¿temperatura_maquina ≤ 81.93?** y en el segundo nivel aparecen dos preguntas sobre la vibración (≤ 0.81 y ≤ 0.61). Cada recuadro muestra su índice de Gini, el número de registros y la clase mayoritaria.

**Interpreta:** Traduce la rama de la derecha a una regla de operación que entendería un supervisor de planta.

---

## Parte 5. Propuesta de optimización

### Ejercicio 5.1 Umbrales de operación (Paso 11)

Construye una tabla con la tasa de falla para cada combinación de rango de temperatura y rango de vibración.

```python
df['rango_vib'] = pd.cut(df['___'], bins=[-1, 0.5, 0.7, 0.9, 5],
                         labels=['<0.5', '0.5-0.7', '0.7-0.9', '>0.9'])
tabla_riesgo = pd.___(df, values='___', index='rango_temp',
                              columns='___', aggfunc='mean', observed=True).round(2)
print(tabla_riesgo)
```

**Pistas**

- Los rangos se calculan sobre la vibración del motor.
- La tabla cruzada con promedios es `pivot_table`.
- El valor que se promedia es la columna de falla.
- Las columnas son los rangos de vibración que acabas de crear.

<details>
<summary>Ver solución</summary>

```python
df['rango_vib'] = pd.cut(df['vibracion_motor'], bins=[-1, 0.5, 0.7, 0.9, 5],
                         labels=['<0.5', '0.5-0.7', '0.7-0.9', '>0.9'])
tabla_riesgo = pd.pivot_table(df, values='falla_maquina', index='rango_temp',
                              columns='rango_vib', aggfunc='mean', observed=True).round(2)
print(tabla_riesgo)
```

</details>

**Resultado esperado:** | temperatura \ vibración | <0.5 | 0.5-0.7 | 0.7-0.9 | >0.9 |
|---|---|---|---|---|
| <65 | 0.00 | 0.00 | 0.02 | 0.13 |
| 65-75 | 0.00 | 0.03 | 0.22 | 0.48 |
| 75-85 | 0.01 | 0.16 | 0.62 | 0.90 |
| >85 | 0.29 | 0.61 | 0.94 | 1.00 |

La columna `>0.9` tiene pocos registros (entre 13 y 29 por celda), así que sus tasas son menos estables.

**Interpreta:** Define una zona verde, una amarilla y una roja de operación con esta tabla. ¿Qué haría el sistema cuando una máquina entra en zona roja?

---

### Ejercicio 5.2 Impacto económico del mantenimiento predictivo (Paso 11)

Estima cuánto ahorraría la planta en el periodo de prueba con supuestos ilustrativos: una falla no prevista cuesta 50,000 y una revisión preventiva 8,000.

```python
costo_falla = ___          # paro no programado + reparación (supuesto)
costo_preventivo = ___      # revisión programada de una máquina (supuesto)

vn, fp, fn, vp = confusion_matrix(y_test, pred_bosque).ravel()
sin_modelo = (vp + ___) * costo_falla
con_modelo = ___ * costo_falla + (vp + fp) * costo_preventivo
ahorro = sin_modelo - con_modelo
print('Costo sin modelo:', f'{sin_modelo:,}')
print('Costo con modelo:', f'{con_modelo:,}')
print('Ahorro estimado:', f'{ahorro:,}', f'({ahorro / sin_modelo:.1%})')
```

**Pistas**

- Usa los dos costos del enunciado.
- Sin modelo, todas las fallas reales (detectadas o no) ocurren sin aviso.
- Con modelo, solo las fallas no detectadas siguen costando como falla; las alertas (verdaderas y falsas) cuestan una revisión.

<details>
<summary>Ver solución</summary>

```python
costo_falla = 50000          # paro no programado + reparación (supuesto)
costo_preventivo = 8000      # revisión programada de una máquina (supuesto)

vn, fp, fn, vp = confusion_matrix(y_test, pred_bosque).ravel()
sin_modelo = (vp + fn) * costo_falla
con_modelo = fn * costo_falla + (vp + fp) * costo_preventivo
ahorro = sin_modelo - con_modelo
print('Costo sin modelo:', f'{sin_modelo:,}')
print('Costo con modelo:', f'{con_modelo:,}')
print('Ahorro estimado:', f'{ahorro:,}', f'({ahorro / sin_modelo:.1%})')
```

</details>

**Resultado esperado:** costo sin modelo **2,300,000**, costo con modelo **844,000** y ahorro estimado **1,456,000 (63.3%)**. Los costos son ilustrativos; en una planta real se toman de mantenimiento y finanzas.

**Interpreta:** ¿Cómo cambia el ahorro si la revisión preventiva costara 20,000? Pruébalo y explica qué pasa con el valor de cada falso positivo.

---

### Ejercicio 5.3 Estrategias de mejora (Paso 11)

Completa la tabla con una estrategia por área, apoyándote en los resultados de los ejercicios anteriores.

| Área | Hallazgo en los datos | Estrategia propuesta |
|---|---|---|
| Mantenimiento | ___ | ___ |
| Control de calidad | ___ | ___ |
| Operación de la línea | ___ | ___ |

**Pistas**

- Mantenimiento: usa el modelo con mejor detección de fallas del ejercicio 4.2 y la tabla de zonas del 5.1.
- Control de calidad: usa los defectos por rango de temperatura del 2.2.
- Operación: recuerda qué variable afecta a los defectos pero no a las fallas (2.3 y 1.3).

---

### Ejercicio 5.4 Visión artificial para detectar defectos (Paso 12)

Los sensores de esta práctica miden la máquina, pero no la pieza. Un sistema de visión artificial usa cámaras en la línea y un modelo de redes neuronales convolucionales entrenado con fotos de piezas buenas y defectuosas. Responde:

1. ¿Qué defectos podría detectar una cámara que los sensores actuales no ven?
2. ¿En qué punto de la línea colocarías la cámara y por qué?
3. ¿Qué datos necesitarías reunir antes de entrenar el modelo?
4. ¿Cómo combinarías la alerta de la cámara con el modelo de fallas de esta práctica?

---

## Conclusiones

Responde en tu reporte técnico:

1. ¿Cómo puede la ciencia de datos reducir fallas en la producción?
2. ¿Qué variables resultaron más importantes en la detección de fallas?
3. ¿Qué beneficios económicos tendría implementar mantenimiento predictivo?
4. ¿Cómo podría integrarse visión artificial en el control de calidad?

## Evidencia de la práctica

- Código fuente (este notebook o script con todos los ejercicios resueltos).
- Gráficas de análisis exploratorio.
- Resultados del modelo predictivo: accuracy, matrices de confusión e importancia de variables.
- Reporte técnico de 2 a 3 cuartillas con interpretación de resultados, análisis de resultados y propuesta de mejora productiva.
