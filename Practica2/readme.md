## Autopráctica 2: Aplicación de ciencia de datos en procesos comerciales

**Ciencia de Datos en el Sector Privado (CDT015)**

**Caso.** Una empresa de comercio electrónico quiere aumentar sus ingresos entendiendo mejor el comportamiento de sus clientes. Cuenta con edad, frecuencia de compra, monto promedio, canal de adquisición (web, app, redes sociales) y el historial de compras de los últimos 12 meses de 500 clientes. La dirección pide identificar segmentos, detectar oportunidades de personalización y proponer estrategias comerciales diferenciadas.

**Objetivo.** Aplicar técnicas de segmentación de clientes para identificar patrones de comportamiento, diseñar estrategias de personalización y proponer acciones que agreguen valor a la organización.

**Instrucciones**

1. Ejecuta las celdas en orden, de arriba hacia abajo. Cada ejercicio usa variables creadas en los anteriores.
2. En cada ejercicio, reemplaza los `___` por el código correcto y ejecuta la celda.
3. Si te atoras, abre la **Pista**. Abre **Ver solución** solo después de intentarlo.
4. Revisa la sección **Verifica tu resultado** para comprobar que vas bien.
5. Al final de cada parte responde **Tu interpretación**. Esas respuestas son la base de tu reporte técnico.

**Hints**

* **Datos sintéticos:** el dataset se genera con semilla 42. Si no cambias la semilla, tus números deben coincidir con los resultados esperados.
* **Nombres de columnas:** respeta mayúsculas y minúsculas: `FrecuenciaCompra`, `MontoPromedio`, `Historial12Meses`, `Canal`, `Edad`.
* **Supuestos:** los porcentajes de impacto, margen y retención de las Partes 3 y 4 son ilustrativos. Al terminar, cámbialos por los tuyos y justifícalos en el reporte.
* **Requisitos:** `pip install numpy pandas matplotlib scikit-learn`

### Parte 0. Preparación

Carga las librerías y genera el dataset con el script de la práctica. Esta celda ya está completa: solo ejecútala.

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# Script de la práctica (genera el dataset)
np.random.seed(42)
data = pd.DataFrame({
    "Edad": np.random.randint(18, 70, 500),
    "FrecuenciaCompra": np.random.randint(1, 20, 500),
    "MontoPromedio": np.random.randint(100, 5000, 500),
    "Canal": np.random.choice(["Web", "App", "Redes"], 500)
})
data["Historial12Meses"] = (
    data["FrecuenciaCompra"] *
    data["MontoPromedio"] *
    np.random.uniform(0.8, 1.2, len(data))
)
print("Primeros registros:")
print(data.head())
```

**Verifica tu resultado:** el primer cliente tiene 56 años, 16 compras al año, monto promedio de 1,348, canal Web e historial cercano a 20,416.

#### 0.1 Revisar la estructura del dataset

Antes de analizar, confirma cuántos clientes y variables hay, el tipo de cada columna y si hay datos faltantes.

```python
print(data.___)
data.___()
print(data.isna().___())
```

<details><summary><b>Pista</b></summary>

* El atributo que devuelve (filas, columnas) no lleva paréntesis.
* El método que muestra tipos de datos y valores no nulos es el mismo que usarías para obtener *información* general.
* Para contar nulos por columna, suma los `True` que devuelve `isna()`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
print(data.shape)
data.info()
print(data.isna().sum())
```

</details>

**Verifica tu resultado:** `(500, 5)`; cuatro columnas numéricas y `Canal` de tipo texto (`object`, o `str` en pandas 3); cero nulos en todas las columnas.

### Parte 1. Análisis exploratorio

Pasos 1 a 3 de la práctica: estadísticos descriptivos, patrones preliminares y gráficas.

#### 1.1 Estadísticos descriptivos (Paso 1)

Calcula media, mediana y desviación estándar de la edad y de las tres variables de compra.

```python
compras = [___, ___, ___]
variables = ['Edad'] + compras
resumen = data[variables].agg([___, ___, ___]).round(1)
resumen
```

<details><summary><b>Pista</b></summary>

* Las tres variables de compra son la frecuencia, el monto promedio y el historial de 12 meses.
* `agg` recibe una lista de nombres de funciones como texto. En pandas la desviación estándar se llama `'std'`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
compras = ['FrecuenciaCompra', 'MontoPromedio', 'Historial12Meses']
variables = ['Edad'] + compras
resumen = data[variables].agg(['mean', 'median', 'std']).round(1)
resumen
```

</details>

**Verifica tu resultado:** frecuencia media 9.9 y mediana 10.0; monto medio 2,628.9 y mediana 2,729.0; historial medio 25,654.9, mediana 19,394.1 y desviación 21,435.2. En el historial la media queda muy por encima de la mediana: es la primera señal de asimetría.

#### 1.2 Patrones preliminares por correlación y canal (Paso 2)

Revisa si las variables se relacionan entre sí, cómo se reparten los clientes por canal y si algún canal compra distinto.

```python
print(data[variables].___().round(2))
print(data[___].value_counts(normalize=___).round(3))
print(data.___('Canal')[compras].___().round(1))
```

<details><summary><b>Pista</b></summary>

* La matriz de correlación se obtiene con un método de tres letras.
* `normalize` convierte conteos en proporciones; necesita un valor booleano.
* Para comparar canales, agrupa por `Canal` y calcula el promedio.

</details>

<details><summary><b>Ver solución</b></summary>

```python
print(data[variables].corr().round(2))
print(data['Canal'].value_counts(normalize=True).round(3))
print(data.groupby('Canal')[compras].mean().round(1))
```

</details>

**Verifica tu resultado:** la correlación entre frecuencia y monto es casi nula (-0.06); el historial se correlaciona con ambas (0.65 y 0.60) porque se calcula a partir de ellas. Los canales están balanceados (App 35.2%, Web 33.4%, Redes 31.4%) y sus promedios de compra son muy parecidos.

#### 1.3 Concentración de las compras (Pareto)

Calcula qué porcentaje del total de compras aporta el 20% de clientes con mayor historial.

```python
ordenado = data['Historial12Meses'].sort_values(ascending=___)
n_top = int(len(data) * ___)
top20 = ordenado.___(n_top).sum() / ordenado.___()
print(f"El 20% de clientes con mayor historial concentra {top20:.1%} de las compras")
```

<details><summary><b>Pista</b></summary>

* Para quedarte con los mejores clientes primero, ordena de mayor a menor.
* El 20% se expresa como proporción.
* Toma los primeros `n_top` registros y divide su suma entre la suma de todos.

</details>

<details><summary><b>Ver solución</b></summary>

```python
ordenado = data['Historial12Meses'].sort_values(ascending=False)
n_top = int(len(data) * 0.2)
top20 = ordenado.head(n_top).sum() / ordenado.sum()
print(f"El 20% de clientes con mayor historial concentra {top20:.1%} de las compras")
```

</details>

**Verifica tu resultado:** El 20% de clientes con mayor historial concentra **47.3%** de las compras (100 clientes).

#### 1.4 Distribución de compras (Paso 3)

Grafica un histograma para cada variable de compra.

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titulos = ['Frecuencia de compra (compras al año)',
           'Monto promedio por compra',
           'Compras de los últimos 12 meses']
for ax, col, titulo in zip(axes, ___, titulos):
    ax.___(data[col], bins=___, edgecolor='white')
    ax.set_title(titulo)
    ax.set_xlabel(col)
    ax.set_ylabel(___)
plt.tight_layout()
plt.show()
```

<details><summary><b>Pista</b></summary>

* `zip` recorre en paralelo los ejes, las columnas y los títulos: las columnas son la lista que creaste en 1.1.
* El método del eje que dibuja un histograma se llama igual que en inglés, abreviado.
* Usa 20 intervalos. El eje Y cuenta personas.

</details>

<details><summary><b>Ver solución</b></summary>

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titulos = ['Frecuencia de compra (compras al año)',
           'Monto promedio por compra',
           'Compras de los últimos 12 meses']
for ax, col, titulo in zip(axes, compras, titulos):
    ax.hist(data[col], bins=20, edgecolor='white')
    ax.set_title(titulo)
    ax.set_xlabel(col)
    ax.set_ylabel('Número de clientes')
plt.tight_layout()
plt.show()
```

</details>

**Verifica tu resultado:** frecuencia y monto se ven casi planos (uniformes); el historial tiene sesgo a la derecha: la mayoría de los clientes queda por debajo de 30,000 y una cola pequeña llega a 80,000 o más.

#### 1.5 Relación entre frecuencia y monto (Paso 3)

Haz un diagrama de dispersión de frecuencia contra monto, con un color por canal.

```python
fig, ax = plt.subplots(figsize=(7, 4.5))
for canal, grupo in data.___('Canal'):
    ax.scatter(grupo[___], grupo[___], s=14, alpha=0.6, label=___)
ax.set_xlabel('Frecuencia de compra (compras al año)')
ax.set_ylabel('Monto promedio por compra')
ax.set_title('Relación entre frecuencia y monto')
ax.legend(title='Canal')
plt.tight_layout()
plt.show()
```

<details><summary><b>Pista</b></summary>

* Recorrer un `groupby` entrega pares (nombre del grupo, subtabla).
* El eje X es la frecuencia y el eje Y el monto, como dicen las etiquetas.
* La etiqueta de cada serie es el nombre del canal.

</details>

<details><summary><b>Ver solución</b></summary>

```python
fig, ax = plt.subplots(figsize=(7, 4.5))
for canal, grupo in data.groupby('Canal'):
    ax.scatter(grupo['FrecuenciaCompra'], grupo['MontoPromedio'], s=14, alpha=0.6, label=canal)
ax.set_xlabel('Frecuencia de compra (compras al año)')
ax.set_ylabel('Monto promedio por compra')
ax.set_title('Relación entre frecuencia y monto')
ax.legend(title='Canal')
plt.tight_layout()
plt.show()
```

</details>

**Verifica tu resultado:** una nube sin tendencia, con columnas verticales (la frecuencia solo toma valores enteros de 1 a 19) y los tres canales mezclados en todas las zonas.

**Tu interpretación:**

1. ¿Qué te dicen la media y la mediana del historial sobre cómo se reparte el gasto entre clientes?
2. Si la frecuencia y el monto no están correlacionados, ¿qué tipos de cliente esperas encontrar al combinarlos?
3. ¿El canal de adquisición distingue a los clientes en este dataset? ¿Qué implica para una estrategia por canal?

_Escribe aquí tu respuesta._

### Parte 2. Segmentación

Pasos 4 a 6: K-Means con al menos 3 segmentos, número óptimo de clusters y descripción de cada segmento. Se segmenta por comportamiento de compra (frecuencia y monto).

#### 2.1 Estandarizar las variables

K-Means usa distancias. Sin escalar, el monto (cientos a miles) dominaría sobre la frecuencia (1 a 19). Estandariza ambas para que pesen igual.

```python
columnas_seg = [___, ___]
escalador = ___().fit(data[columnas_seg])
X = escalador.___(data[columnas_seg])
pd.DataFrame(X, columns=columnas_seg).describe().round(2)
```

<details><summary><b>Pista</b></summary>

* Solo se usan las dos variables de comportamiento, no el historial (que es su producto) ni la edad.
* La clase que deja cada variable con media 0 y desviación 1 ya está importada en la Parte 0.
* Después de `fit`, el método que aplica la escala es `transform`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
columnas_seg = ['FrecuenciaCompra', 'MontoPromedio']
escalador = StandardScaler().fit(data[columnas_seg])
X = escalador.transform(data[columnas_seg])
pd.DataFrame(X, columns=columnas_seg).describe().round(2)
```

</details>

**Verifica tu resultado:** en ambas columnas la media es 0.00 y la desviación estándar es 1.00.

#### 2.2 K-Means con 3 segmentos (Paso 4)

Aplica K-Means con el mínimo que pide la práctica y revisa el perfil y la calidad de los grupos.

```python
km3 = KMeans(n_clusters=___, n_init=10, random_state=42).fit(___)
prueba = data.assign(cluster=km3.___)
print(prueba.groupby('cluster')[compras].mean().round(1))
print('Clientes por cluster:', prueba['cluster'].value_counts().sort_index().tolist())
print('Silueta con k=3:', round(___(X, km3.labels_), 3))
```

<details><summary><b>Pista</b></summary>

* El modelo se entrena con los datos estandarizados, no con `data`.
* El atributo con el grupo asignado a cada cliente termina en guion bajo.
* La función que mide qué tan separados están los grupos se importó desde `sklearn.metrics`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
prueba = data.assign(cluster=km3.labels_)
print(prueba.groupby('cluster')[compras].mean().round(1))
print('Clientes por cluster:', prueba['cluster'].value_counts().sort_index().tolist())
print('Silueta con k=3:', round(silhouette_score(X, km3.labels_), 3))
```

</details>

**Verifica tu resultado:** clientes por cluster `[136, 211, 153]` y silueta **0.401**. El cluster de 211 clientes es el de mayor frecuencia (15.5) e historial (39,304.7).

#### 2.3 Número óptimo de clusters (Paso 5)

Prueba k de 2 a 10 y registra la inercia (método del codo) y el coeficiente de silueta.

```python
resultados = []
for k in range(___, ___):
    modelo = KMeans(n_clusters=___, n_init=10, random_state=42).fit(X)
    resultados.append((k, modelo.___, silhouette_score(X, modelo.labels_)))

res = pd.DataFrame(resultados, columns=['k', 'inercia', 'silueta'])
print(res.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(res['k'], res['inercia'], marker='o')
axes[0].axvline(___, color='red', linestyle='--', label='k elegido')
axes[0].set_title('Método del codo')
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Inercia')
axes[0].legend()
axes[1].plot(res['k'], res['silueta'], marker='o', color='green')
axes[1].set_title('Coeficiente de silueta')
axes[1].set_xlabel('Número de clusters (k)')
axes[1].set_ylabel('Silueta promedio')
plt.tight_layout()
plt.show()
```

<details><summary><b>Pista</b></summary>

* `range` no incluye el límite superior: para llegar a 10 escribe 11.
* En cada vuelta el número de clusters es la variable del ciclo.
* La suma de distancias al centroide se guarda en un atributo que termina en guion bajo.
* Marca con la línea roja el k donde la curva se dobla y la silueta es máxima.

</details>

<details><summary><b>Ver solución</b></summary>

```python
resultados = []
for k in range(2, 11):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    resultados.append((k, modelo.inertia_, silhouette_score(X, modelo.labels_)))

res = pd.DataFrame(resultados, columns=['k', 'inercia', 'silueta'])
print(res.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(res['k'], res['inercia'], marker='o')
axes[0].axvline(4, color='red', linestyle='--', label='k elegido')
axes[0].set_title('Método del codo')
axes[0].set_xlabel('Número de clusters (k)')
axes[0].set_ylabel('Inercia')
axes[0].legend()
axes[1].plot(res['k'], res['silueta'], marker='o', color='green')
axes[1].set_title('Coeficiente de silueta')
axes[1].set_xlabel('Número de clusters (k)')
axes[1].set_ylabel('Silueta promedio')
plt.tight_layout()
plt.show()
```

</details>

**Verifica tu resultado:** la inercia cae de 608.6 (k=2) a 375.1 (k=3) y 235.3 (k=4); después baja mucho más lento. La silueta más alta es **0.427 con k=4**. Ambos criterios apuntan a 4 segmentos.

#### 2.4 Modelo final y nombre de cada segmento (Paso 6)

Entrena el modelo con el k elegido y nombra cada segmento comparando su frecuencia y monto promedio contra la mediana de todos los clientes.

```python
kmeans = KMeans(n_clusters=___, n_init=10, random_state=42).fit(X)
data['Segmento_id'] = kmeans.labels_

perfil = data.groupby('Segmento_id')[columnas_seg].mean()
f_med = data['FrecuenciaCompra'].___()
m_med = data['MontoPromedio'].___()

def nombrar(fila):
    alta_f = fila['FrecuenciaCompra'] >= f_med
    alto_m = fila['MontoPromedio'] >= m_med
    if alta_f and alto_m:
        return ___
    if alta_f and not alto_m:
        return ___
    if not alta_f and alto_m:
        return 'Ocasionales de ticket alto'
    return 'Ocasionales de bajo valor'

data['Segmento'] = data['Segmento_id'].map(perfil.apply(nombrar, axis=___))
data['Segmento'].value_counts()
```

<details><summary><b>Pista</b></summary>

* Usa el k que elegiste en 2.3.
* La referencia es la mediana, no la media.
* Compra mucho y gasta mucho: alto valor. Compra mucho pero con ticket bajo: reacciona a promociones, es decir, sensible al precio.
* Para aplicar la función a cada fila (cada segmento), `axis` vale 1.

</details>

<details><summary><b>Ver solución</b></summary>

```python
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
data['Segmento_id'] = kmeans.labels_

perfil = data.groupby('Segmento_id')[columnas_seg].mean()
f_med = data['FrecuenciaCompra'].median()
m_med = data['MontoPromedio'].median()

def nombrar(fila):
    alta_f = fila['FrecuenciaCompra'] >= f_med
    alto_m = fila['MontoPromedio'] >= m_med
    if alta_f and alto_m:
        return 'Alto valor'
    if alta_f and not alto_m:
        return 'Sensibles al precio'
    if not alta_f and alto_m:
        return 'Ocasionales de ticket alto'
    return 'Ocasionales de bajo valor'

data['Segmento'] = data['Segmento_id'].map(perfil.apply(nombrar, axis=1))
data['Segmento'].value_counts()
```

</details>

**Verifica tu resultado:** Alto valor 137, Ocasionales de ticket alto 134, Sensibles al precio 117 y Ocasionales de bajo valor 112 (suman 500). Las medianas son 10 compras y 2,729 de monto.

#### 2.5 Perfil de cada segmento

Resume cada segmento: tamaño, comportamiento, ingreso total y qué parte de los clientes y del ingreso representa.

```python
seg = data.groupby('Segmento').agg(
    clientes=('Segmento', 'size'),
    frecuencia=('FrecuenciaCompra', ___),
    monto=('MontoPromedio', 'mean'),
    historial=('Historial12Meses', 'mean'),
    ingreso=('Historial12Meses', ___),
    edad=('Edad', 'mean'),
)
seg['% clientes'] = seg['clientes'] / ___ * 100
seg['% ingreso'] = seg['ingreso'] / seg['ingreso'].___() * 100
seg = seg.sort_values(___, ascending=False).round(1)
seg
```

<details><summary><b>Pista</b></summary>

* Cada argumento de `agg` es una pareja (columna, función). La frecuencia se resume con su promedio y el ingreso con su total.
* El total de clientes es el número de filas de `data`.
* Ordena por la columna de ingreso para que el segmento que más vende quede arriba.

</details>

<details><summary><b>Ver solución</b></summary>

```python
seg = data.groupby('Segmento').agg(
    clientes=('Segmento', 'size'),
    frecuencia=('FrecuenciaCompra', 'mean'),
    monto=('MontoPromedio', 'mean'),
    historial=('Historial12Meses', 'mean'),
    ingreso=('Historial12Meses', 'sum'),
    edad=('Edad', 'mean'),
)
seg['% clientes'] = seg['clientes'] / len(data) * 100
seg['% ingreso'] = seg['ingreso'] / seg['ingreso'].sum() * 100
seg = seg.sort_values('ingreso', ascending=False).round(1)
seg
```

</details>

**Verifica tu resultado:** Alto valor: 27.4% de los clientes y **57.0% del ingreso**. Le siguen Ocasionales de ticket alto (26.8% / 20.7%), Sensibles al precio (23.4% / 16.6%) y Ocasionales de bajo valor (22.4% / 5.7%). La edad promedio es parecida en todos (alrededor de 44 años).

#### 2.6 Canal de adquisición por segmento

Calcula qué porcentaje de cada segmento llegó por cada canal.

```python
(pd.___(data['Segmento'], data['Canal'], normalize=___) * 100).round(1)
```

<details><summary><b>Pista</b></summary>

* La tabla cruzada de pandas cuenta combinaciones de dos columnas.
* Para que cada fila sume 100%, normaliza por fila: el valor es el texto `'index'`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
(pd.crosstab(data['Segmento'], data['Canal'], normalize='index') * 100).round(1)
```

</details>

**Verifica tu resultado:** todos los valores quedan entre 29% y 38%: ningún segmento depende de un canal en particular.

#### 2.7 Visualización de los clusters

Grafica a los clientes coloreados por segmento y marca los centroides en la escala original.

```python
colores = {'Alto valor': '#1f77b4', 'Sensibles al precio': '#ff7f0e',
           'Ocasionales de ticket alto': '#2ca02c', 'Ocasionales de bajo valor': '#9467bd'}
fig, ax = plt.subplots(figsize=(7.5, 5))
for nombre, grupo in data.groupby('Segmento'):
    ax.scatter(grupo['FrecuenciaCompra'], grupo['MontoPromedio'], s=16, alpha=0.7,
               color=colores[___], label=f"{nombre} (n={len(grupo)})")

centros = escalador.___(kmeans.cluster_centers_)
ax.scatter(centros[:, 0], centros[:, 1], marker=___, s=160, color='black', label='Centroides')
ax.set_xlabel('Frecuencia de compra (compras al año)')
ax.set_ylabel('Monto promedio por compra')
ax.set_title('Segmentos de clientes (K-Means, k = 4)')
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.show()
```

<details><summary><b>Pista</b></summary>

* El color se busca en el diccionario con el nombre del segmento del ciclo.
* Los centroides están en escala estandarizada. El método que deshace la estandarización empieza con `inverse_`.
* Usa una X mayúscula como marcador, entre comillas.

</details>

<details><summary><b>Ver solución</b></summary>

```python
colores = {'Alto valor': '#1f77b4', 'Sensibles al precio': '#ff7f0e',
           'Ocasionales de ticket alto': '#2ca02c', 'Ocasionales de bajo valor': '#9467bd'}
fig, ax = plt.subplots(figsize=(7.5, 5))
for nombre, grupo in data.groupby('Segmento'):
    ax.scatter(grupo['FrecuenciaCompra'], grupo['MontoPromedio'], s=16, alpha=0.7,
               color=colores[nombre], label=f"{nombre} (n={len(grupo)})")

centros = escalador.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centros[:, 0], centros[:, 1], marker='X', s=160, color='black', label='Centroides')
ax.set_xlabel('Frecuencia de compra (compras al año)')
ax.set_ylabel('Monto promedio por compra')
ax.set_title('Segmentos de clientes (K-Means, k = 4)')
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.show()
```

</details>

**Verifica tu resultado:** cuatro cuadrantes, uno por segmento. Los centroides quedan cerca de (14.8, 3,583), (14.7, 1,239), (5.0, 3,923) y (4.8, 1,366).

#### 2.8 Peso de cada segmento: clientes contra ingresos

Compara en una gráfica de barras qué porcentaje de clientes y qué porcentaje del ingreso aporta cada segmento.

```python
comparativo = seg[[___, ___]]
ax = comparativo.plot(kind=___, figsize=(8, 4), edgecolor='white')
ax.set_ylabel('Porcentaje')
ax.set_xlabel('')
ax.set_title('Peso de cada segmento: clientes vs. ingresos')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()
```

<details><summary><b>Pista</b></summary>

* Selecciona las dos columnas de porcentaje que creaste en 2.5, con sus nombres exactos.
* El tipo de gráfica de barras verticales en pandas es `'bar'`.

</details>

<details><summary><b>Ver solución</b></summary>

```python
comparativo = seg[['% clientes', '% ingreso']]
ax = comparativo.plot(kind='bar', figsize=(8, 4), edgecolor='white')
ax.set_ylabel('Porcentaje')
ax.set_xlabel('')
ax.set_title('Peso de cada segmento: clientes vs. ingresos')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()
```

</details>

**Verifica tu resultado:** en Alto valor la barra de ingreso (57%) es el doble de la de clientes (27%); en Ocasionales de bajo valor pasa lo contrario (22% de clientes, 6% del ingreso).

**Tu interpretación:**

1. ¿Por qué elegiste ese número de segmentos? Usa la inercia y la silueta para justificarlo.
2. Describe cada segmento en lenguaje de negocio: quién es, cuánto compra y cuánto aporta.
3. ¿Qué segmento aporta mayor valor a la empresa? ¿Cuál es el más numeroso pero menos rentable?

_Escribe aquí tu respuesta._

### Parte 3. Personalización

Pasos 7 y 8: estrategia personalizada por segmento e impacto potencial en ingresos.

#### 3.1 Estrategia personalizada por segmento (Paso 7)

Completa la tabla con una acción concreta de cada tipo para cada segmento. Puedes editar esta celda con doble clic.

| Segmento | Descuentos | Recomendaciones automáticas | Programa de lealtad |
|---|---|---|---|
| Alto valor | ___ | ___ | ___ |
| Ocasionales de ticket alto | ___ | ___ | ___ |
| Sensibles al precio | ___ | ___ | ___ |
| Ocasionales de bajo valor | ___ | ___ | ___ |

<details><summary><b>Pista</b></summary>

* **Alto valor:** no necesitan descuento para comprar; un descuento aquí regala margen. Piensa en beneficios exclusivos y en recomendaciones de productos complementarios.
* **Ocasionales de ticket alto:** ya gastan mucho por compra; el reto es que compren más seguido (recordatorios, recompra).
* **Sensibles al precio:** compran seguido pero barato; busca subir el ticket (paquetes, envío gratis a partir de un monto).
* **Ocasionales de bajo valor:** el costo de la acción debe ser bajo (campañas automatizadas, cupón de reactivación).

</details>

**Verifica tu resultado:** cada acción debe responder a la combinación de frecuencia y monto del segmento, no ser la misma para todos.

#### 3.2 Impacto potencial en ingresos (Paso 8)

Estima el ingreso adicional, el costo y el beneficio neto de tus estrategias. Los porcentajes de la tabla de supuestos son un punto de partida: primero completa el cálculo con ellos y luego cámbialos por los tuyos.

```python
nombres = ['Alto valor', 'Sensibles al precio',
           'Ocasionales de ticket alto', 'Ocasionales de bajo valor']
supuestos = pd.DataFrame({
    'uplift_%': [6, 12, 15, 8],   # aumento esperado del ingreso del segmento
    'costo_%':  [2, 4, 3, 2],     # costo de la acción como % del ingreso del segmento
}, index=nombres)

impacto = seg[['ingreso']].join(___)
impacto['adicional'] = impacto['ingreso'] * impacto[___] / 100
impacto['costo'] = impacto['ingreso'] * impacto[___] / 100
impacto['neto'] = impacto[___] - impacto[___]

impacto.loc['Total'] = impacto.___()
total = impacto.loc['Total']
impacto.loc['Total', 'uplift_%'] = total['adicional'] / total['ingreso'] * 100
impacto.loc['Total', 'costo_%'] = total['costo'] / total['ingreso'] * 100
impacto['neto_%'] = impacto['neto'] / impacto['ingreso'] * 100
impacto.round(1)
```

<details><summary><b>Pista</b></summary>

* `join` une por el índice: ambas tablas están indexadas por el nombre del segmento.
* El ingreso adicional sale del porcentaje de aumento; el costo, del porcentaje de costo.
* Neto = lo que se gana de más menos lo que cuesta lograrlo.
* La fila Total suma todas las columnas; después se recalculan los porcentajes porque sumar porcentajes no tiene sentido.

</details>

<details><summary><b>Ver solución</b></summary>

```python
nombres = ['Alto valor', 'Sensibles al precio',
           'Ocasionales de ticket alto', 'Ocasionales de bajo valor']
supuestos = pd.DataFrame({
    'uplift_%': [6, 12, 15, 8],   # aumento esperado del ingreso del segmento
    'costo_%':  [2, 4, 3, 2],     # costo de la acción como % del ingreso del segmento
}, index=nombres)

impacto = seg[['ingreso']].join(supuestos)
impacto['adicional'] = impacto['ingreso'] * impacto['uplift_%'] / 100
impacto['costo'] = impacto['ingreso'] * impacto['costo_%'] / 100
impacto['neto'] = impacto['adicional'] - impacto['costo']

impacto.loc['Total'] = impacto.sum()
total = impacto.loc['Total']
impacto.loc['Total', 'uplift_%'] = total['adicional'] / total['ingreso'] * 100
impacto.loc['Total', 'costo_%'] = total['costo'] / total['ingreso'] * 100
impacto['neto_%'] = impacto['neto'] / impacto['ingreso'] * 100
impacto.round(1)
```

</details>

**Verifica tu resultado:** con los supuestos dados, el ingreso total es 12,827,465.2 y el beneficio neto total 825,047.0 (**6.4%** del ingreso). El mayor neto lo da Ocasionales de ticket alto (318,213.5), por encima de Alto valor (292,561.9), aunque este último vende casi tres veces más.

**Tu interpretación:**

1. ¿Por qué un segmento con menos ingreso puede dar más beneficio neto que el de alto valor?
2. Justifica tus porcentajes de aumento y costo. ¿Cuál es el supuesto más débil?
3. ¿Qué riesgos existen al personalizar excesivamente (privacidad, saturación, canibalización de margen)?

_Escribe aquí tu respuesta._

### Parte 4. Valor comercial

Pasos 9 y 10: valor promedio de pedido, estimación básica de CLV y campaña de fidelización.

#### 4.1 Valor promedio de pedido (Paso 9)

El valor promedio de pedido (AOV) es el ingreso total entre el número total de compras. Calcúlalo para toda la base y para cada segmento.

```python
aov_global = data['Historial12Meses'].___() / data[___].sum()
print(f"AOV global: {aov_global:,.0f}")

compras_seg = data.groupby('Segmento')['FrecuenciaCompra'].___()
aov_seg = seg[___] / compras_seg
aov_seg.round(0).sort_values(ascending=False)
```

<details><summary><b>Pista</b></summary>

* Numerador: todo lo que se vendió. Denominador: todas las compras realizadas (la suma de las frecuencias).
* Por segmento, divide el ingreso de `seg` entre las compras de ese segmento.
* No es lo mismo que el promedio de `MontoPromedio`: el AOV pondera por cuántas veces compra cada cliente.

</details>

<details><summary><b>Ver solución</b></summary>

```python
aov_global = data['Historial12Meses'].sum() / data['FrecuenciaCompra'].sum()
print(f"AOV global: {aov_global:,.0f}")

compras_seg = data.groupby('Segmento')['FrecuenciaCompra'].sum()
aov_seg = seg['ingreso'] / compras_seg
aov_seg.round(0).sort_values(ascending=False)
```

</details>

**Verifica tu resultado:** AOV global de **2,584**. Por segmento: Ocasionales de ticket alto 3,958; Alto valor 3,603; Ocasionales de bajo valor 1,352; Sensibles al precio 1,236.

#### 4.2 Estimación básica de CLV (Paso 9)

CLV básico = AOV x frecuencia anual x margen x vida esperada del cliente. La vida esperada se aproxima como 1 / (1 - retención). Usa un margen de 30% y una retención anual de 80%.

```python
margen = ___
retencion = ___
vida = 1 / (___ - retencion)

clv = pd.DataFrame({'AOV': aov_seg, 'frecuencia': seg['frecuencia']})
clv['CLV por cliente'] = clv['AOV'] * clv[___] * ___ * ___
clv['CLV del segmento'] = clv['CLV por cliente'] * seg[___]
clv = clv.sort_values('CLV por cliente', ascending=False)
print(f"Vida esperada: {vida:.0f} años | Margen: {margen:.0%}")
print(f"CLV promedio por cliente: {clv['CLV del segmento'].sum() / len(data):,.0f}")
clv.round(0)
```

<details><summary><b>Pista</b></summary>

* Escribe los porcentajes como proporciones (30% es 0.30).
* Si el 80% se queda cada año, el 20% se va: la vida esperada es 1 entre esa tasa de salida.
* El CLV del segmento multiplica el CLV por cliente por el número de clientes del segmento.

</details>

<details><summary><b>Ver solución</b></summary>

```python
margen = 0.30
retencion = 0.80
vida = 1 / (1 - retencion)

clv = pd.DataFrame({'AOV': aov_seg, 'frecuencia': seg['frecuencia']})
clv['CLV por cliente'] = clv['AOV'] * clv['frecuencia'] * margen * vida
clv['CLV del segmento'] = clv['CLV por cliente'] * seg['clientes']
clv = clv.sort_values('CLV por cliente', ascending=False)
print(f"Vida esperada: {vida:.0f} años | Margen: {margen:.0%}")
print(f"CLV promedio por cliente: {clv['CLV del segmento'].sum() / len(data):,.0f}")
clv.round(0)
```

</details>

**Verifica tu resultado:** vida esperada de 5 años y CLV promedio por cliente de **38,431**. Alto valor tiene el mayor CLV por cliente y de segmento (cerca de 10,958,101 en total).

#### 4.3 Campaña de fidelización basada en datos (Paso 10)

Diseña una campaña para el segmento que elijas. Completa cada punto con datos de tus resultados. Puedes editar esta celda con doble clic.

* **Segmento objetivo y por qué (cita su % de ingreso y su CLV):** ___
* **Objetivo medible (por ejemplo, subir la frecuencia de X a Y compras al año):** ___
* **Mecánica (puntos, niveles, beneficios, recompra):** ___
* **Canal de contacto (recuerda lo que viste en 2.6):** ___
* **Indicador de éxito y cómo lo medirías (grupo de control):** ___
* **Costo estimado y beneficio esperado (usa tu tabla de 3.2):** ___

<details><summary><b>Pista</b></summary>

Un punto más de retención cambia mucho el CLV: con 80% la vida esperada es 5 años; con 85% sube a 6.7. Calcula cuánto valdría ese cambio en tu segmento para justificar el costo de la campaña.

</details>

**Verifica tu resultado:** la campaña debe tener un objetivo con número, un indicador y una justificación económica que salga de tus tablas.

### Conclusiones

Responde con base en tus resultados. Estas respuestas cierran tu reporte técnico (2 a 3 cuartillas: descripción de segmentos, propuesta estratégica y justificación económica).

**Tu interpretación:**

1. ¿Cuál es el segmento que aporta mayor valor a la empresa?
2. En cuestión de estrategia, ¿es mejor atraer nuevos clientes o fidelizar a los existentes? Usa el CLV para argumentar.
3. ¿La segmentación ayuda en la toma de decisiones? Da un ejemplo concreto de este caso.
4. ¿Cuáles son los riesgos de personalizar excesivamente?

_Escribe aquí tu respuesta._
