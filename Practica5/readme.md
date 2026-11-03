# Autopráctica 5 · Análisis de Redes Sociales, Emociones y Noticias Falsas

**Asignatura:** Ciencia de Datos en el Sector Privado (CDT015) · UNITEC
**Unidad 5:** Casos de aplicación de la ciencia de datos en el sector de las comunicaciones
**Duración estimada:** 2 horas · **Modalidad:** autoguiada (Jupyter Notebook o Google Colab)

---

## Cómo usar esta autopráctica

Esta guía te lleva paso a paso por la Práctica 5 del practicario. En cada parte encontrarás:

- **Qué vas a hacer** y por qué importa en un contexto empresarial.
- **Código** listo para copiar en una celda de tu notebook.
- **Resultados esperados**, ocultos en una sección desplegable. Intenta interpretar tu salida **antes** de abrirla.
- **Preguntas de reflexión** que debes responder en tu reporte técnico.

> **Importante:** el dataset se genera con `np.random.seed(42)`. Si no modificas el script, tus números deben coincidir con los resultados esperados. Si no coinciden, revisa que hayas copiado el script completo y que no hayas ejecutado celdas en otro orden.

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

En Google Colab todas las librerías ya vienen instaladas.

### Estructura sugerida de tu repositorio de entrega

```
practica5/
├── practica5.ipynb
├── datos_comunicaciones.csv
├── figuras/
└── reporte_practica5.pdf
```

---

## Contexto del caso

Una empresa desea monitorear publicaciones y el comportamiento de usuarios en redes sociales para **mejorar la focalización de campañas publicitarias** y **detectar riesgos de desinformación**.

La dirección de análisis digital solicita un modelo básico que permita:

1. Identificar patrones de conducta de los usuarios.
2. Analizar emociones predominantes en publicaciones.
3. Detectar posibles noticias falsas.
4. Segmentar perfiles para campañas de publicidad digital.

### Diccionario de datos

| Variable | Tipo | Descripción |
|---|---|---|
| `id_publicacion` | entero | Identificador único de la publicación |
| `tipo_contenido` | categórica | noticia, publicidad, entretenimiento, politica, tecnologia |
| `emocion` | categórica | Emoción ya etiquetada: positiva, negativa, neutral |
| `segmento_usuario` | categórica | Perfil del autor: joven, adulto, profesional, estudiante |
| `clics` | entero | Número de clics recibidos |
| `compartidos` | entero | Número de veces que se compartió |
| `reacciones` | entero | Número de reacciones |
| `noticia_falsa` | binaria | 0 = real, 1 = falsa (variable objetivo) |

> **Observación para tu reporte:** el caso menciona "texto de la publicación", pero el dataset **no incluye una columna de texto**; la emoción ya viene etiquetada. En la Parte 2 verás cómo se obtendría esa etiqueta a partir del texto con un ejemplo pequeño.

---

## Paso 0 · Generar el dataset

Ejecuta el script oficial de la práctica:

```python
import numpy as np
import pandas as pd

# Semilla para reproducibilidad
np.random.seed(42)

# Número de registros
n = 1000

# Tipos de contenido
tipos_contenido = ["noticia", "publicidad", "entretenimiento", "politica", "tecnologia"]

# Emociones posibles
emociones = ["positiva", "negativa", "neutral"]

# Segmentos de usuario
segmentos = ["joven", "adulto", "profesional", "estudiante"]

# Generación de variables
id_publicacion = np.arange(1, n + 1)
tipo_contenido = np.random.choice(tipos_contenido, n)
emocion = np.random.choice(emociones, n)
segmento_usuario = np.random.choice(segmentos, n)

clics = np.random.poisson(50, n)
compartidos = np.random.poisson(20, n)
reacciones = np.random.poisson(100, n)

# Probabilidad de noticia falsa (mayor en política)
prob_fake = []

for tipo in tipo_contenido:
    if tipo == "politica":
        prob_fake.append(0.35)
    elif tipo == "noticia":
        prob_fake.append(0.20)
    else:
        prob_fake.append(0.10)

prob_fake = np.array(prob_fake)

noticia_falsa = (np.random.rand(n) < prob_fake).astype(int)

# Crear DataFrame
df = pd.DataFrame({
    "id_publicacion": id_publicacion,
    "tipo_contenido": tipo_contenido,
    "emocion": emocion,
    "segmento_usuario": segmento_usuario,
    "clics": clics,
    "compartidos": compartidos,
    "reacciones": reacciones,
    "noticia_falsa": noticia_falsa
})

# Guardar dataset
df.to_csv("datos_comunicaciones.csv", index=False)

print("Dataset generado correctamente: datos_comunicaciones.csv")
print("Total de registros:", len(df))
print(df.head())
```

**Antes de continuar**, lee el script con atención y responde: ¿de qué variables depende realmente `noticia_falsa`? Guarda tu respuesta; la retomaremos en la Parte 3.

---

## Parte 1 · Análisis exploratorio de publicaciones y usuarios

Antes de modelar, una empresa necesita saber **qué contenido genera respuesta** y **quién la genera**. Este diagnóstico orienta dónde invertir en publicidad.

### 1.1 Carga y revisión general

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("datos_comunicaciones.csv")

print("Número de publicaciones:", len(df))
print(df.info())
print(df.describe().round(2))
```

### 1.2 Tipo de contenido y nivel de interacción

Crearemos una métrica resumen de **interacción total** para comparar publicaciones.

```python
df["interaccion"] = df["clics"] + df["compartidos"] + df["reacciones"]

print(df["tipo_contenido"].value_counts())

resumen_tipo = df.groupby("tipo_contenido")[["clics", "compartidos", "reacciones", "interaccion"]].mean().round(2)
print(resumen_tipo)
```

### 1.3 Patrones de conducta

```python
# Publicaciones con mayor respuesta
top = df.sort_values("interaccion", ascending=False).head(10)
print(top[["id_publicacion", "tipo_contenido", "segmento_usuario", "interaccion", "noticia_falsa"]])

# Segmentos de usuario con mayor actividad
print(df.groupby("segmento_usuario")["interaccion"].agg(["count", "mean"]).round(2))

# Correlación entre métricas de interacción
print(df[["clics", "compartidos", "reacciones"]].corr().round(3))
```

> El dataset no tiene un ID de usuario, así que "usuarios con mayor actividad" se aproxima con `segmento_usuario`. Menciona esta limitación en tu reporte.

### 1.4 Gráficas

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.countplot(data=df, x="tipo_contenido", ax=axes[0])
axes[0].set_title("Publicaciones por tipo de contenido")
axes[0].tick_params(axis="x", rotation=30)

sns.boxplot(data=df, x="tipo_contenido", y="clics", ax=axes[1])
axes[1].set_title("Clics por tipo de contenido")
axes[1].tick_params(axis="x", rotation=30)

sns.histplot(data=df, x="interaccion", bins=30, kde=True, ax=axes[2])
axes[2].set_title("Distribución de la interacción total")

plt.tight_layout()
plt.savefig("figuras/parte1_exploratorio.png", dpi=150)  # crea antes la carpeta figuras/
plt.show()
```

<details>
<summary>Resultados esperados de la Parte 1</summary>

- 1,000 publicaciones repartidas de forma casi uniforme: noticia 210, politica 206, tecnologia 204, entretenimiento 190, publicidad 190.
- Promedios generales: ~49.8 clics, ~19.9 compartidos y ~99.8 reacciones; interacción total ≈ 170.
- Los promedios por tipo de contenido son prácticamente iguales (clics entre 49.4 y 50.2).
- Las correlaciones entre clics, compartidos y reacciones son cercanas a 0 (≤ 0.013).
- Las 5 publicaciones con mayor interacción son las IDs 249, 383, 608, 30 y 985 (204 a 215 interacciones), y **ninguna es noticia falsa**.

**Interpretación:** en este dataset la interacción **no depende** del tipo de contenido ni del segmento. Esto ocurre porque el script genera clics, compartidos y reacciones de forma independiente. En datos reales esperarías diferencias claras; aquí la conclusión correcta es reportar que **no hay patrón**, no inventarlo.

</details>

**Reflexión 1:** ¿Qué tipo de contenido tiene más clics? ¿La diferencia es suficientemente grande como para tomar una decisión de negocio?

---

## Parte 2 · Análisis de emociones

Las emociones influyen en cuánto se difunde un contenido. Las marcas buscan asociarse con emociones positivas y vigilar picos de emociones negativas.

### 2.1 ¿Cómo se obtiene una emoción a partir del texto? (ejemplo didáctico)

En un sistema real, la columna `emocion` se calcula a partir del texto. Este ejemplo usa un **léxico** (lista de palabras con carga emocional):

```python
positivas = {"excelente", "feliz", "increíble", "gracias", "bueno", "encanta", "logro"}
negativas = {"terrible", "odio", "fraude", "malo", "indignante", "peligro", "mentira"}

def clasificar_emocion(texto):
    palabras = texto.lower().replace(",", "").replace(".", "").replace("!", "").split()
    puntaje = sum(p in positivas for p in palabras) - sum(p in negativas for p in palabras)
    if puntaje > 0:
        return "positiva"
    elif puntaje < 0:
        return "negativa"
    return "neutral"

ejemplos = [
    "Me encanta el nuevo producto, excelente servicio!",
    "Esto es un fraude, indignante y terrible.",
    "El evento se realizará el lunes a las 10 horas.",
]
for t in ejemplos:
    print(clasificar_emocion(t), "→", t)
```

> Este método es simple y falla con sarcasmo, negaciones ("no es bueno") y jerga. En la industria se usan modelos de lenguaje entrenados para análisis de sentimiento.

### 2.2 Emociones predominantes

```python
print(df["emocion"].value_counts(normalize=True).round(3))

tabla_emocion = pd.crosstab(df["tipo_contenido"], df["emocion"])
print(tabla_emocion)

print(df.groupby("emocion")["interaccion"].mean().round(2))
```

### 2.3 ¿Qué publicaciones generan mayor reacción emocional?

```python
reaccion_emocional = df.pivot_table(index="tipo_contenido", columns="emocion",
                                    values="reacciones", aggfunc="mean").round(1)
print(reaccion_emocional)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
tabla_emocion.plot(kind="bar", ax=axes[0])
axes[0].set_title("Emociones por tipo de contenido")
axes[0].tick_params(axis="x", rotation=30)

sns.heatmap(reaccion_emocional, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1])
axes[1].set_title("Reacciones promedio por tipo y emoción")

plt.tight_layout()
plt.savefig("figuras/parte2_emociones.png", dpi=150)
plt.show()
```

<details>
<summary>Resultados esperados de la Parte 2</summary>

Tabla de emociones por tipo de contenido:

| tipo_contenido | negativa | neutral | positiva |
|---|---|---|---|
| entretenimiento | 64 | 54 | 72 |
| noticia | 56 | 72 | 82 |
| politica | 77 | 64 | 65 |
| publicidad | 62 | 78 | 50 |
| tecnologia | 77 | 57 | 70 |

- **Política** y **tecnología** tienen la mayor cantidad de publicaciones negativas (77 cada una).
- **Noticia** es el tipo con más publicaciones positivas (82).
- **Publicidad** es mayoritariamente neutral (78), lo que es coherente con contenido informativo de marca.
- La interacción promedio por emoción es casi idéntica: negativa 168.6, neutral 169.8, positiva 170.0.

**Interpretación:** las diferencias en la tabla se deben al azar (la emoción se asignó aleatoriamente). En tu reporte puedes describir la distribución, pero no concluyas que una emoción "genera más interacción": los datos no lo respaldan.

</details>

**Reflexión 2:** ¿Qué tipo de contenido generó mayor respuesta emocional? ¿Por qué sería riesgoso para una marca publicar junto a contenido político con carga negativa?

---

## Parte 3 · Detección de noticias falsas

Un clasificador de desinformación protege la reputación de la plataforma y de los anunciantes. Aquí aprenderás que **un accuracy alto puede ser engañoso**.

### 3.1 Preparar los datos

```python
from sklearn.model_selection import train_test_split

X = pd.get_dummies(
    df[["tipo_contenido", "emocion", "segmento_usuario", "clics", "compartidos", "reacciones"]],
    drop_first=True
).astype(float)
y = df["noticia_falsa"]

print("Proporción de noticias falsas:", round(y.mean(), 3))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

> `stratify=y` mantiene la misma proporción de noticias falsas en entrenamiento y prueba. Es indispensable cuando una clase es minoritaria.

### 3.2 Entrenar y evaluar dos modelos

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

modelos = {
    "Regresión logística": LogisticRegression(max_iter=1000),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=3, random_state=42),
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    print(f"\n=== {nombre} ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("Matriz de confusión:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))
```

<details>
<summary>Resultados esperados (modelos sin balancear)</summary>

- Regresión logística: **accuracy 0.825**, matriz `[[165, 0], [35, 0]]`.
- Árbol de decisión: **accuracy 0.825**, matriz `[[164, 1], [34, 1]]`.

**¡Cuidado!** La regresión logística **no detectó ninguna** de las 35 noticias falsas del conjunto de prueba. Su accuracy de 82.5% es exactamente lo que obtendrías diciendo "todo es real", porque el 82.5% de las publicaciones de prueba son reales. El modelo no aprendió nada útil.

</details>

### 3.3 Corregir el desbalance de clases

```python
modelos_bal = {
    "Regresión logística (balanceada)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Árbol de decisión (balanceado)": DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced"),
}

for nombre, modelo in modelos_bal.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    print(f"\n=== {nombre} ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("Matriz de confusión:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))

# Visualizar la matriz de confusión del árbol balanceado
from sklearn.metrics import ConfusionMatrixDisplay
arbol = modelos_bal["Árbol de decisión (balanceado)"]
ConfusionMatrixDisplay.from_estimator(arbol, X_test, y_test,
                                      display_labels=["Real", "Falsa"], cmap="Blues")
plt.title("Árbol de decisión balanceado")
plt.savefig("figuras/parte3_matriz_confusion.png", dpi=150)
plt.show()

# ¿Qué variables usa el árbol?
importancia = pd.Series(arbol.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importancia.head(5).round(3))
```

<details>
<summary>Resultados esperados (modelos balanceados)</summary>

| Modelo | Accuracy | Falsas detectadas (de 35) | Recall (falsas) | Precisión (falsas) |
|---|---|---|---|---|
| Regresión logística balanceada | 0.760 | 24 | 0.686 | 0.393 |
| Árbol de decisión balanceado | 0.780 | 24 | 0.686 | 0.421 |

- El accuracy **bajó**, pero ahora el modelo detecta 24 de 35 noticias falsas. Ese es el modelo útil.
- A cambio, marca como falsas unas 33 a 37 noticias reales (falsos positivos).
- Variable más importante del árbol: `tipo_contenido_politica` (0.606), seguida de `clics` (0.166) y `tipo_contenido_noticia` (0.100).

**Interpretación:** el modelo descubrió lo que estaba en el script: la falsedad depende casi solo del **tipo de contenido** (política 39% falsas, noticia 18%, el resto ~8–12%). Las métricas de interacción no aportan información real; si aparecen en el árbol es por ruido. Compara esto con tu respuesta del Paso 0.

</details>

### 3.4 Riesgos de clasificación incorrecta

| Error | Qué significa | Riesgo para la empresa |
|---|---|---|
| **Falso negativo** | Noticia falsa clasificada como real | La desinformación circula, daño reputacional, anuncios junto a contenido falso, posibles sanciones |
| **Falso positivo** | Noticia real clasificada como falsa | Censura injustificada, pérdida de confianza de usuarios y medios, acusaciones de sesgo |

**Reflexión 3:** Si fueras responsable de la plataforma, ¿preferirías el modelo con accuracy de 0.825 o el de 0.780? Justifica con la matriz de confusión. ¿Es justo que un modelo marque contenido político como sospechoso solo por ser político?

---

## Parte 4 · Segmentación de perfiles y publicidad

Segmentar permite dirigir cada campaña al público con mayor probabilidad de responder, reduciendo el costo por resultado.

### 4.1 Segmentación por nivel de interacción (reglas de negocio)

```python
cortes = df["interaccion"].quantile([0.33, 0.66])
df["nivel_interaccion"] = pd.cut(df["interaccion"],
                                 bins=[-1, cortes.iloc[0], cortes.iloc[1], np.inf],
                                 labels=["baja", "media", "alta"])
print(cortes)
print(df["nivel_interaccion"].value_counts())

# Interés temático: interacción promedio por segmento y tipo de contenido
interes = df.pivot_table(index="segmento_usuario", columns="tipo_contenido",
                         values="interaccion", aggfunc="mean").round(1)
print(interes)

sns.heatmap(interes, annot=True, fmt=".1f", cmap="Greens")
plt.title("Interés temático: interacción promedio")
plt.savefig("figuras/parte4_interes_tematico.png", dpi=150)
plt.show()
```

### 4.2 Segmentación con K-Means (repaso de la Práctica 2)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

Z = StandardScaler().fit_transform(df[["clics", "compartidos", "reacciones"]])

# Método del codo
inercias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).inertia_ for k in range(1, 9)]
plt.plot(range(1, 9), inercias, marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.savefig("figuras/parte4_codo.png", dpi=150)
plt.show()

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
df["cluster"] = km.labels_
print(df.groupby("cluster")[["clics", "compartidos", "reacciones", "interaccion"]].mean().round(1))
print(df["cluster"].value_counts())
```

<details>
<summary>Resultados esperados de la Parte 4</summary>

- Cortes de interacción: percentil 33 = 164 y percentil 66 = 174. Niveles: baja 357, media 316, alta 327.
- En el mapa de interés temático, las combinaciones más altas son **joven–noticia (173.4)**, **adulto–entretenimiento (172.7)** y **joven–política (172.1)**; aun así, todas están entre 165 y 174.
- La curva del codo **desciende de forma suave sin un codo claro** (3000 → 2346 → 1899 → 1552…). Esto indica que no existen grupos naturales bien separados.
- Con k = 3 se obtienen tres clusters casi del mismo tamaño (≈330 cada uno):

| Cluster | Clics | Compartidos | Reacciones | Interacción | Lectura de negocio |
|---|---|---|---|---|---|
| 0 | 52.5 | 16.4 | 105.2 | 174.2 | Reaccionan y dan clic, pero comparten poco |
| 1 | 50.4 | 24.6 | 102.8 | 177.8 | **Difusores**: alto nivel de compartidos |
| 2 | 46.3 | 18.7 | 91.2 | 156.2 | Baja interacción |

</details>

### 4.3 Estrategia de focalización (mínimo dos segmentos)

Usa tus resultados para completar una tabla como esta en tu reporte:

| Segmento | Objetivo de campaña | Formato y mensaje | Métrica de éxito |
|---|---|---|---|
| Difusores (cluster 1) | Alcance orgánico | Contenido fácil de compartir, retos, programas de referidos | Compartidos por publicación |
| Baja interacción (cluster 2) | Reactivación | Contenido breve, incentivos, mejores horarios de publicación | Tasa de clics (CTR) |

**Reflexión 4:** ¿Cómo mejora la segmentación una campaña digital? ¿Qué pasaría si la empresa gasta lo mismo en todos los segmentos cuando la diferencia entre ellos es tan pequeña?

---

## Parte 5 · Prevención de abusos y fraudes

Las plataformas detectan bots, granjas de clics y campañas coordinadas de desinformación buscando **comportamientos anormales**.

### 5.1 Detección de patrones anormales de actividad

```python
from sklearn.ensemble import IsolationForest

# Método 1: puntuación z sobre compartidos (difusión inusual)
df["z_compartidos"] = (df["compartidos"] - df["compartidos"].mean()) / df["compartidos"].std()
sospechosas_z = df[df["z_compartidos"] > 2.5]
print("Publicaciones con difusión inusual (z > 2.5):", len(sospechosas_z))

# Método 2: Isolation Forest (anomalías multivariadas)
iso = IsolationForest(contamination=0.02, random_state=42)
df["anomalia"] = iso.fit_predict(df[["clics", "compartidos", "reacciones"]]) == -1
print("Anomalías detectadas:", df["anomalia"].sum())
print("Proporción de falsas entre anomalías:", round(df.loc[df["anomalia"], "noticia_falsa"].mean(), 3))

sns.scatterplot(data=df, x="clics", y="compartidos", hue="anomalia", palette={False: "lightgray", True: "red"})
plt.title("Publicaciones con actividad anormal")
plt.savefig("figuras/parte5_anomalias.png", dpi=150)
plt.show()
```

<details>
<summary>Resultados esperados de la Parte 5</summary>

- 5 publicaciones con z > 2.5 en compartidos.
- Isolation Forest marca 20 anomalías (el 2% solicitado); el 20% de ellas son falsas, una proporción similar al promedio general (17.3%).

**Interpretación:** en este dataset la actividad anormal **no** se asocia con desinformación, porque el script no simuló bots ni campañas coordinadas. En una plataforma real, un pico repentino de compartidos en pocos minutos sí es una señal clásica de difusión artificial.

</details>

### 5.2 Propuesta de monitoreo preventivo

Redacta una propuesta breve (media cuartilla) que cubra estos elementos:

1. **Señales a vigilar:** picos de compartidos por minuto, cuentas nuevas con actividad muy alta, mismo texto publicado por muchas cuentas, contenido político con baja verificación.
2. **Flujo de decisión:** modelo de clasificación + detector de anomalías → cola de revisión humana → acción (etiquetar, reducir alcance o retirar).
3. **Umbrales:** qué tan sensible debe ser el sistema y quién decide cuando el modelo duda.
4. **Gobernanza:** revisión periódica de sesgos, transparencia con los usuarios, mecanismo de apelación y protección de datos personales.

**Reflexión 5:** ¿Qué ventajas ofrece la ciencia de datos para prevenir abusos? ¿Por qué no conviene que un modelo retire contenido de forma automática sin revisión humana?

---

## Lista de verificación de entrega

- [ ] Notebook con todo el código ejecutado y sin errores
- [ ] Gráficas del análisis exploratorio (carpeta `figuras/`)
- [ ] Tabla y gráfica comparativa del análisis de emociones
- [ ] Matriz de confusión del modelo de clasificación
- [ ] Reporte técnico de 2 a 3 cuartillas con:
  - [ ] Interpretación de resultados
  - [ ] Análisis de patrones
  - [ ] Propuesta de segmentación y focalización publicitaria
  - [ ] Detección de noticias falsas y riesgos de clasificación
  - [ ] Propuesta de monitoreo preventivo
  - [ ] Respuestas a las preguntas de conclusión

## Preguntas de conclusión

1. ¿Qué patrones de conducta fueron más relevantes en el análisis?
2. ¿Qué tipo de contenido generó mayor respuesta emocional?
3. ¿Cuáles son los riesgos de clasificar incorrectamente una noticia falsa?
4. ¿Cómo puede la segmentación de perfiles mejorar una estrategia publicitaria?
5. ¿Qué ventajas ofrece la ciencia de datos para prevenir abusos en plataformas digitales?

## Criterios de evaluación

| Criterio | Qué se revisa |
|---|---|
| Análisis e interpretación de los datos | Estadísticas correctas y conclusiones respaldadas por los números |
| Patrones de conducta | Identificación honesta de patrones (o de su ausencia) |
| Análisis de emociones | Tabla, gráfica e interpretación |
| Modelo de clasificación | Implementación funcional y evaluación más allá del accuracy |
| Segmentación publicitaria | Propuesta coherente con los segmentos encontrados |
| Reporte técnico y conclusiones | Claridad, estructura y lenguaje de informe empresarial |

## Bibliografía

- Anderson, D. (2018). *Estadística descriptiva aplicada*. PATRIA.
- Davenport, T. H., & Harris, J. G. (2017). *Compitiendo con analítica: El nuevo enfoque para ganar en los negocios*. Harvard Business Review Press.
- Provost, F., & Fawcett, T. (2013). *Ciencia de datos para los negocios: Lo que necesita saber sobre minería de datos y pensamiento analítico de datos*. O'Reilly Media.
