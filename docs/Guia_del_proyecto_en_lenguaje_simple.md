# Qué hace cada documento y para qué nos sirve

Esta guía está pensada para entender el trabajo sin tener que seguir el código línea por línea. Recorre qué recibe cada notebook, qué transformaciones realiza y qué resultados entrega.

Describe el contenido revisado del repositorio al 18 de noviembre de 2026. Los resultados visibles provienen de ejecuciones guardadas en los notebooks; escribir esta guía no implicó volver a entrenarlos ni realizar operaciones.

## 1. Qué estamos intentando hacer

Queremos estudiar si podemos estimar mejor la probabilidad de que un contrato de Bitcoin de 15 minutos termine en **YES**, y si esa estimación permite identificar compras que resulten convenientes.

Son dos preguntas relacionadas:

1. **¿Podemos estimar bien lo que va a pasar?**
2. **¿El precio al que podríamos comprar hace que esa predicción sea aprovechable?**

Acertar el resultado no alcanza por sí solo. También importa cuánto pagamos por el contrato.

### Un ejemplo para tener presente durante la lectura

En las simulaciones del proyecto, comprar una unidad YES significa que recibiremos 1 dólar si termina en YES y 0 si termina en NO.

Imaginemos que el precio de compra es 0,60 dólares y que nuestro modelo estima una probabilidad de YES del 70 %. El modelo sugiere que la compra podría tener una ventaja: estima una probabilidad de cobro mayor que el precio pagado, antes de costos.

Pero esa estimación puede estar equivocada. Aunque estuviera bien estimada, una operación particular puede perder: si termina en NO, perdemos los 0,60 dólares pagados. Por eso el trabajo estudia muchas decisiones históricas y separa períodos para aprender de períodos para evaluar.

El recorrido del repositorio es el siguiente:

| Etapa | Pregunta que responde |
|---|---|
| Presentación | ¿Qué queremos investigar y por qué? |
| Notebook 01 | ¿Podemos reunir los datos necesarios? |
| Notebook 02 | ¿Podemos convertirlos en ejemplos ordenados y utilizables? |
| Notebook 03 | ¿Qué probabilidades estima un modelo de árboles y qué ocurre al usarlas para simular compras? |
| Notebook 04 | ¿Mejora algo si el modelo observa una secuencia y filtramos algunas predicciones? |
| Copy1 y Copy2 | ¿Los resultados cambian cuando repetimos el entrenamiento con otra aleatoriedad? |

## 2. La presentación: la propuesta del trabajo

**Documento:** [Presentación intermedia](../Presentacion%20MIA302-1.pptx.pdf).

### Qué contiene

Explica la pregunta del proyecto, las fuentes de información y la forma general de investigarla. La idea es contrastar lo que refleja Kalshi con el índice BRTI y con información de Polymarket.

Kalshi es el mercado de los contratos que queremos estudiar. BRTI es el índice de referencia relacionado con su liquidación. Polymarket aparece como otra fuente que podría aportar información que Kalshi todavía no refleje.

La presentación también muestra exploraciones iniciales: relación entre precios y distancia al nivel de referencia, movimientos del índice y diferencias entre períodos de mayor o menor volatilidad.

### Para qué nos sirve

Sirve para entender el propósito del equipo antes de abrir el código: qué hipótesis se quiere probar y qué significa obtener una mejora.

Hay que leerla como una presentación intermedia. El código actual ya incorpora modelos que allí figuraban como próximos pasos y trabaja con un período más largo. A su vez, **los modelos actuales todavía no usan Polymarket como variable de entrada**.

## 3. Notebook 01: reunir la materia prima

**Archivo:** [01 — Descarga histórica](../01_BTC15m_historico_completo_corregido.ipynb).

### Qué entra

Entra información histórica de Kalshi y BRTI, junto con archivos de descargas anteriores que ya puedan estar guardados.

De Kalshi interesa conocer los contratos, sus horarios, los precios que tuvieron y su resultado final. De BRTI interesa conocer cómo fue cambiando el índice durante ese tiempo.

### Qué pasa en el medio

Primero se define el período a estudiar. Después se buscan los contratos que cerraron dentro de ese período, tanto en el catálogo reciente como en el histórico. Esto importa porque los contratos antiguos pueden estar archivados y no aparecer en una consulta de los más recientes.

Para cada contrato se recuperan velas de un minuto: resúmenes de cómo se comportaron sus precios durante ese intervalo. La configuración actual usa estas velas, no una descarga de cada operación individual.

En paralelo se reúne el histórico de BRTI, dividido en archivos por hora. Dividirlo permite retomar una descarga sin empezar todo de nuevo.

El notebook reutiliza archivos previos, intenta recuperarse de ciertas fallas y registra qué información pudo obtener. También revisa cobertura: que exista un archivo no significa que tenga todos los datos esperados.

### Qué sale

Salen los datos originales organizados en carpetas y reportes que indican contratos encontrados, errores y períodos incompletos.

Todavía no sale una predicción ni una señal de compra. Sale la **materia prima** que necesitan los siguientes notebooks.

### Por qué nos sirve

Para investigar una decisión histórica necesitamos reconstruir qué se sabía entonces y qué terminó ocurriendo. Sin esos datos, no podemos entrenar ni evaluar el modelo de manera razonable.

También necesitamos detectar huecos. Si faltan precios en un momento importante, podríamos interpretar mal una situación o atribuirle al modelo una ventaja que proviene de datos incompletos.

### Dónde aparece Polymarket

Este notebook tiene referencias a Polymarket y revisa un registro de archivos descargados previamente. Sus salidas muestran que esos archivos existían en el entorno de una ejecución anterior.

Sin embargo, **el código actual de este notebook no contiene el proceso completo para descargarlos**, y los modelos posteriores no los incorporan. Hay evidencia de trabajo previo con esa fuente, pero falta integrarla al recorrido actual de modelado.

## 4. Notebook 02: transformar los datos en ejemplos para aprender

**Archivo:** [02 — Dataset y análisis exploratorio](../02_BTC15m_walkforward_corregido.ipynb).

### Qué entra

Entran los archivos de contratos y precios de Kalshi, más el histórico BRTI que preparó el notebook 01.

Son datos de distintas fuentes y frecuencias. Antes de entrenar un modelo, hay que ponerlos en una estructura común.

### Qué pasa en el medio

**Primero construye una tabla de situaciones.** Cada fila representa un contrato en un momento determinado. Un mismo contrato aparece varias veces, porque su situación cambia minuto a minuto.

Por ejemplo, una fila puede describirlo cuando faltan diez minutos y otra cuando faltan cinco. Ambas comparten el resultado final, pero contienen información distinta sobre lo que se sabía en cada momento.

**Después agrega información que ayude a interpretar esa situación:**

- Cuánto falta para el vencimiento.
- Qué tan lejos está BRTI del **strike**, el nivel de referencia del contrato.
- Cuánto se movió BRTI recientemente.
- Si el mercado está teniendo movimientos pequeños o grandes: su **volatilidad**.
- Qué precio y qué referencias de compra y venta muestra Kalshi.

Para la volatilidad utiliza un cálculo que da más importancia a los movimientos recientes. Así representa que el mercado puede estar tranquilo durante un período y agitado durante otro.

**Luego alinea los tiempos.** Una fila que representa una decisión no debería incluir información que recién apareció después. Por eso utiliza datos anteriores o contemporáneos bajo las convenciones del notebook y aplica controles sobre las ventanas de BRTI.

**También filtra problemas de calidad:** datos faltantes, ventanas incompletas o precios fuera de rango. Conserva el resultado YES/NO como la respuesta que el modelo debe aprender, separado de las variables que podrá usar para predecir.

Finalmente produce gráficos para explorar retornos, volatilidad y relación entre los precios de Kalshi y los resultados observados.

### Qué sale

Sale una tabla preparada para análisis y entrenamiento, llamada **dataset**. Se guarda una versión completa y otra restringida a filas que cumplen los controles principales.

También salen gráficos y reportes de calidad que permiten revisar qué se conservó y qué faltó.

### Por qué nos sirve

Este notebook convierte archivos sueltos en ejemplos del tipo:

> «En este momento faltaban cinco minutos, el índice estaba por encima de la referencia y la volatilidad era alta. Después, el contrato terminó en YES».

Al reunir muchos ejemplos, podemos investigar qué relaciones existen entre esa información y el resultado final.

La alineación temporal es esencial: conocer la respuesta después nos sirve para evaluar, pero no podemos dársela al modelo como si ya la hubiera conocido al decidir.

### Qué significan sus gráficos

Ayudan a entender los datos y a detectar anomalías. Por ejemplo, muestran si los contratos de precio alto terminan en YES más frecuentemente que los de precio bajo.

Los grupos de volatilidad ayudan a describir diferentes condiciones de mercado. En los modelos actuales se utiliza la volatilidad numérica; **la etiqueta del grupo no se usa como predictor**.

## 5. Notebook 03: probar un modelo de árboles

**Archivo:** [03 — XGBoost](../03_BTC15m_XGBoost_walkforward_final_octubre.ipynb).

### Qué entra

Entra la tabla preparada por el notebook 02: situaciones de los contratos, variables de mercado y resultados históricos.

### Qué pasa en el medio

**Primero divide los datos respetando el tiempo.** Usa un período para aprender, otro posterior para elegir configuraciones y un tercero para evaluar. Después avanza un mes y repite el procedimiento con más historia disponible.

Por ejemplo, puede aprender con datos hasta marzo, ajustar decisiones con abril y evaluar en mayo. Luego incorpora más datos y evalúa junio. Este avance mes a mes se llama **walk-forward**.

**Después entrena XGBoost**, un modelo que combina árboles de decisión. Aprende relaciones entre las características de una situación y la frecuencia con que termina en YES. Su salida es una probabilidad estimada.

Compara dos formas de alimentarlo:

- **A:** información de BRTI y tiempo restante.
- **B:** la misma información, sumando precio y spread de Kalshi.

Esto permite estudiar qué cambia cuando le mostramos también lo que ya está reflejando el mercado de los contratos.

**Luego transforma probabilidades en decisiones simuladas.** Recorre cada contrato y busca un momento en que la probabilidad estimada supere suficientemente el precio de compra de referencia. También limita en qué parte de los quince minutos puede entrar.

Los umbrales y momentos permitidos se eligen con validación. Después se aplican al período de evaluación sin cambiarlos con sus resultados.

La simulación admite **como máximo una compra YES por contrato** y la mantiene hasta liquidación. No compra NO ni intenta vender antes del vencimiento.

### Qué sale

Salen probabilidades estimadas, medidas de calidad predictiva, una lista de compras simuladas y gráficos de su ganancia o pérdida acumulada.

También entrena modelos finales con los datos disponibles hasta octubre y guarda archivos para poder reutilizarlos con datos posteriores.

### Por qué nos sirve

Es el primer paso que conecta de forma directa **aprender de los datos** con **evaluar una regla de entrada**.

Permite preguntar si una estimación mejor lleva efectivamente a seleccionar compras mejores. Son cosas diferentes: el modelo puede mejorar al predecir y aun así perder en la simulación porque los precios de entrada no resultaron convenientes.

## 6. Notebook 04: mirar la evolución reciente con una LSTM

**Archivo:** [04 — LSTM y MC Dropout](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre.ipynb).

### Qué entra

Entra el mismo dataset. La diferencia principal está en cómo se le presenta al modelo.

XGBoost recibe una descripción de la situación actual. La LSTM recibe una **secuencia de hasta diez observaciones consecutivas del mismo contrato**, terminando en el momento de decisión.

### Qué pasa en el medio

**Primero construye esas secuencias.** Se asegura de que pertenezcan al mismo contrato y de que no incluyan minutos posteriores a la decisión. Si falta historia continua, utiliza una secuencia más corta.

**Después entrena una LSTM**, una red diseñada para procesar secuencias. La idea es que pueda aprovechar cómo fueron cambiando las variables, además de sus valores actuales.

Por ejemplo, dos contratos podrían tener una distancia similar al strike, pero haber llegado a ella de maneras diferentes: uno tras una subida sostenida y otro después de una caída. El modelo intenta aprender si esa evolución reciente aporta información.

Como en XGBoost, compara los conjuntos A y B y repite la evaluación mes a mes.

**Luego agrega una segunda forma de decidir, llamada Monte Carlo Dropout.**

### Qué intenta hacer MC Dropout, sin entrar en la matemática

Con la red ya entrenada, repite cien veces la predicción introduciendo pequeñas variaciones internas mediante dropout. Los pesos aprendidos permanecen iguales; cambia qué unidades participan en cada pasada.

En vez de obtener una sola estimación, obtiene un conjunto de probabilidades algo diferentes. Eso permite observar cuánto varía la respuesta bajo ese procedimiento.

Para comprar, la versión MC utiliza un valor de la parte baja de esas estimaciones. Busca que la diferencia con el precio siga siendo favorable incluso bajo una lectura más conservadora.

Ejemplo: la predicción normal es 65 %, pero ese valor conservador es 57 %. Si el precio de compra es 60 centavos, la predicción normal podría aceptar la entrada y el filtro MC podría rechazarla.

Esto **no significa que el filtro conozca la incertidumbre real con exactitud**. Tampoco significa que un intervalo del 90 % garantice un 90 % de probabilidad de ganar. Es una forma concreta de filtrar señales que hay que evaluar con datos.

### Qué sale

Salen dos conjuntos de resultados para cada grupo de variables:

- Compras seleccionadas usando la predicción normal.
- Compras seleccionadas usando el filtro MC.

Se comparan cantidad de operaciones, ganancias o pérdidas, resultado medio y caídas de la curva acumulada. También se guardan los modelos finales y la información necesaria para volver a utilizarlos.

### Por qué nos sirve

Permite estudiar dos ideas: si la historia reciente ayuda a predecir y si un filtro más conservador ayuda a elegir entradas.

Importa mirar tanto el resultado total como cuántas veces opera. Un filtro puede cambiar la ganancia porque evita ciertas compras y reduce mucho la actividad. Además, el código actual elige por separado algunos parámetros de entrada para cada versión: la diferencia observada no proviene únicamente del filtro.

## 7. Copy1 y Copy2: repetir la prueba con otra aleatoriedad

**Archivos:** [Copy1](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre-Copy1.ipynb) y [Copy2](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre-Copy2.ipynb).

### Qué entra y qué cambia

Usan los mismos datos y el mismo procedimiento que el notebook LSTM principal. La diferencia en el código es la **semilla aleatoria**: 42 en el principal, 4223 en Copy1 y 1234 en Copy2.

La semilla es una configuración que organiza los sorteos internos del entrenamiento. Puede cambiar los pesos iniciales y el orden de los ejemplos, entre otras decisiones aleatorias. Eso puede llevar a modelos y operaciones diferentes, aunque el método sea el mismo.

### Qué sale y por qué nos sirve

Sale otra corrida del experimento, con sus propias predicciones y resultados. Compararlas ayuda a ver si el comportamiento depende mucho de cómo empezó el entrenamiento.

Los resultados guardados efectivamente cambian entre semillas. Eso invita a estudiar su estabilidad, en lugar de quedarse solamente con la corrida que más ganó.

Hay un detalle práctico: **las tres versiones guardan resultados externos con las mismas rutas y nombres**. Si se ejecutan en la misma carpeta, una puede reemplazar los archivos de otra. Las tablas que quedaron dentro de cada notebook sí permiten distinguir sus resultados guardados.

## 8. Los documentos de orientación

El [README](../README.md) es la puerta de entrada: presenta el proyecto y enlaza los archivos.

La [guía técnica](Guia_codigo_BTC15m.md) sirve cuando queremos localizar cómo está implementado un paso, revisar una fórmula o encontrar una celda concreta.

Esta segunda guía sirve para seguir el recorrido general y comprender la función de cada etapa antes de entrar en esos detalles.

## 9. Qué tenemos hoy y qué falta conectar

El recorrido actual permite pasar de datos históricos a ejemplos organizados, entrenar modelos y simular compras YES. Ya podemos estudiar si ciertas variables ayudan a predecir y qué habría ocurrido al aplicar determinadas reglas de entrada.

Hay tres puntos que conviene tener presentes al leer los resultados:

**Polymarket todavía no está integrado a los modelos.** La hipótesis de aprovecharlo como fuente adicional sigue siendo una parte por conectar y evaluar en este flujo.

**Las ganancias y pérdidas mostradas son simuladas.** Se usan precios históricos de referencia, con costos configurados en cero y sin comprobar que cada compra realmente hubiera podido ejecutarse a ese precio. No hay un sistema que envíe órdenes ni una cuenta real operada por estos notebooks.

**El repositorio contiene el código y las salidas guardadas, pero no los datos y modelos exportados.** Para repetir el trabajo hace falta disponer de esos archivos o volver a generarlos. Que un notebook muestre que guardó un modelo en una ejecución previa no significa que ese archivo esté incluido en GitHub.

La conexión que buscamos comprobar a lo largo del trabajo es:

> Tener información útil → estimar probabilidades → compararlas con precios → seleccionar entradas → evaluar resultados en períodos posteriores.

Cada notebook se ocupa de una parte. Entender qué produce uno permite entender qué necesita el siguiente.

---

**Base de esta explicación:** lectura de los seis notebooks, sus salidas guardadas y la presentación, documentada con referencias precisas en la [guía técnica](Guia_codigo_BTC15m.md). No se modificó la lógica de los modelos ni se ejecutaron nuevamente los experimentos.
