# Guía de lectura del código: BTC15m, Kalshi y modelos predictivos

**Fecha de revisión:** 18 de noviembre de 2026.

**Código revisado:** commit `fd9c44f` del repositorio [Bichogian/ML2](https://github.com/Bichogian/ML2).

Esta guía explica qué hace cada notebook, en qué orden se ejecutan sus bloques y cómo se conecta cada paso con la búsqueda de oportunidades en contratos BTC15m. Es una lectura técnica para ponerse al día; el informe integral del trabajo y de sus avances se preparará después.

Para comenzar por una explicación de los procesos, sus entradas y sus salidas, leer la [guía en lenguaje simple](Guia_del_proyecto_en_lenguaje_simple.md).

La revisión se hizo leyendo todas las celdas, comparando las tres versiones LSTM y consultando las salidas ya guardadas. **No se volvieron a descargar datos, entrenar modelos ni enviar órdenes.** Los resultados transcritos son evidencia guardada en los notebooks, no una reproducción independiente.

## Índice

1. [La pregunta del proyecto y qué está implementado](#1-la-pregunta-del-proyecto-y-qué-está-implementado)
2. [Cómo encontrar las líneas citadas](#2-cómo-encontrar-las-líneas-citadas)
3. [Notebook 01: conseguir y auditar los datos](#3-notebook-01-conseguir-y-auditar-los-datos)
4. [Notebook 02: construir el dataset](#4-notebook-02-construir-el-dataset)
5. [Notebook 03: entrenar XGBoost y simular decisiones](#5-notebook-03-entrenar-xgboost-y-simular-decisiones)
6. [Notebook 04: aprender secuencias con LSTM](#6-notebook-04-aprender-secuencias-con-lstm)
7. [Qué cambia en Copy1 y Copy2](#7-qué-cambia-en-copy1-y-copy2)
8. [Qué muestran las salidas guardadas](#8-qué-muestran-las-salidas-guardadas)
9. [Diferencias y puntos a revisar antes de sacar conclusiones](#9-diferencias-y-puntos-a-revisar-antes-de-sacar-conclusiones)
10. [Recorrido sugerido para estudiar el código](#10-recorrido-sugerido-para-estudiar-el-código)

## 1. La pregunta del proyecto y qué está implementado

### 1.1. Qué se quiere estimar

La [presentación](../Presentacion%20MIA302-1.pptx.pdf), páginas 1–4, plantea estimar la probabilidad de que un contrato de BTC de 15 minutos liquide **YES**, usando Kalshi, BRTI y eventualmente Polymarket.

El objetivo predictivo puede escribirse así:

```text
p_modelo = probabilidad estimada de que el contrato termine en YES
          usando la información disponible al decidir
```

El objetivo económico agrega otra pregunta: **¿esa probabilidad justifica el precio al que se podría comprar?**

En el simulador de estos notebooks, una unidad YES paga `1` si el resultado es YES y `0` si es NO. Si se compra a un ask de `0,55` y se estima una probabilidad de `0,65`, la diferencia estimada es `0,10` antes de costos. Esa diferencia es el **edge**.

La estimación puede estar equivocada. Por eso se separan entrenamiento, validación y test y se analiza lo que ocurre fuera de muestra.

### 1.2. Lo que contiene el repositorio actual

| Archivo | Trabajo principal | Aporte al objetivo |
|---|---|---|
| [01 — Descarga](../01_BTC15m_historico_completo_corregido.ipynb) | Descargar contratos, velas y BRTI; auditar cobertura. | Obtener los datos necesarios para estudiar decisiones históricas. |
| [02 — Dataset](../02_BTC15m_walkforward_corregido.ipynb) | Alinear fuentes, construir variables, etiquetas y controles. | Representar qué se sabía en cada decisión y qué terminó pasando. |
| [03 — XGBoost](../03_BTC15m_XGBoost_walkforward_final_octubre.ipynb) | Predecir YES con árboles y evaluar una regla de compra. | Investigar si las probabilidades permiten seleccionar entradas favorables. |
| [04 — LSTM](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre.ipynb) | Usar secuencias y comparar entradas con/sin filtro MC Dropout. | Investigar si la evolución reciente y un filtro de dispersión mejoran la selección. |
| [04 — Copy1](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre-Copy1.ipynb) | Mismo procedimiento LSTM con otra semilla. | Observar sensibilidad al entrenamiento aleatorio. |
| [04 — Copy2](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre-Copy2.ipynb) | Mismo procedimiento LSTM con una tercera semilla. | Observar esa sensibilidad en otra corrida. |

**El flujo actual es de aprendizaje supervisado y backtesting histórico.** No implementa el RL de EarnHFT, un router ni un Q-teacher.

En los modelos hay dos conjuntos de variables:

- **A_BRTI:** tiempo restante, distancia al strike, retorno BRTI y volatilidad.
- **B_BRTI_Kalshi:** lo anterior más precio proxy YES y spread de Kalshi.

Polymarket aparece en constantes y en una auditoría de archivos previos del notebook 01. **Estos seis notebooks no contienen su flujo de descarga completo ni lo incorporan como predictor en 02, 03 o 04.** Las salidas guardadas indican que había archivos de Polymarket en el entorno original; esos archivos no están en el commit revisado.

### 1.3. Flujo completo

```text
Kalshi: contratos, velas y resultado final     BRTI: observaciones del índice
                      \                       /
                       Notebook 01: data/raw/
                                 |
                  Notebook 02: una fila por contrato/minuto
                                 |
                  data/processed/dataset_completo
                         /                     \
             Notebook 03: XGBoost        Notebook 04: LSTM
                         \                     /
                  Probabilidad estimada de YES
                                 |
                 Comparación con ask + regla de entrada
                                 |
                Operaciones simuladas hasta liquidación
```

## 2. Cómo encontrar las líneas citadas

Las referencias usan **número de celda y línea dentro de esa celda**, no líneas del JSON interno del `.ipynb`.

Ejemplo: **N03, C4, L22–28** significa notebook 03, cuarta celda del archivo, líneas 22 a 28 de su código.

- Las celdas se cuentan desde 1 e incluyen Markdown y código.
- Las líneas se cuentan desde 1 e incluyen comentarios y líneas vacías.
- El número de ejecución `In [n]` puede ser diferente y no sirve para esta referencia.
- Las tablas incluyen nombres de funciones y expresiones para encontrarlas con el buscador.

Los rangos corresponden al commit indicado al principio. Si se editan los notebooks, sus posiciones pueden cambiar.

## 3. Notebook 01: conseguir y auditar los datos

**Archivo:** [01_BTC15m_historico_completo_corregido.ipynb](../01_BTC15m_historico_completo_corregido.ipynb).

### 3.1. Qué entra y qué sale

Entran respuestas de APIs y archivos ya descargados que puedan reutilizarse. Salen JSON originales y comprimidos en `data/raw/`, junto con reportes de cobertura. No se entrena un modelo en este notebook.

El período configurado es **1 de diciembre de 2025 a 1 de noviembre de 2026**, con fin excluido y timestamps UTC. Son once meses solicitados; no todos tienen necesariamente contratos disponibles desde el primer día.

### 3.2. Paso a paso

| Dónde | Qué se ejecuta | Para qué sirve |
|---|---|---|
| **C2, L1–14** | Imports y creación de `data/raw`. | Preparar librerías y carpeta de datos originales. |
| **C2, L16–23** | `START`, `END`, serie `KXBTC15M`, límites y flags. | Definir qué contratos buscar. Se descargan velas; los trades individuales están desactivados. |
| **C2, L25–33** | Dos workers, reintentos, timeout y fechas Unix. | Regular concurrencia y expresar el período en el formato de consulta. |
| **C4, L2–32** | `http_session` y `get_json`. | Hacer peticiones, manejar fallas transitorias y esperar ante límites de frecuencia. |
| **C4, L34–53** | `save_json` y `read_cached`. | Guardar mediante reemplazo de un temporal y reutilizar archivos válidos. |
| **C4, L55–69** | `paginated`. | Recorrer páginas con cursor sin aceptar silenciosamente un catálogo truncado por el límite. |
| **C4, L71–73** | URLs de Kalshi y Polymarket. | Definir endpoints base; definir una URL no implica que se descargue esa fuente. |
| **C6, L1–17** | Cutoff y catálogos reciente e histórico de Kalshi. | Buscar contratos aunque hayan pasado al archivo histórico. |
| **C6, L19–48** | Parseo de cierre, filtro por período y deduplicación por ticker. | Construir la lista relevante y detenerse si se intentaría recortarla. |
| **C6, L50–67** | Resumen mensual y `markets.json`. | Documentar contratos encontrados y completitud de paginación. |
| **C8, L3–17** | Cutoffs y ventana de cada contrato. | Elegir cómo consultar sus datos; la ventana nominal usa los 15 minutos anteriores al cierre. |
| **C8, L19–50** | Rama opcional de trades. | Descargar operaciones individuales si se activa el flag; está desactivada en la configuración actual. |
| **C8, L52–85** | Rama de candlesticks. | Reutilizar velas o descargarlas del endpoint reciente/histórico correspondiente. |
| **C8, L88–112** | `ThreadPoolExecutor`, manifest y errores. | Procesar contratos y dejar trazabilidad de fallas y archivos vacíos. |
| **C11, L7–16** | Configuración BRTI y variables de entorno. | Acceder al histórico autenticado sin escribir la clave privada dentro del notebook. |
| **C11, L18–44** | Firma de solicitudes y lectura de horas cacheadas. | Autenticar pedidos y validar archivos reutilizables. |
| **C11, L47–78** | Inicio de `download_brti`, cálculo de horas y caché. | Dividir el período real en horas; `max_hours=0` solicita todo el período. |
| **C11, L79–128** | Descarga, ordenamiento, unión con caché y escritura gzip. | Conservar ticks originales sin reducir cobertura de un archivo previo. |
| **C11, L129–157** | Reintentos y `download_report.json`. | Registrar horas fallidas, incompletas y ticks recuperados. |
| **C11, L163–166** | Llamada condicional a descarga. | Descargar BRTI solo cuando están definidas las credenciales necesarias. |
| **C13, L1–6** | Inventario de archivos. | Ver qué existe en la carpeta; no certifica calidad ni cobertura. |
| **C14, L6–32** | Auditoría de BRTI, Kalshi y manifest previo de Polymarket. | Diferenciar archivos presentes de datos completos y reportar faltantes. |

### 3.3. Por qué hacen falta dos catálogos de Kalshi

Si solo se consulta el catálogo reciente, pueden perderse contratos antiguos. La celda 6 consulta ambos catálogos, filtra localmente por fecha y conserva un registro por ticker. Para duplicados, el recorrido da prioridad al registro histórico.

La selección del endpoint de velas en C8 tiene en cuenta la procedencia y el cutoff de liquidación. Esto intenta evitar que la descarga omita contratos por cambios de ubicación en la API.

### 3.4. Qué significa completo en BRTI

La auditoría de C14 exige **18.000 timestamps únicos por hora**, que corresponderían a cinco observaciones por segundo durante 3.600 segundos.

Una hora con 3.600 observaciones a segundos exactos falla ese control de 5 Hz, pero puede servir parcialmente para el procesamiento a un segundo del notebook 02. Por eso hay que separar:

1. Archivo de la hora presente.
2. Hora completa a 5 Hz.
3. Segundos y ventanas concretas que realmente se usarán como predictores.

`BRTI_RETRY_INCOMPLETE=False` conserva horas cacheadas aunque no tengan 18.000 ticks. Tener cero errores HTTP tampoco garantiza datos completos.

### 3.5. Cómo ayuda a operar contratos

El modelo necesita saber qué contrato existía, cómo cotizaba, cuál era su referencia y cómo liquidó. Este notebook reúne esa materia prima. Una omisión de contratos o un dato mal fechado puede producir una oportunidad aparente que nunca existió.

**Detalles desactualizados:** C10 y algunos comentarios de C11 todavía mencionan 30 días y 720 horas. El código calcula las horas usando `START` y `END`; para la configuración guardada son **8.040 horas**. La auditoría también conserva un mensaje de advertencia referido a 30 días. Es texto pendiente de actualizar, no el período que realmente usa la función.

## 4. Notebook 02: construir el dataset

**Archivo:** [02_BTC15m_walkforward_corregido.ipynb](../02_BTC15m_walkforward_corregido.ipynb).

### 4.1. Qué representa una fila

Una fila representa **un contrato en un instante de decisión**, normalmente al cierre de una vela de un minuto. Un contrato aporta varias filas y todas comparten su resultado final.

Ejemplo conceptual:

| Contrato | Momento | Minutos restantes | Información de mercado | Resultado final |
|---|---|---:|---|---:|
| X | Minuto 1 | 14 | Datos disponibles en ese instante | 1 |
| X | Minuto 2 | 13 | Datos actualizados hasta ese instante | 1 |
| X | Minuto 3 | 12 | Datos actualizados hasta ese instante | 1 |

El resultado `target_yes` es la etiqueta que el modelo intenta aprender. **No se incluye como predictor.**

### 4.2. Lectura de Kalshi

| Dónde | Qué hace | Relación con el objetivo |
|---|---|---|
| **C2, L11–20** | Carpetas, período, `LAMBDA=0.96`, lag BRTI de dos segundos. | Fijar convenciones de procesamiento y temporalidad. |
| **C4, L1–25** | `read_json`, `unix_s`, `first_number`. | Normalizar respuestas y timestamps. |
| **C4, L27–33** | Cargar `markets.json`. | Recuperar contratos y campos disponibles. |
| **C6, L1–3** | `parse_outcome`. | Convertir YES/NO en etiquetas 1/0. |
| **C6, L9–41** | Lectura de precios, controles y asserts de formatos. | Conservar precios en USD dentro de `[0,1]`; no dividir por 100. |
| **C6, L43–72** | Recorrer contratos y velas. | Crear decisiones entre inicio y cierre; el inicio nominal es `close − 900`. |
| **C6, L74–87** | Merge por ticker, tiempo restante, midpoint, proxy y spread. | Reunir cada observación con su contrato y variables de mercado. |
| **C6, L88–100** | Resúmenes y controles de rango. | Detectar datos ausentes o escalas inválidas antes de modelar. |

`kalshi_yes_price_proxy` usa primero el cierre de operaciones de la vela y, si falta, el midpoint de sus cierres bid/ask. No es siempre el midpoint, y tampoco representa necesariamente la cotización disponible para ejecutar al decidir.

`kalshi_spread = ask − bid` sirve como característica de la amplitud de las referencias bid/ask. El código no inventa bid o ask si faltan.

### 4.3. BRTI, strike y ventanas sin mirar adelante

| Dónde | Qué hace | Para qué sirve |
|---|---|---|
| **C9, L1–23** | Leer gzip y retener ticks en segundos exactos. | Reducir frecuencia sin cargar todos los ticks en la tabla final. |
| **C9, L26–41** | Horas esperadas, fallidas y segundos ausentes. | Medir cobertura real. |
| **C9, L43–55** | Localizar discontinuidades. | Documentar huecos entre segundos. |
| **C10, L1–8** | Media móvil de 60 segundos y `means_before`. | Usar solo ventanas completas y consecutivas. |
| **C10, L10–15** | Reconstrucción de strike y prioridad al strike API. | Definir la referencia contra la que se compara BRTI. |
| **C10, L16** | `100 × (brti_mean_60s / strike − 1)`. | Medir distancia porcentual al nivel relevante para el contrato. |

Para una decisión a tiempo `t`, el predictor BRTI usa la media de segundos en **`[t−62, t−2)`**. Se deja un margen de dos segundos y no se rellenan huecos mediante interpolación.

El strike se toma de la API cuando existe. Si falta, se reconstruye como la media de los 60 segundos anteriores al inicio nominal del contrato. Sin embargo, el filtro de entrenamiento posterior **exige `strike_source == 'api'`**: las filas con strike solamente reconstruido quedan fuera del core.

Tener un strike histórico en la API no demuestra por sí solo cuándo fue observable durante el contrato. El propio notebook deja esa limitación documentada.

### 4.4. Retornos y volatilidad EWMA

**C12, L1–16** calcula:

1. Precio BRTI en cada minuto exacto.
2. Retorno logarítmico de un minuto, exigiendo continuidad de segundos.
3. Promedio exponencial de retornos al cuadrado con `λ=0.96`.
4. Raíz de ese promedio, multiplicada por 10.000, para expresar volatilidad en bps por raíz de minuto.
5. Una máscara de continuidad de los últimos 60 retornos para evitar usar inmediatamente estimaciones afectadas por huecos.

La volatilidad describe cuánto se está moviendo el índice. Una misma distancia al strike puede significar algo distinto en un mercado tranquilo que en uno muy volátil.

**C12, L17–27** usa `merge_asof(..., direction='backward', tolerance=120)` para asignar a cada decisión el último registro de minuto disponible hasta `decision_ts − 2`. La comprobación de L24–25 evita que ese timestamp sea posterior al cutoff.

### 4.5. Qué filas se consideran aptas

**C14, L16–33** crea las marcas temporales y tres indicadores principales:

| Indicador | Qué comprueba |
|---|---|
| `brti_complete` | Hay media BRTI de 60 segundos válida. |
| `ewma_valid` | Hay volatilidad aceptada bajo los controles de continuidad. |
| `valid_for_training` | Strike API, BRTI y EWMA disponibles, proxy de Kalshi temporalmente compatible, retorno y etiqueta válidos. |

El filtro de precios bid/ask también se aplica luego en los modelos. Estos indicadores **no certifican latencia de publicación, liquidez o ejecución real**.

**C14, L35–49** crea un split exploratorio cronológico por contrato: aproximadamente 40 % train, 10 % validation y 50 % test. Es para diagnósticos de este notebook. **03 y 04 no entrenan usando esa columna: construyen sus propios folds mensuales.**

### 4.6. Gráficos, clusters y calibración

| Dónde | Qué calcula | Cómo interpretarlo |
|---|---|---|
| **C16, L1–9** | Gráfico de retornos BRTI. | Ver períodos de movimientos fuertes y anomalías. |
| **C17, L1–15** | EWMA de todo el período y últimas 48 horas. | Visualizar cambios en la variabilidad. |
| **C18, L4–16** | Fracción de precios muy bajos a siete minutos del vencimiento. | Detectar saltos extremos de distribución o escala; no identifica automáticamente su causa. |
| **C20, L1–23** | K-means con tres clusters, ajustado solo en train exploratorio. | Describir regímenes de volatilidad; ordena etiquetas por centroide. |
| **C22, L2–16** | Agrupar precios YES al centavo y calcular frecuencia de YES. | Comparar referencia de mercado con frecuencia histórica observada. |
| **C22, L18–31** | Curva de calibración. | Ver si buckets de precio se aproximan a su frecuencia de liquidación. |
| **C23, L1–22** | Errores con signo, ECE y sesgo medio. | Medir discrepancias descriptivas de calibración. |

`vol_cluster` se exporta, pero **no figura entre los predictores de 03 o 04**. Allí sí entra la volatilidad continua.

Los gráficos de calibración usan el dataset completo y varias filas del mismo contrato. Son descriptivos: no constituyen una evaluación independiente para elegir la estrategia ni observaciones estadísticamente independientes.

### 4.7. Qué exporta

**C25, L2–27** selecciona columnas y guarda:

- `dataset_completo.csv` y, si hay soporte, `.parquet`.
- `dataset_core.csv` y `.parquet`, restringido a `valid_for_training`.

**C25, L29–61** guarda `auditoria_dataset.json`, `auditoria_horas_brti.csv` y `auditoria_gaps_brti.csv`, además de imprimir rutas y cantidades.

03 y 04 cargan `dataset_completo`, y después vuelven a aplicar filtros. No suponen que cada fila exportada ya sea apta.

### 4.8. Diccionario de las variables que efectivamente usan los modelos

| Variable | Qué representa | Por qué podría aportar información |
|---|---|---|
| `minutes_remaining` | Minutos hasta el cierre. | Cuánto tiempo queda para que cambie la situación. |
| `brti_distance_to_strike_pct` | Distancia de la media BRTI al strike, en porcentaje. | Cuán por encima o debajo está la referencia. |
| `brti_return_1m` | Movimiento logarítmico reciente de BRTI. | Dirección y magnitud del cambio reciente. |
| `ewma_volatility_bps_min` | Volatilidad reciente suavizada. | Escala de movimientos posibles. |
| `kalshi_yes_price_proxy` | Cierre de operaciones o midpoint como respaldo. | Información que ya refleja el mercado objetivo. |
| `kalshi_spread` | Diferencia entre referencias ask y bid. | Amplitud del mercado cotizado en esas velas. |

## 5. Notebook 03: entrenar XGBoost y simular decisiones

**Archivo:** [03_BTC15m_XGBoost_walkforward_final_octubre.ipynb](../03_BTC15m_XGBoost_walkforward_final_octubre.ipynb).

### 5.1. Qué aprende XGBoost

Aprende una relación entre variables de una observación y su etiqueta final YES/NO. Produce `predict_proba(... )[:,1]`: probabilidad estimada de YES.

En este notebook cada fila se presenta como una observación tabular. No se le entrega explícitamente una secuencia completa como a la LSTM.

### 5.2. Datos y configuración

| Dónde | Qué hace |
|---|---|
| **C2, L12–18** | Carpetas, semilla, embargo de 900 segundos, costos cero y mínimo de 20 trades de validación. |
| **C2, L19–26** | Tres configuraciones de árboles, seis umbrales de edge y seis ventanas temporales. |
| **C2, L29–32** | Buscar y cargar `dataset_completo`. |
| **C3, L11–48** | `prepare_data`: definir A/B, comprobar columnas, filtrar calidad, timestamps, bid/ask y faltantes. |

Se exigen las mismas variables disponibles y bid/ask válidos para comparar A y B sobre una base común. Por eso **A_BRTI no es una evaluación sobre todos los contratos que solo tengan BRTI**: comparte restricciones de disponibilidad de Kalshi.

### 5.3. Walk-forward: cómo se evita entrenar con el futuro

**C3, L50–75** genera folds mensuales:

| Mes de test | Train expansivo, antes de aplicar embargo | Validación |
|---|---|---|
| Mayo 2026 | Diciembre 2025–marzo 2026 | Abril |
| Junio | Diciembre–abril | Mayo |
| Julio | Diciembre–mayo | Junio |
| Agosto | Diciembre–junio | Julio |
| Septiembre | Diciembre–julio | Agosto |
| Octubre | Diciembre–agosto | Septiembre |

La tabla resume meses; el código excluye contratos alrededor de fronteras mediante un embargo de 15 minutos. Se usa `ticker` para mantener todas las filas de un contrato en el mismo bloque.

- **Train:** aprender parámetros del modelo.
- **Validation:** elegir configuración, número de árboles y regla económica.
- **Test:** medir el resultado del procedimiento elegido sin reajustarlo con ese mes.

**C3, L77–79** asigna pesos inversos al número de filas por contrato. Así un contrato con más minutos observados no domina la pérdida solo por aportar más filas.

### 5.4. Entrenar y elegir un modelo

El código principal está en **C5**:

1. **L2–10:** preparar datos, recorrer folds y conjuntos A/B, calcular pesos.
2. **L12–18:** crear y entrenar cada `XGBClassifier` con hasta 1.500 árboles y early stopping de 75 rondas.
3. **L19–23:** comparar log loss ponderada de validación y conservar la mejor configuración.
4. **L24–25:** predecir en test y calcular métricas.
5. **L26–36:** guardar configuración y política del último fold para el entrenamiento final.
6. **L37–45:** ejecutar el backtest de test si hay una política válida.
7. **L46–48:** exportar configuraciones, gráficos, métricas y advertencia de ejecutabilidad.

El early stopping evita seguir agregando árboles cuando la validación deja de mejorar. El criterio predictivo es log loss, no acertar simplemente más etiquetas.

### 5.5. De probabilidad a una operación simulada

**C4** contiene la lógica económica:

| Líneas | Acción |
|---|---|
| **L2–11** | `make_trade_frame`: reunir predicciones y precios, filtrar referencias inválidas y ordenar por contrato/tiempo. |
| **L13–17** | `simulate_strategy`: definir costos y filtrar minutos restantes permitidos. |
| **L20–24** | Recorrer momentos en orden y calcular `edge = p_model − ask`. |
| **L25–28** | Calcular costo de compra, pago final y PnL. |
| **L29–38** | Guardar el registro de la operación y retornos. |
| **L39** | `break`: una vez comprada una unidad YES, dejar de buscar entradas para ese contrato. |

La regla es **comprar YES como máximo una vez por contrato y mantener hasta liquidación**. No simula compras de NO, ventas anticipadas, coberturas ni operaciones cruzadas entre plataformas.

El instante elegido es **la primera señal que cumple la regla**, no el minuto más favorable elegido retrospectivamente dentro del contrato.

Ejemplo dentro de las convenciones del simulador:

```text
Probabilidad estimada:      0,65
Ask histórico de referencia: 0,55
Edge:                       0,10
Costos configurados:         0,00

Si termina YES: PnL = 1 − 0,55 = +0,45
Si termina NO:  PnL = 0 − 0,55 = −0,55
```

Es la diferencia entre **ventaja esperada según el modelo** y **resultado realizado de una operación**. Una entrada con edge positivo puede perder.

### 5.6. Cómo se elige cuándo entrar

**C3, L92–103**, `choose_strategy`, prueba las 36 combinaciones de ventana y edge en validación. Descarta las que generen menos de 20 operaciones y elige la de mayor PnL proxy total, con desempates definidos en el código.

La regla seleccionada se aplica después al test. Incluso si todas las opciones válidas pierden, la función puede elegir la menos mala: **no exige PnL de validación positivo**.

### 5.7. Qué se evalúa y se guarda

**C3, L81–90** calcula log loss, Brier, versiones ponderadas por contrato y AUC.

- **Log loss:** penaliza probabilidades incorrectas, especialmente cuando son muy seguras.
- **Brier:** promedio del error cuadrático entre probabilidad y resultado 0/1.
- **AUC:** ordenamiento de positivos frente a negativos; no incluye precio de entrada.
- **PnL:** resultado económico del backtest bajo sus convenciones.

Una mejora de log loss o AUC no garantiza PnL positivo. Tampoco el win rate basta: importa cuánto se paga por cada contrato.

**C3, L105–133** exporta métricas, operaciones y curvas en `data/modeling/walkforward_xgboost/`. El PnL es una suma por una unidad comprada en cada operación, no una rentabilidad porcentual de una cuenta con capital inicial y restricciones de financiación.

### 5.8. Modelos finales para noviembre

**C7, L1–34** entrena de nuevo A y B con todos los datos aceptados anteriores al 1 de noviembre, incluyendo octubre. Usa la configuración y cantidad de árboles elegidas en el último fold mediante validación de septiembre.

Guarda modelos `.json` y metadatos en `modelos_para_noviembre/`. Ese modelo final **no es el que produjo la evaluación fuera de muestra de octubre**, porque ahora octubre forma parte de sus datos de entrenamiento.

## 6. Notebook 04: aprender secuencias con LSTM

**Archivo principal:** [04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre.ipynb](../04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre.ipynb).

### 6.1. Qué pregunta agrega

XGBoost ve las variables actuales de una fila. La LSTM recibe hasta **10 observaciones consecutivas del mismo contrato**, terminando en la decisión actual.

La pregunta es si la evolución reciente de esas variables aporta más información que una foto aislada. No se mezclan secuencias de contratos distintos ni se utilizan filas posteriores a la decisión.

### 6.2. Configuración y preparación

| Dónde | Qué hace |
|---|---|
| **C2, L15–20** | Semillas y selección de CPU/CUDA. |
| **C2, L21–35** | Rutas, `SEQ_LEN=10`, batch de 256, máximo 60 épocas, paciencia 8 y costos cero. |
| **C2, L36–44** | Ventanas, edges y tres configuraciones LSTM. |
| **C2, L48–51** | 100 realizaciones MC y niveles centrales 80 %, 90 %, 95 % y 99 %. |
| **C3, L11–90** | Mismos filtros, grupos A/B, folds, pesos y métricas que en 03. |
| **C3, L92–133** | Funciones auxiliares heredadas; el flujo LSTM usa su selección específica de C8 y resultados de C10. |

La equivalencia de A/B y folds permite comparar procedimientos con una convención temporal común. No convierte automáticamente la comparación en una prueba estadística de superioridad.

### 6.3. Construir las secuencias

**C4**, `ContractSequenceDataset`:

1. **L2–7:** recibe un bloque train/validation/test y transforma sus variables con un escalador ya ajustado.
2. **L8–11:** reserva arrays de secuencias, longitudes, etiquetas y pesos.
3. **L12–15:** procesa cada ticker por separado y asigna peso inverso al número de filas del contrato.
4. **L16–25:** para cada fila, retrocede como máximo diez observaciones mientras los timestamps estén separados por exactamente 60 segundos.
5. **L26–27:** comprueba que todas las observaciones pertenecen al mismo contrato y son contemporáneas o anteriores.
6. **L29–32:** devuelve tensores para entrenar o predecir.

Si una decisión solo tiene tres observaciones previas consecutivas contando la actual, usa una secuencia de longitud tres. No completa su historia con minutos futuros ni atraviesa huecos para llegar artificialmente a diez.

### 6.4. Qué hace la red

**C5**, `LSTMClassifier`:

| Líneas | Componente | Función |
|---|---|---|
| **L5–6** | Proyección lineal y dropout de entrada. | Convertir 4 o 6 variables en una representación de dimensión 16 y regularizar. |
| **L7–9** | LSTM. | Procesar la secuencia y aprender dependencias temporales. |
| **L10–11** | Dropout final y clasificador de dos clases. | Obtener dos logits: NO y YES. |
| **L12–15** | Inicialización. | Establecer pesos y sesgos iniciales. |
| **L17–22** | `forward`. | Empaquetar longitudes reales, obtener el último estado oculto y clasificar. |

`pack_padded_sequence` permite que los ceros de relleno no se interpreten como observaciones reales. La probabilidad de YES se obtiene luego con softmax sobre los dos logits.

Hay dropout explícito incluso para una sola capa LSTM, porque el dropout interno de `nn.LSTM` no actúa entre capas cuando solo existe una.

### 6.5. Entrenar la red, paso a paso

El bloque principal está en **C8**:

1. **L18–29:** preparar datos, recorrer folds y ajustar `StandardScaler` **solo con train**; transformar los otros bloques con ese mismo escalador.
2. **L32–43:** recorrer configuraciones, fijar semillas, crear red, `AdamW`, función de pérdida y DataLoader.
3. **L45–55:** hacer épocas de entrenamiento:
   - `zero_grad`: borrar gradientes del paso anterior.
   - `model(...)`: obtener logits para el batch.
   - `CrossEntropyLoss`: medir discrepancia entre predicción y etiqueta.
   - Ponderar por contrato y promediar.
   - `backward`: calcular gradientes.
   - `clip_grad_norm_`: limitar su magnitud.
   - `optimizer.step`: modificar pesos para reducir la pérdida.
4. **L56–67:** evaluar en validación, conservar pesos de la mejor época y detenerse tras ocho épocas sin mejora suficiente.
5. **L68–79:** usar la red ganadora para predicciones deterministas y MC en validation/test.
6. **L80–89:** guardar predicciones y configuración del último fold.
7. **L90–120:** seleccionar y evaluar políticas económicas.
8. **L122–125:** exportar configuraciones, predicciones y métricas.

El modelo se elige por log loss ponderada de validación. Las reglas económicas se eligen después, también en validación.

### 6.6. Predicción normal y Monte Carlo Dropout

**Predicción determinista —C6, L4–11—:** `model.eval()` desactiva dropout y genera una probabilidad por fila sin actualizar parámetros.

**Predicción MC —C6, L14–36—:** activa dropout con `model.train()`, pero usa `torch.no_grad()`. Repite 100 veces la predicción con diferentes máscaras. **No reentrena la red ni cambia sus pesos**; cambia qué unidades se apagan en cada pasada.

**Resumen —C6, L38–46—:** calcula media, desvío y cuantiles de las 100 probabilidades por decisión.

Ejemplo conceptual:

```text
Misma decisión, misma red entrenada:
pasada 1 → 0,64
pasada 2 → 0,61
pasada 3 → 0,67
... hasta 100 pasadas

Se resume la dispersión y se obtiene un límite inferior.
```

Un intervalo central del 80 % usa los cuantiles 10 % y 90 %; uno del 99 % usa 0,5 % y 99,5 %. Con 100 realizaciones, esos cuantiles extremos son especialmente sensibles a unas pocas muestras.

La dispersión MC refleja variación bajo dropout. **No prueba que la verdadera probabilidad esté dentro del intervalo con esa cobertura, ni que el contrato tenga 80 % o 99 % de probabilidad de ganar.**

### 6.7. Cómo cambia la decisión de compra

**C7, L14–50**, `simulate_strategy`, conserva la estrategia de una compra YES como máximo por contrato y mantener hasta liquidación.

- **Baseline, L26–27:** usa `p_model` como puntuación para compararla con el ask.
- **MC, L22–25:** usa `mc_lower_80`, `mc_lower_90`, etc., según el nivel elegido.
- **L31–38:** busca la primera señal aceptada y calcula el PnL final.
- **L39–49:** registra la entrada y detiene la búsqueda para ese contrato.

Ejemplo: probabilidad normal de 0,65, límite inferior MC de 0,58 y ask de 0,60. La predicción normal puede aceptar comprar con un edge mínimo pequeño; el filtro MC rechaza esa entrada porque su puntuación conservadora es menor que el ask.

### 6.8. Qué comparación hace realmente baseline vs MC

**C8, L1–16**, `select_policy`, prueba ventanas, edges y niveles MC, exige al menos 20 operaciones y selecciona mayor PnL de validación.

La red y los datos de test son los mismos para baseline y MC, pero **las políticas económicas se seleccionan por separado**. Pueden terminar con distinta ventana y distinto edge, además de diferente puntuación.

Por eso la comparación actual combina el efecto del filtro con el de retocar esas políticas en validación. Aunque el Markdown menciona variantes con igual política económica, el bucle ejecutado solo evalúa `('baseline','mc')` con selección separada. No hay una comparación adicional que mantenga obligatoriamente fija la misma política en ambos.

Como en 03, `select_policy` no exige PnL positivo de validación: elige la mejor alternativa que cumpla el mínimo de operaciones.

### 6.9. Resultados y modelos finales

**C10, L1–28** reúne operaciones y calcula:

- Cantidad de operaciones y contratos.
- PnL total y por operación.
- Proporción de operaciones con PnL positivo.
- Drawdown de la curva acumulada y gráfico comparativo.

El drawdown de C10 es una diferencia monetaria en unidades de PnL, con valores no positivos; **no es un porcentaje** sobre una cuenta. Menos negativo indica una menor caída en esa convención.

**C12, L3–81** entrena modelos finales con datos hasta octubre incluido, conservando arquitectura y épocas elegidas en el último fold. Guarda pesos, escalador, features, arquitectura y políticas en `.pt`.

**C14, L1–23** demuestra cómo reconstruir el modelo B y el escalador desde el checkpoint. No conecta una API de trading ni recibe datos en vivo.

La clase `ContractSequenceDataset` todavía exige `target_yes` en **C4, L10**. Para inferir contratos que aún no liquidaron se necesita adaptar la interfaz para no requerir esa etiqueta. El notebook ya lo reconoce en C13.

## 7. Qué cambia en Copy1 y Copy2

La comparación de todas las celdas fuente muestra **una sola diferencia de código/Markdown** entre cada copia y el principal: **C2, L15**, la semilla.

| Variante | `SEED` |
|---|---:|
| Principal | 42 |
| Copy1 | 4223 |
| Copy2 | 1234 |

El resto de las diferencias del `.ipynb` corresponde a salidas y/o metadatos de ejecución. Cambiar la semilla afecta inicialización de redes y aleatoriedad del entrenamiento; las épocas, configuración ganadora y operaciones pueden cambiar como consecuencia.

No son tres arquitecturas distintas ni tres estrategias independientes diseñadas a mano. Son tres corridas del mismo procedimiento con semillas diferentes.

### Atención con las rutas de salida

Las tres versiones usan **el mismo `OUT`** en C8, L19:

```text
data/modeling/walkforward_lstm_mc/
```

También guardan los mismos nombres de CSV, gráficos y modelos finales. Si se ejecutan desde la misma carpeta, una corrida puede reemplazar los resultados de otra. Las salidas incrustadas dentro de cada notebook conservan sus tablas, pero no identifican por sí solas qué variante dejó los archivos externos actuales.

Para comparar corridas reproduciblemente habrá que separar resultados por semilla o registrar su procedencia. Esta guía documenta el punto; no modifica ese comportamiento.

## 8. Qué muestran las salidas guardadas

### 8.1. Cobertura y dataset

Las salidas de **N01 C6/C8/C11/C14** registran:

- 30.381 contratos listados en el período.
- 104 contratos sin velas según el manifest de descarga.
- 8.040 archivos horarios BRTI; 3.794 horas completas bajo el criterio de 18.000 ticks.
- Referencias a un manifest previo de Polymarket con 956.898 puntos reportados. El notebook actual lo lee, no demuestra cómo se generó.

Las salidas de **N02 C6/C9/C14/C25** registran:

- 423.135 filas y 30.272 contratos con observaciones utilizables antes del filtro core.
- 10.559 segundos BRTI ausentes y 620 discontinuidades.
- 395.105 filas marcadas `valid_for_training`.

Los 30.277 contratos con alguna vela del manifest de N01 y los 30.272 con filas de N02 no son la misma medida: leer una vela no implica que caiga dentro de la ventana de decisiones utilizable. No deben igualarse automáticamente.

### 8.2. PnL proxy que imprimieron los modelos

Los valores siguientes son **sumas de PnL simulado comprando una unidad por operación**, con costos configurados en cero, para test de mayo–octubre. No son porcentajes de rentabilidad de una cuenta.

**XGBoost, N03 C5:**

| Modelo | Operaciones | PnL proxy total |
|---|---:|---:|
| A_BRTI | 8.806 | −29,180 |
| B_BRTI_Kalshi | 5.620 | −32,772 |

**LSTM, C10 de cada variante:**

| Semilla | A baseline | A MC | B baseline | B MC |
|---|---:|---:|---:|---:|
| 42 | −84,984 | −35,200 | −8,115 | +7,300 |
| 4223 | −117,294 | −81,853 | +39,939 | +25,914 |
| 1234 | −98,318 | −108,622 | +8,304 | +10,816 |

Estas tablas sirven para localizar lo que ya se probó. No justifican elegir una semilla por el test ni afirmar una rentabilidad estable: hay sensibilidad a la corrida, diferentes cantidades de operaciones y supuestos de ejecución pendientes.

En XGBoost, B mejora la log loss respecto de A en las tablas guardadas, pero su PnL acumulado es peor. Es un ejemplo concreto de por qué **mejor predicción probabilística y mejor resultado económico son evaluaciones distintas**.

## 9. Diferencias y puntos a revisar antes de sacar conclusiones

### 9.1. Presentación y código corresponden a alcances diferentes

La presentación describe unos 2.848 contratos entre septiembre y octubre de 2026, incluye Polymarket y deja los modelos de ML como siguiente etapa. El código actual solicita diciembre de 2025–octubre de 2026 y ya contiene XGBoost/LSTM.

La presentación también menciona selección de EWMA con QLIKE y silhouette para clusters. En estos notebooks `LAMBDA=0.96` y `n_clusters=3` están fijados; **no se ejecuta aquí una nueva búsqueda QLIKE ni el cálculo de silhouette**. Son referencias a trabajo previo, no pasos presentes en este flujo.

El benchmark de usar directamente el precio de Kalshi como probabilidad no se incluye como modelo separado en las tablas de métricas walk-forward. B incorpora ese precio como feature, y el notebook 02 hace calibración descriptiva, pero eso no sustituye una comparación fuera de muestra explícita contra la probabilidad de mercado.

### 9.2. El filtro de costos no suma edge y costos

En **N03 C4, L22–24** y **N04 C7, L33–35**, el código acepta una señal cuando:

```text
score − ask >= min_edge
y
score − ask >= cost
```

Es equivalente a pedir `score − ask >= max(min_edge, cost)`. **No exige `score − ask >= min_edge + cost`**, aunque algunos textos del notebook sugieren esa lectura.

Con costos cero ambas condiciones coinciden con pedir el edge mínimo. Si se agregan costos, la diferencia importa. Por ejemplo, una diferencia de 0,08 supera un edge mínimo de 0,06 y un costo de 0,03 por separado, pero no alcanza el total 0,09.

Además, el costo se descuenta como costo de entrada; no se incorpora una tabla real de tarifas por contrato ni una simulación de liquidación con todas sus reglas. No se cambió esta lógica en la revisión.

### 9.3. Una referencia histórica no certifica ejecución

Los backtests usan el cierre del ask de la vela que acaba de terminar, sin un minuto adicional de espera. La disponibilidad efectiva de esa vela y de esa cotización en ese instante no está certificada.

No se modelan en detalle fills, latencia de envío, profundidad disponible, cantidades, cola, límites de cuenta o impacto. Tampoco se mantiene una cuenta con capital y exposición máximos. Por eso el código nombra su resultado `pnl_proxy`.

### 9.4. Algunos nombres y comentarios no describen pasos activos

- N01 conserva referencias a 30 días aunque calcula once meses solicitados.
- N04 menciona «tres ablaciones» en el encabezado, pero `models_features` define dos: A y B.
- N04 anuncia variantes con igual política económica, pero el bucle actual selecciona políticas separadas para baseline y MC.
- `MIN_KALSHI_AGE=0` está definido, pero el filtro efectivo exige igualdad entre cierre de vela y decisión. Cambiar solo esa constante no agrega latencia automáticamente.
- Hay imports y funciones auxiliares que no se usan en el recorrido principal. Por ejemplo, importar `IsotonicRegression` en N03 no implica que se estén recalibrando las probabilidades.

### 9.5. Qué falta en el repositorio para reproducir todo

El commit revisado contiene notebooks, salidas incrustadas, README y presentación. **No contiene `data/raw`, `data/processed`, CSV de resultados ni modelos `.pt`/`.json` exportados.** Que una salida diga «Guardado» demuestra lo que informó una ejecución previa; no que el archivo exista en este clon.

Para reproducir harían falta datos o repetir descargas, acceso BRTI cuando corresponda, dependencias compatibles y una carpeta de trabajo consistente. Las versiones de paquetes no están fijadas en un archivo de entorno del commit revisado.

Las semillas ayudan, pero no garantizan reproducibilidad idéntica entre cualquier GPU, CPU y versión de librerías. Para el informe futuro convendrá registrar entorno, datos, semilla y artefactos de cada corrida.

## 10. Recorrido sugerido para estudiar el código

1. **Entender una sola operación:** N03 C4, especialmente `edge`, `payoff` y `pnl`. Es la conexión más directa entre predicción y contrato.
2. **Entender una fila del dataset:** N02 C6, C10 y C12. Identificar momento de decisión, strike, distancia y volatilidad.
3. **Entender qué información se permite usar:** N02 C14 y N03 C3 `monthly_folds`.
4. **Seguir un fold de XGBoost:** N03 C5, desde entrenamiento hasta elección de política y test.
5. **Entender una secuencia LSTM:** N04 C4 y C5, sin entrar todavía en dropout.
6. **Seguir una actualización de pesos:** N04 C8, L45–55.
7. **Agregar MC Dropout:** N04 C6 y C7. Separar dispersión de predicciones de probabilidad de liquidación.
8. **Comparar las semillas y salidas:** sección 7 y tablas de la sección 8.
9. **Volver a la descarga:** N01, para comprender de dónde sale cada variable y qué cobertura tuvo.

Para cada bloque conviene responder: **¿qué recibe?, ¿qué produce?, ¿qué conoce en ese momento?, ¿cómo contribuye a estimar YES o decidir una entrada?**

### Fuentes de esta guía

- Los seis notebooks enlazados, todas sus celdas fuente y salidas textuales guardadas.
- `Presentacion MIA302-1.pptx.pdf`, nueve páginas, con lectura de texto y revisión visual de las diapositivas.
- Inventario y estado Git del repositorio al realizar la revisión.

No se verificaron las APIs actuales ni se reprodujeron los resultados con datos externos. Las descripciones se refieren al comportamiento del código revisado y las convenciones declaradas en él.
