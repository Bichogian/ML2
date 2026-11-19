# ML2

Repositorio compartido para los trabajos de Machine Learning 2.

## TP: contratos BTC de 15 minutos

El proyecto investiga si podemos estimar la probabilidad de liquidación YES de contratos BTC15m de Kalshi y usar esas estimaciones para evaluar reglas de entrada. La presentación plantea contrastar información de Kalshi, BRTI y Polymarket; los modelos de los notebooks actuales usan BRTI y Kalshi.

- [Presentación intermedia](Presentacion%20MIA302-1.pptx.pdf).
- [Informe preliminar de avances para revisión](docs/informe_avances/Informe_avances_preliminar.md).
- [Guía en lenguaje simple: qué entra, qué pasa y qué sale de cada notebook](docs/Guia_del_proyecto_en_lenguaje_simple.md).
- [Guía del código, paso a paso y con referencias a celdas y líneas](docs/Guia_codigo_BTC15m.md).

## Notebooks

| Orden | Archivo | Función |
|---|---|---|
| 01 | [Descarga histórica](01_BTC15m_historico_completo_corregido.ipynb) | Contratos y velas de Kalshi, BRTI y controles de cobertura. |
| 02 | [Dataset y exploración](02_BTC15m_walkforward_corregido.ipynb) | Variables, sincronización temporal, calidad y exportación. |
| 03 | [XGBoost](03_BTC15m_XGBoost_walkforward_final_octubre.ipynb) | Walk-forward mensual y simulación de compras YES hasta liquidación. |
| 04 | [LSTM + MC Dropout](04_BTC15m_LSTM_walkforward_MC_Dropout_final_octubre.ipynb) | Secuencias y comparación de entrada determinista frente a filtro MC. |

Las variantes `Copy1` y `Copy2` del notebook 04 cambian la semilla a 4223 y 1234, respectivamente; el principal usa 42. Todas escriben en las mismas rutas de salida, por lo que ejecutar una puede reemplazar artefactos de otra.

## Datos y resultados

El flujo lee y escribe rutas relativas `data/raw/`, `data/processed/`, `data/modeling/` y `figures/`. Estos datos y artefactos no están incluidos en el commit inicial de los notebooks; las salidas visibles dentro de ellos corresponden a ejecuciones previas.

Los resultados económicos son **PnL proxy de simulaciones históricas**, con costos configurados en cero y sin verificación de ejecución o liquidez. Los notebooks no envían órdenes de trading.
