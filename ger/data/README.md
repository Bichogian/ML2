# Datos de la evaluación BTC15M

El notebook tiene dos modos. **Histórico** reproduce las cifras guardadas con los recortes congelados del estudio. **Demo** genera datos ficticios y permite comprobar el entrenamiento, los cortes temporales, la evaluación y la simulación. Las cifras de la demostración corresponden a datos ficticios.

## Insumos para la corrida histórica

| Ruta relativa en el directorio de datos | Contenido |
|---|---|
| historico/kalshi_btc15m.parquet | Candidatos T−5 del panel DEV, enero–agosto de 2026, con fuente, bid, ask, resultado y reloj. |
| historico/binance_btcusdt_1m.parquet | Velas BTCUSDT de un minuto para los mismos meses. |
| historico/procedencia.json | Huellas SHA-256 del recorte y de sus fuentes. |
| kalshi_btc15m.parquet, binance_btcusdt_1m.parquet y procedencia.json | Recorte más corto para la comparación controlada de BRTI. |
| brti_predecision.parquet y procedencia_brti.json | Ventana de 60 pasos del índice y umbral conocido al corte, junto con sus huellas. |

La carpeta pública contiene el código, el notebook ejecutado y resultados agregados. Los recortes históricos se mantienen en el paquete local del estudio. Para ejecutar en modo histórico se necesita una copia autorizada de los insumos anteriores. El cargador verifica el manifiesto antes de ajustar. La ruta por defecto es data y se puede indicar otra con la variable BTC15M_DATOS.

## Procedencia

- **Kalshi.** Cotizaciones y resultados del panel histórico de desarrollo de KXBTC15M, con cierres anteriores a septiembre de 2026. Las etiquetas se basan en liquidaciones oficiales. [Documentación de la API](https://docs.kalshi.com/) · [Términos de datos](https://kalshi-public-docs.s3.amazonaws.com/kalshi-data-terms-of-service.pdf)
- **Binance.** Velas spot BTCUSDT de un minuto, normalizadas a nanosegundos UTC. [Binance Data Vision](https://github.com/binance/binance-public-data)
- **BRTI y captura rápida de Kalshi.** Grabación de un tercero, revisión 6c729b59f42e002fc8fe9dce2fe5abbda86a1d11, con sello de origen y recepción cuando están disponibles. [narutowindy/kalshiBTC15min](https://huggingface.co/datasets/narutowindy/kalshiBTC15min)
