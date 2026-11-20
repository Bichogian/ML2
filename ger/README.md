# LSTM para predecir Kalshi BTC15M

**Aprendizaje Automático II.** ¿Una LSTM mejora la probabilidad de mercado de un contrato KXBTC15M a cinco minutos del vencimiento? Comparamos una arquitectura many-to-one con Kalshi, su recalibración y una regresión logística. Evaluamos el dato adicional de BRTI en una cohorte con cobertura comparable.

- [Notebook histórico ejecutado](estudio_btc15m.ipynb)
- [Informe en Markdown](informe.md) · [Informe en HTML](informe.html)
- [Datos, archivos y procedencia](data/README.md)

## Resultado

El panel de enero a agosto contiene **6.174 contratos elegibles**. La prueba expansiva de mayo a agosto evalúa **4.745 contratos en 59 días**. Por corte se entrenó con **1.040 a 2.428 contratos** y **12 a 31 días observados**. Los meses iniciales tienen captura parcial y los resultados se muestran también por fuente.

La LSTM obtuvo **0,368049** de log loss frente a **0,368322** de Kalshi. El intervalo pareado de la diferencia incluye cero. La LSTM mejora ligeramente frente a la regresión logística en esta cohorte. En una comparación focal sobre las mismas filas, agregar la media previa de BRTI relativa al umbral cambia el log loss en aproximadamente **+0,000002 nats**, con intervalo que incluye cero. La simulación de los cuatro cortes termina en **−3,32 USD** sobre 72 compras de una unidad.

## Ejecutar

Usá **Python 3.11** y abrí una terminal en esta carpeta.

~~~bash
python -m pip install -r requirements.txt
python -m jupyterlab estudio_btc15m.ipynb
~~~

Seleccioná el kernel del entorno donde instalaste los paquetes y ejecutá todas las celdas. El ajuste corre en CPU.

- **Histórico.** El modo por defecto usa los insumos enumerados en [Datos](data/README.md). La copia pública conserva las salidas históricas. Al ejecutar de nuevo, el cargador pide los recortes con sus manifiestos.
- **Demostración.** Para ejecutar el flujo comprobable con datos ficticios, fijá la variable de entorno BTC15M_MODO a demo antes de iniciar JupyterLab. En Linux o macOS usá el comando siguiente.

~~~bash
BTC15M_MODO=demo python -m jupyterlab estudio_btc15m.ipynb
~~~

En PowerShell:

~~~powershell
$env:BTC15M_MODO = "demo"
python -m jupyterlab estudio_btc15m.ipynb
~~~

El notebook identifica el modo en la primera salida. La comparación focal de BRTI se ejecuta en modo histórico. La prueba sintética de cortes temporales se puede ejecutar con este comando.

~~~bash
python -m unittest discover -s tests -p test_walkforward.py -v
~~~
