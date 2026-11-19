# Anotaciones BG: ideas y aportes para el TP de ML2

**Fecha:** 18 de noviembre de 2026.

Estas notas reúnen las propuestas de BG para ampliar el trabajo sobre contratos BTC15m de Kalshi. Los primeros tres puntos tienen el orden de prioridad indicado por BG. Después se incluyen dos ideas exploratorias adicionales.

Son propuestas para desarrollar y evaluar; no describen modificaciones ya implementadas ni mejoras de rendimiento ya demostradas.

## 1. Analizar operaciones que estuvieron en ganancia y terminaron perdiendo

### Qué me interesa conocer

Quiero revisar cuántos contratos compramos, en cuántos estuvimos ganando durante su vida y cuántos terminaron perdiendo por mantenerlos hasta la liquidación.

Por ejemplo: compramos YES a 0,40, durante el contrato aparece una posibilidad de vender a 0,60, después cambia la situación y el contrato termina liquidando NO. Me interesa entender si, mientras estábamos ganando, la información del modelo ya sugería que convenía salir.

### Qué revisaríamos

- Cuántas operaciones tuvieron una ganancia de salida disponible en algún momento.
- Cuántas de esas operaciones terminaron con pérdida al esperar hasta el vencimiento.
- Qué ganancia llegaron a mostrar y cuánto duró esa situación.
- Qué predicción tenía el modelo en esos momentos y cómo fue cambiando.
- En cuáles una regla de salida basada en información disponible entonces habría indicado cerrar.
- Qué resultado habría dado esa regla frente a mantener hasta liquidación.

Para estimar una salida miraríamos el precio al que se podía vender —bid— y los costos. Con las velas actuales, eso seguiría siendo una aproximación histórica, no una confirmación de ejecución.

### Para qué nos sirve

Este análisis permitiría distinguir problemas de entrada de problemas de gestión posterior. Puede ocurrir que una compra haya tenido una evolución favorable, pero que el procedimiento actual no tenga una forma de aprovecharla antes de que desaparezca.

También separa dos preguntas: qué habría sido posible mirando toda la historia y qué habría decidido realmente una regla aplicada en ese momento. No se trata de elegir retrospectivamente el precio máximo.

**Resultado esperado de este paso:** una tabla de operaciones con su recorrido, la evolución de las predicciones y una comparación de posibles salidas frente a la liquidación final.

## 2. Incorporar cierres anticipados cuando se agota la ventaja de mantener

### Qué quiero agregar

Me interesa poder ir realizando ganancias cuando una operación deja de ser suficientemente atractiva, sin esperar obligatoriamente al vencimiento.

La idea es que, si ya estoy ganando y el modelo pasa a valorar el YES aproximadamente al precio al que puedo venderlo, considere cerrar. En ese escenario se habría reducido la ventaja de seguir expuesto y prefiero asegurar la ganancia disponible.

### Cómo sería el proceso

Después de comprar, el sistema seguiría observando el contrato y actualizando su predicción. En cada revisión compararía:

- El precio al que entré.
- El precio al que podría salir ahora.
- La valoración actual del modelo.
- El tiempo restante, la situación del mercado y los costos.

Con esa información decidiría si mantener o cerrar la posición. El margen para considerar que la ventaja se agotó quedaría por definir y evaluar.

Por ejemplo: compré a 0,40, ahora puedo vender a 0,65 y el modelo estima un valor cercano a 0,65. La propuesta es considerar una salida que asegure aproximadamente 0,25 por contrato antes de costos, en lugar de seguir expuesto hasta el resultado final.

### Para qué nos sirve

Agrega gestión durante la vida del contrato. Permite estudiar si podemos conservar parte de las ganancias que identifica el análisis del paso 1 y reducir exposición cuando la nueva información ya no justifica seguir manteniendo.

El objetivo de este paso es **cerrar la posición existente**. Pasar después a una posición neta en NO sería otra decisión, que podría estudiarse por separado.

**Resultado esperado de este paso:** un backtest que compare la regla actual de mantener hasta liquidación con una regla que permita cerrar antes usando información contemporánea.

## 3. Combinar contexto largo del activo y contexto corto del contrato

### Qué quiero agregar

Me gustaría que la decisión combine dos escalas de información:

- **Contexto largo:** una visión más amplia del comportamiento del activo, sus movimientos históricos y el entorno en el que está operando.
- **Contexto corto:** lo que ocurrió recientemente dentro del contrato, por ejemplo durante sus últimas diez observaciones.

La intención es que el modelo pueda ponderar ambas miradas, de una forma más cercana a considerar tanto el contexto general como la situación inmediata.

### Cómo podría organizarse

Una parte del modelo procesaría variables de contexto amplio, como tendencias, retornos y volatilidad de períodos más largos. Otra parte procesaría la secuencia reciente del contrato.

Después se combinarían sus representaciones o predicciones mediante pesos aprendidos. Una primera versión podría usar una combinación sencilla; otra posibilidad sería permitir que la importancia de cada parte cambie según la volatilidad y el tiempo restante.

Los períodos concretos y la forma de combinar las partes quedan por definir. El contexto largo se calcularía con información disponible hasta la decisión, sin incorporar datos posteriores.

### Una distinción importante

La LSTM actual aprende sus parámetros usando muchos ejemplos históricos, pero cada predicción recibe una ventana reciente del contrato y variables que resumen parte del pasado. Esta propuesta busca agregar **contexto amplio explícito a cada decisión**, además de ese aprendizaje general.

Los nombres contexto largo y contexto corto evitan confundir esta propuesta con los conjuntos actuales **A = BRTI** y **B = BRTI + Kalshi**. Son dos clasificaciones diferentes.

**Resultado esperado de este paso:** comparar el modelo actual con una versión que combine ambas escalas y estudiar qué aporta cada una.

## Idea adicional: actualizar una distribución de escenarios durante el contrato

### Qué estoy imaginando

Me interesa construir una distribución que vaya cambiando con la situación del contrato y permita reevaluar, cada cierto tiempo, cómo podría terminar.

La pregunta sería:

> Con el precio actual, el tiempo que queda y el comportamiento que viene teniendo el activo, ¿qué desenlaces son plausibles ahora y sigue teniendo sentido mantener mi posición?

No se plantea conocer de antemano el futuro. Se plantea actualizar los escenarios posibles a medida que llega información nueva y usarlos como apoyo para decidir.

### Qué información podría combinar

- El tiempo que falta hasta el vencimiento.
- El precio actual del subyacente y su distancia al strike.
- Sus movimientos recientes y su volatilidad actual.
- Los patrones de su distribución histórica más amplia.
- El precio actual del contrato y la posición que tenemos, al momento de convertir los escenarios en una decisión.

La distribución histórica general serviría como referencia, ajustada o condicionada por la situación presente. La forma concreta de construir esa distribución es parte de la investigación.

### Qué haría el Monte Carlo

Con esa distribución actualizada se generarían muchos escenarios posibles para el período restante. Cada escenario se evaluaría usando la regla de liquidación del contrato, obteniendo una proporción de resultados YES y NO.

Al pasar el tiempo, cambiarían tanto el horizonte restante como la información de entrada. Se volverían a generar escenarios y a comparar la valoración resultante con el precio de salida disponible.

Por ejemplo, una compra pudo ser atractiva al inicio, pero tres minutos después los nuevos escenarios podrían mostrar que la ventaja se redujo. Esa actualización podría ayudar a decidir mantener o cerrar sin esperar al vencimiento.

### Qué diferencia tiene respecto del MC Dropout actual

El MC Dropout actual repite predicciones de la misma red variando sus máscaras internas. Esta idea propone simular **escenarios del mercado condicionados por su estado**, para reevaluar el desenlace del contrato.

Es una propuesta exploratoria adicional. Quedaría por estudiar cómo estimar la distribución, con qué frecuencia actualizarla y qué aporta a las decisiones frente al procedimiento actual.

## Otra idea posible: Q-teacher y agentes que aprendan a decidir

**Referencia:** [EarnHFT](EarnHFT.pdf).

### Qué me gustaría explorar

Me parece interesante hacer algo parecido al Q-teacher y los agentes de EarnHFT para ver si mejora el rendimiento de nuestra estrategia.

Los modelos actuales estiman probabilidades y una regla transforma esas probabilidades en compras. Esta alternativa agregaría un componente que aprenda a tomar decisiones durante la vida del contrato.

### Cómo podría adaptarse

Primero necesitaríamos un simulador que lleve registro de la posición, las entradas, las salidas y los costos. Sobre ese entorno se podrían definir acciones como no operar, comprar, mantener o cerrar.

Después podríamos calcular retrospectivamente, sobre históricos de entrenamiento y las reglas del simulador, qué tan conveniente habría sido cada acción. Esa información funcionaría como **Q-teacher** para orientar el aprendizaje del agente.

El agente podría recibir las probabilidades de XGBoost o LSTM, los contextos largo y corto, el tiempo restante, los precios y su posición actual. Su tarea sería aprender qué hacer con esa información para mejorar el resultado de las operaciones.

También podría estudiarse si conviene tener agentes especializados en situaciones diferentes y un selector que decida cuál utilizar, tomando como inspiración la organización de EarnHFT.

### Para qué nos serviría

Permitiría investigar si aprender la gestión de la posición aporta algo frente a reglas fijas de entrada y salida. El resultado a comparar sería el de la estrategia completa, incluyendo costos y decisiones intermedias.

El teacher usaría la continuación conocida de los datos de entrenamiento para preparar su guía. Durante la evaluación en períodos nuevos, el agente tomaría decisiones con información disponible hasta ese momento.

### Cómo se relaciona con las prioridades

Esta idea puede apoyarse en los pasos anteriores: analizar primero las oportunidades de salida, incorporar un simulador que permita cerrar y luego explorar agentes que aprendan esa gestión. La adaptación del Q-teacher se plantea como una línea posterior de investigación, no como un reemplazo inmediato de todo lo que ya está hecho.
