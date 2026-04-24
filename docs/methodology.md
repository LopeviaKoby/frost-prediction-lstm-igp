# Metodologia del Proyecto

## 1. Problema de modelado

### Pregunta de trabajo

Dado un conjunto de observaciones meteorologicas horarias en una estacion EMA del IGP, se desea pronosticar si ocurrira una helada 12 horas despues.

### Definicion baseline adoptada

- Tipo de problema: clasificacion binaria
- Target baseline: `frost_event_t_plus_12h`
- Regla de helada baseline: `tempsup_min(t+12h) <= 0 °C`

Esta decision es deliberadamente simple:

- es coherente con una primera linea base operativa;
- evita convertir prematuramente el proyecto en un problema mas complejo de pronostico multisalida;
- permite priorizar sensibilidad/recall, que es relevante cuando un falso negativo puede ocultar un evento adverso.

## 2. Contexto bibliografico accionable

### Trasmonte et al. (2008)

Ideas utiles para este proyecto:

- En la cuenca del Mantaro, el riesgo de helada depende fuertemente de altitud, geomorfologia, pendiente, uso de suelo y cobertura vegetal.
- El impacto agricola es especialmente relevante en temporada lluviosa (septiembre a abril).
- El estudio usa temperatura minima historica y una perspectiva espacial de riesgo.

Adopcion metodologica:

- Aunque nuestro dataset actual es esencialmente temporal y puntual, este trabajo justifica usar la cuenca del Mantaro como contexto agroclimatico y documentar que el modelo actual no resuelve la componente espacial del riesgo.

### Saavedra y Takahashi (2017)

Ideas utiles para este proyecto:

- Las heladas radiativas en los Andes centrales estan fuertemente controladas por la radiacion de onda larga descendente, nubosidad baja, humedad especifica y humedad del suelo.
- La variabilidad de la temperatura minima es mas marcada en la temporada seca/fria.
- La precipitacion previa y la humedad del suelo pueden mitigar heladas.

Adopcion metodologica:

- Priorizar temperatura, radiacion, humedad, viento y precipitacion entre las variables de entrada.
- Usar precipitación como proxy simple de estado hidrico superficial, dado que no tenemos humedad de suelo directa.
- Declarar explicitamente que la nubosidad baja no esta observada directamente y solo puede aproximarse mediante radiacion infrarroja y humedad.

### Sulca et al. (2018)

Ideas utiles para este proyecto:

- Los eventos frios extremos en Mantaro, incluso en verano austral, estan asociados con adveccion de aire frio y seco y teleconexiones de gran escala.
- La MJO y la propagacion de trenes de ondas de Rossby extratropicales modulan las anomalias frias.

Adopcion metodologica:

- Este trabajo respalda que hay forzantes de gran escala que nuestro baseline no observa.
- Por ello, el LSTM baseline debe ser interpretado como un modelo local de series temporales, no como un modelo atmosferico completo.
- Se deja como mejora futura la incorporacion de predictores externos (reanalisys, indices MJO/ENSO, nubosidad satelital).

### Bazo et al. (2021)

Ideas utiles para este proyecto:

- Los mecanismos de accion anticipatoria pueden activarse con lead times de varios dias y con umbrales probabilisticos combinados con vulnerabilidad.
- La toma de decision no depende solo de temperatura, sino tambien del impacto esperado.

Adopcion metodologica:

- El proyecto se enfoca primero en la capa hazard: deteccion/pronostico de helada.
- La capa de impacto y priorizacion territorial debe plantearse como una fase posterior.
- Desde la evaluacion ya conviene resaltar recall para no subestimar eventos peligrosos.

### Marengo et al. (2023)

Ideas utiles para este proyecto:

- Una ola de frio puede definirse con percentiles climatologicos y persistencia de varios dias.
- Las irrupciones frias severas estan asociadas a configuraciones sinopticas de gran escala y adveccion meridional intensa.

Adopcion metodologica:

- Para una tesis de pregrado, es razonable separar dos escalas:
  - helada local horaria, que si puede abordarse con EMA + LSTM;
  - ola de frio regional, que requiere otra definicion y otra escala temporal.
- Por eso se evita mezclar en una sola salida eventos sinopticos regionales con heladas puntuales.

### Espinoza et al. (2013)

Ideas utiles para este proyecto:

- Las intrusiones de aire frio al este de los Andes suelen durar 2 o 3 dias y pueden propagarse rapidamente desde latitudes subtropicales.
- Estas intrusiones se asocian a circulaciones de gran escala y a condiciones secas/subsidentes que refuerzan el enfriamiento radiativo.

Adopcion metodologica:

- La ventana baseline de 24 horas es una primera aproximacion razonable para captar el preacondicionamiento local.
- En iteraciones posteriores puede probarse una ventana de 48 o 72 horas, pero no en esta etapa.

## 3. Variables del baseline

Variables meteorologicas disponibles:

- temperatura superficial (`tempsup_*`)
- humedad relativa (`HR_*`)
- radiacion infrarroja (`radinf_*`)
- direccion del viento (`dir_*`)
- velocidad del viento (`vel_*`)
- precipitacion (`pp_*`)
- presion atmosferica (`press_*`)

Variables temporales agregadas:

- hora
- mes
- dia del anio
- dia de la semana
- indicadores binarios de noche y temporada seca
- variables ciclicas seno/coseno

## 4. Limpieza y supuestos de calidad

Reglas baseline:

- valores fuera de rangos fisicamente plausibles se convierten en `NaN`;
- se evita imputacion con informacion futura;
- primero se propaga informacion pasada (`forward fill`) con limite corto;
- los faltantes remanentes se completan con medianas del conjunto de entrenamiento.

Esto no pretende ser un protocolo meteorologico definitivo, sino una capa de control de calidad suficiente para una linea base reproducible.

## 5. Split temporal

Se adopta un split cronologico por tiempo objetivo:

- entrenamiento: targets hasta 2023-12-31
- validacion: targets durante 2024
- prueba: targets desde 2025-01-01

Justificacion:

- evita mezclar pasado y futuro;
- permite usar la historia inmediatamente anterior al bloque de validacion/prueba;
- se parece mas a una situacion operativa real.

## 6. Generacion de secuencias supervisadas

Configuracion baseline:

- ventana: 24 horas
- horizonte: 12 horas
- salida: probabilidad de helada a `t+12h`

Representacion:

- entrada `X`: tensor `(n_muestras, 24, n_features)`
- salida `y`: etiqueta binaria asociada al tiempo objetivo futuro

## 7. Arquitectura baseline LSTM

Arquitectura elegida:

- una sola capa LSTM de 32 unidades
- `Dropout(0.2)` despues del codificador LSTM
- una capa densa oculta de 16 unidades
- salida sigmoide de 1 neurona

Razon:

- lo suficientemente expresiva para una primera linea base secuencial;
- lo bastante pequena para CPU;
- evita sobreingenieria en una etapa de recuperacion.

## 8. Entrenamiento y evaluacion

Entrenamiento:

- hasta 20 epocas
- `batch_size = 32`
- `EarlyStopping`
- pesos de clase para compensar desbalance

Metricas principales:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- matriz de confusion

Criterio interpretativo:

- para heladas, el recall merece atencion especial porque los falsos negativos pueden ser costosos.

## 9. Limitaciones declaradas

- No hay informacion directa de nubosidad, humedad del suelo ni forzantes de gran escala.
- El dataset parece corresponder a una estacion puntual, no a una red espacial.
- La calidad de algunos extremos sugiere ruido instrumental o de integracion.
- El baseline no hace calibracion de umbral ni optimizacion de hiperparametros.

## 10. Siguientes pasos despues del baseline

1. Comparar contra referencias simples: persistencia, dummy classifier, regresion logistica.
2. Revisar umbral operativo con curva precision-recall y costo de falsos negativos.
3. Probar ventana de 48 horas si la laptop lo soporta.
4. Incorporar variables externas de gran escala solo cuando la linea base local este estable.
