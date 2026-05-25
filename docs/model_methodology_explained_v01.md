# Metodologia Aplicada al Modelo y Criterios Cientificos

## 1. Objetivo metodologico

El objetivo de esta etapa no fue encontrar el modelo final mas preciso, sino construir una linea base reproducible y cientificamente defendible para pronostico de heladas.

La prioridad fue:

- asegurar consistencia temporal;
- evitar fuga de informacion;
- trabajar con una arquitectura razonable para laptop;
- alinear el planteamiento con la fisica del problema y con literatura de referencia.

## 2. Definicion del problema

Se definio el problema como una clasificacion binaria:

- entrada: secuencia historica de variables meteorologicas horarias;
- salida: ocurrencia o no de helada a `t+12h`.

Definicion operacional de helada:

- `temp2m_min(t+12h) <= 0 °C`

Razones para esta definicion:

- es simple y trazable;
- es coherente con una primera fase de tesis;
- permite entrenar un baseline sin introducir de entrada complejidad excesiva.
- refleja la decision metodologica de abandonar `tempsup` por problemas de lectura fuera de rango.

## 3. Datos utilizados

Variables EMA integradas:

- temperatura a 2 metros
- humedad relativa
- radiacion infrarroja
- direccion del viento
- velocidad del viento
- precipitacion
- presion atmosferica

Variables temporales agregadas:

- hora
- mes
- dia del anio
- dia de la semana
- indicadores de noche y temporada seca
- codificacion ciclica seno/coseno
- componentes vectoriales del viento
- rezagos y rolling features causales de `temp2m_mean`

## 4. Criterios cientificos usados

### A. Criterio temporal

Las heladas no ocurren aleatoriamente. El analisis exploratorio muestra una estructura temporal clara:

- mayor incidencia en madrugada;
- mayor frecuencia en meses frios/secos;
- presencia de memoria temporal del sistema atmosferico y de superficie.

Esto justifica:

- usar secuencias temporales;
- trabajar con ventanas historicas en lugar de observaciones aisladas.

### B. Criterio fisico

El modelo se apoya en variables consistentes con la literatura sobre heladas andinas:

- temperatura;
- humedad;
- radiacion;
- viento;
- superficie/suelo.

En particular, Saavedra y Takahashi enfatizan el rol de:

- nubosidad baja;
- humedad atmosferica;
- estado del suelo;
- balance radiativo nocturno.

Por eso la integracion de ERA5 se planteo como complemento fisico y no solo como aumento de volumen de datos.

### C. Criterio de evaluacion

No se priorizo solo accuracy.

Para un sistema de alerta de heladas, una falsa omision puede ser costosa. Por eso se dio importancia a:

- recall/sensibilidad;
- F1-score;
- ROC-AUC;
- matriz de confusion.

## 5. Decisiones metodologicas aplicadas

### A. Split cronologico

Se dividio el conjunto en entrenamiento, validacion y prueba respetando el tiempo.

Justificacion:

- evita mezclar pasado y futuro;
- representa mejor un escenario de pronostico real.

### B. Escalado solo con entrenamiento

Las transformaciones numericas se ajustaron usando unicamente el conjunto de entrenamiento.

Justificacion:

- evita fuga de informacion desde validacion o prueba hacia el modelo.

### C. Generacion de secuencias supervisadas

Se construyeron secuencias de 24 horas para predecir la helada 12 horas despues.

Justificacion:

- capta dinamica reciente sin sobrecargar memoria;
- es razonable para una laptop;
- permite una primera aproximacion a la memoria temporal del fenomeno.

### D. Arquitectura baseline LSTM

Se uso una LSTM pequena, con regularizacion ligera y salida sigmoide.

Justificacion:

- suficiente para una primera linea base;
- defendible a nivel tesis;
- evita sobreingenieria temprana.

### E. Detencion temprana

Se permitio `early stopping`.

Justificacion:

- reduce riesgo de sobreajuste;
- reduce costo computacional;
- es apropiado para una fase baseline.

## 6. Mejoras concretas ya aplicadas

Respecto a un trabajo inicial centrado en notebooks aislados, las mejoras reales ya implementadas son:

- reorganizacion del proyecto en pipeline reproducible;
- modularizacion del codigo en `src/`;
- integracion y limpieza de datos con reglas fisicas simples;
- definicion explicita del target de pronostico;
- guardado sistematico de metricas, predicciones y figuras;
- estrategia ERA5 chunked, trazable y robusta para ejecucion larga.

## 7. Hallazgos que ya pueden sostenerse

Con base en los datos actuales:

- las heladas se concentran en horas de madrugada y primeras horas de la manana;
- la incidencia aumenta en meses frios, particularmente alrededor de junio-julio-agosto;
- variables como temperatura minima, humedad, viento y radiacion muestran diferencias entre horas con y sin helada;
- el baseline LSTM ya es capaz de capturar una senal temporal util.

Metricas test observadas en el baseline `v02`:

- accuracy: `0.9694`
- precision: `0.4338`
- recall: `0.9709`
- F1: `0.5996`
- ROC-AUC: `0.9960`

## 8. Lo que aun falta probar

Hay varias preguntas abiertas que deben convertirse en la siguiente fase de trabajo:

### A. Ventana temporal

- probar si 24 horas es suficiente o si 48-72 horas mejora la prediccion.

### B. Horizonte de pronostico

- comparar `t+6h`, `t+12h` y `t+24h`.

### C. Variables ERA5

- cuantificar si la incorporacion de nubosidad, flujos de calor, temperatura del suelo y humedad del suelo mejora el modelo.

### D. Umbral operativo

- estudiar si el umbral `0.5` es el mas adecuado o si conviene moverlo para priorizar recall.

### E. Hiperparametros

- numero de unidades LSTM;
- batch size;
- dropout;
- learning rate;
- longitud de secuencia.

## 9. Lo que todavia no se debe afirmar

En esta etapa no corresponde afirmar todavia:

- que la LSTM ya es el mejor modelo posible;
- que el sistema esta listo para operacion;
- que las variables ERA5 ya demostraron mejora;
- que el tuning de hiperparametros ya fue realizado.

## 10. Mensaje metodologico para presentacion

La idea central que se puede presentar con honestidad es:

> El avance principal no es solo haber entrenado una LSTM, sino haber construido una metodologia reproducible y coherente con la fisica del problema, sobre la cual ya se puede iterar con integracion ERA5, ajuste de umbrales e incorporacion de nuevos experimentos.
