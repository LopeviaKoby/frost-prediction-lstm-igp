# AGENT

## Idioma y tono
- Responder y documentar en espanol tecnico, claro y didactico.
- Mantener el nivel de un proyecto final de pregrado en Ciencia de Datos e Ingenieria de IA.

## Objetivo actual
- Recuperar el proyecto de prediccion de heladas con una linea base LSTM reproducible.
- No hacer optimizacion de hiperparametros en esta etapa.

## Convenciones del repositorio
- Los datos crudos viven en `data/raw/`.
- Los artefactos intermedios reproducibles viven en `data/interim/` y `data/processed/`.
- El codigo reutilizable vive en `src/`.
- Los notebooks documentan, no reemplazan, el pipeline en `src/`.
- El material heredado se conserva en `archive/legacy/`.

## Reglas practicas
- No borrar archivos antiguos; archivarlos si estorban.
- No usar el notebook monolitico legado como fuente de verdad.
- Evitar configuraciones pesadas que comprometan una laptop sin GPU utilizable.
- Mantener nombres versionados con sufijo `v01`, `v02`, etc.

## Objetivo minimo de modelado
- Problema baseline: clasificacion binaria de helada a `t+12h`.
- Ventana baseline: 24 horas de historia.
- Evaluacion prioritaria: recall/sensibilidad, F1 y ROC-AUC.
