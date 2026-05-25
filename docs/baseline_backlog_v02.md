# Backlog Baseline v02

## Objetivo

Consolidar un `baseline v02` donde la definicion de helada se base en `temp2m_min` y no en `tempsup_min`, incorporando solo cambios metodologicos y de ingenieria que mantengan el proyecto reproducible.

## Estado actual

- La decision metodologica ya es clara: `tempsup` se descarta como sensor objetivo por problemas de lectura y periodos prolongados fuera de rango.
- El pipeline ya fue corregido en codigo para usar `temp2m` como target.
- `temp2m_hourly_2018_2025.csv` ya fue incorporado a `data/raw/`.
- El baseline LSTM `v02` ya fue reentrenado y los artefactos principales fueron regenerados.

## Prioridad 1: cierre del baseline corregido

1. Ingresar `temp2m_hourly_2018_2025.csv` al repo principal.
   - Ruta preferida: `data/raw/temp2m_hourly_2018_2025.csv`
   - Ruta transitoria aceptable: `data/external/temp2m_hourly_2018_2025.csv`
   - Alternativa controlada: variable de entorno `FROST_TEMP2M_SOURCE_PATH`
   - Estado: completado en `data/raw/`

2. Regenerar dataset procesado `v02`.
   - Salidas esperadas:
     - `data/interim/frost_hourly_clean_v02.csv`
     - `data/processed/frost_dataset_v02.csv`
     - `reports/tables/data_quality_report_v02.csv`
   - Estado: completado

3. Reentrenar baseline LSTM `v02`.
   - Salidas esperadas:
     - `models/lstm/lstm_baseline_v02.keras`
     - `models/lstm/lstm_baseline_v02_scaler.joblib`
     - `models/lstm/lstm_baseline_v02_metadata.json`
     - `outputs/metrics/lstm_baseline_v02_metrics.json`
     - `outputs/predictions/lstm_baseline_v02_predictions.csv`
   - Estado: completado

4. Regenerar figuras de evaluacion y presentacion usando `v02`.
   - Estado: completado

Metricas test observadas en el cierre de Prioridad 1:

- accuracy: `0.9694`
- precision: `0.4338`
- recall: `0.9709`
- F1: `0.5996`
- ROC-AUC: `0.9960`

## Prioridad 2: mejoras ya incorporadas en feature engineering

1. Verificar impacto individual de `dir_mean_sin` y `dir_mean_cos`.
2. Verificar impacto de:
   - `temp2m_mean_lag_1h`
   - `temp2m_mean_lag_6h`
   - `temp2m_mean_roll_mean_3h`
   - `temp2m_mean_roll_std_6h`
3. Comparar contra una variante sin estas features para medir valor real.

## Prioridad 3: baseline tabular interpretable

1. Implementar `RandomForest` reproducible en `src/models/`.
2. Reusar el mismo split temporal que el LSTM.
3. Reportar:
   - confusion matrix
   - precision
   - recall
   - F1
   - ROC-AUC
   - importancia de variables

## Prioridad 4: capa operativa

1. Agregar evaluacion de severidad:
   - Normal: `T > 0`
   - Helada meteorologica: `0 >= T > -3`
   - Helada severa/agricola: `T <= -3`
2. Mantener esta capa como salida de evaluacion, no como nuevo target principal.

## Prioridad 5: evidencia metodologica

1. Documentar formalmente la razon para abandonar `tempsup`.
2. Incorporar al texto de tesis:
   - evidencia visual de lecturas fuera de rango;
   - discusion con asesor;
   - justificacion de por que `temp2m` es mas confiable para el baseline.

## No hacer por ahora

- No incorporar ERA5 adicional antes de cerrar `v02`.
- No reabrir `tempsup` como sensor objetivo.
- No mezclar comparaciones entre el baseline antiguo y el nuevo sin aclarar que cambian target y sensor.
- No mover notebooks del repo alternativo al repo principal.
