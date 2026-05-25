# Prediccion de Heladas en el Valle del Mantaro

Proyecto de recuperacion y avance hacia una linea base LSTM para prediccion de heladas usando datos meteorologicos horarios asociados al IGP/EMA.

## Estado actual del repositorio

Este repositorio fue reconstruido a partir de material parcial recuperado. La prioridad de esta version es:

- ordenar el proyecto;
- limpiar archivos innecesarios y sensibles (pdfs, configuraciones locales, reportes pesados);
- separar exploracion, preprocesamiento y modelado;
- documentar supuestos y limitaciones;
- mantener un baseline LSTM reproducible sin optimizacion de hiperparametros.

## Objetivo baseline

Problema baseline adoptado:

- tarea: clasificacion binaria
- target: ocurrencia de helada a `t+12h`
- regla de helada v02: `temp2m_min(t+12h) <= 0 °C`
- secuencia de entrada: 24 horas historicas

Correccion metodologica vigente:

- `tempsup` deja de usarse como sensor objetivo por problemas de lectura fuera de rango observados en distribuciones y tramos prolongados.
- `temp2m` pasa a ser la referencia de temperatura para el baseline corregido.

Estado de ejecucion actual:

- dataset reproducible `v02` regenerado;
- baseline LSTM `v02` reentrenado;
- metricas, predicciones y figuras `v02` ya disponibles en el repositorio.

## Estructura del proyecto

```text
.
├── .codex/
│   └── AGENT.md
├── archive/
│   └── legacy/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/
│   ├── methodology.md
│   ├── laptop_training_requirements.md
│   └── recovery_plan.md
├── models/
│   ├── baselines/
│   └── lstm/
├── notebooks/
│   ├── 00_project_context/
│   ├── 01_eda/
│   ├── 02_preprocessing/
│   ├── 03_feature_engineering/
│   ├── 04_modeling/
│   └── 05_evaluation/
├── outputs/
│   ├── logs/
│   ├── metrics/
│   └── predictions/
├── reports/
│   ├── figures/
│   ├── paper_notes/
│   └── tables/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── visualization/
├── .gitignore
└── requirements.txt
```

## Datos disponibles

Fuentes crudas recuperadas:

- `temp2m_hourly_2018_2025.csv`
- `HR_hourly_2018_2025.csv`
- `radinf_hourly_2018_2025.csv`
- `dir_hourly_2018_2025.csv`
- `vel_hourly_2018_2025.csv`
- `pp_hourly_2018_2025.csv`
- `press_hourly_2018_2025.csv`

Inventario rapido observado:

- periodo: 2018-01-01 a 2025-08-31
- resolucion: horaria
- registros modelables `v02`: 68,652 filas procesadas
- columnas integradas `v02`: 48
- la reconstruccion `v02` usa `temp2m_min` como sensor objetivo

Nota de ingesta:

- el pipeline busca primero `data/raw/temp2m_hourly_2018_2025.csv`;
- si no existe, acepta `data/external/temp2m_hourly_2018_2025.csv`;
- de forma transitoria tambien puede usarse `FROST_TEMP2M_SOURCE_PATH`.

## Pipeline reproducible

### 1. Construir dataset limpio y procesado

```bash
.\ema-venv\Scripts\python.exe -m src.data.make_dataset
```

Salidas principales:

- `data/interim/frost_hourly_clean_v02.csv`
- `data/processed/frost_dataset_v02.csv`
- `reports/tables/data_quality_report_v02.csv`

### 2. Entrenar la linea base LSTM

```bash
.\ema-venv\Scripts\python.exe -m src.models.train_lstm_baseline
```

Salidas principales:

- `models/lstm/lstm_baseline_v02.keras`
- `models/lstm/lstm_baseline_v02_scaler.joblib`
- `models/lstm/lstm_baseline_v02_metadata.json`
- `outputs/metrics/lstm_baseline_v02_metrics.json`
- `outputs/predictions/lstm_baseline_v02_predictions.csv`
- `reports/figures/lstm_baseline_v02_training_curve.png`
- `reports/figures/lstm_baseline_v02_confusion_matrix.png`
- `reports/figures/lstm_baseline_v02_roc_curve.png`

Metricas test del baseline `v02`:

- accuracy: `0.9694`
- precision: `0.4338`
- recall: `0.9709`
- F1: `0.5996`
- ROC-AUC: `0.9960`
- tasa positiva en test: `2.36%`

Artefactos de presentacion regenerados:

- `reports/figures/presentation_v02_frost_rate_by_month_hour.png`
- `reports/figures/presentation_v02_frost_heatmap_month_hour.png`
- `reports/figures/presentation_v02_yearly_frost_counts.png`
- `reports/figures/presentation_v02_morning_frost_incidence_intensity.png`
- `reports/figures/presentation_v02_key_feature_distributions.png`
- `reports/figures/presentation_v02_prediction_timeline_winter_sample.png`
- `reports/figures/presentation_v02_threshold_tradeoff.png`
- `reports/figures/presentation_v02_metrics_summary.png`

## Notebooks

Los notebooks se usan como documentacion ejecutable del pipeline:

- contexto del proyecto y literatura
- entendimiento de datos
- EDA
- preprocesamiento
- ingenieria de variables
- generacion de secuencias supervisadas
- referencias ML simples
- entrenamiento LSTM baseline

## Decisiones de stack

Se mantiene TensorFlow/Keras porque:

- ya estaba presente en el entorno recuperado;
- el notebook legado lo usaba;
- para una primera LSTM pequena en laptop es suficiente y simple.

## Material legado

El material previo se conserva en:

- `archive/legacy/notebooks/`

Esto permite recuperar ideas del trabajo anterior sin usarlo como verdad unica del proyecto.

## Documentacion clave

- [docs/recovery_plan.md](docs/recovery_plan.md)
- [docs/methodology.md](docs/methodology.md)
- [docs/baseline_backlog_v02.md](docs/baseline_backlog_v02.md)
- [docs/laptop_training_requirements.md](docs/laptop_training_requirements.md)
- [reports/paper_notes/literature_synthesis_v01.md](reports/paper_notes/literature_synthesis_v01.md)
