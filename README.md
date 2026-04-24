# Prediccion de Heladas en el Valle del Mantaro

Proyecto de recuperacion y avance hacia una linea base LSTM para prediccion de heladas usando datos meteorologicos horarios asociados al IGP/EMA.

## Estado actual del repositorio

Este repositorio fue reconstruido a partir de material parcial recuperado. La prioridad de esta version es:

- ordenar el proyecto;
- separar exploracion, preprocesamiento y modelado;
- documentar supuestos y limitaciones;
- llegar a un primer entrenamiento LSTM reproducible sin optimizacion de hiperparametros.

## Objetivo baseline

Problema baseline adoptado:

- tarea: clasificacion binaria
- target: ocurrencia de helada a `t+12h`
- regla de helada: `tempsup_min(t+12h) <= 0 °C`
- secuencia de entrada: 24 horas historicas

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

- `tempsup_hourly_2018_2025.csv`
- `HR_hourly_2018_2025.csv`
- `radinf_hourly_2018_2025.csv`
- `dir_hourly_2018_2025.csv`
- `vel_hourly_2018_2025.csv`
- `pp_hourly_2018_2025.csv`
- `press_hourly_2018_2025.csv`

Inventario rapido observado:

- periodo: 2018-01-01 a 2025-08-31
- resolucion: horaria
- registros: 67,200
- columnas integradas: 28
- desbalance preliminar: ~5.39% de heladas usando `tempsup_min <= 0 °C`

## Pipeline reproducible

### 1. Construir dataset limpio y procesado

```bash
.\ema-venv\Scripts\python.exe -m src.data.make_dataset
```

Salidas principales:

- `data/interim/frost_hourly_clean_v01.csv`
- `data/processed/frost_dataset_v01.csv`
- `reports/tables/data_quality_report_v01.csv`

### 2. Entrenar la linea base LSTM

```bash
.\ema-venv\Scripts\python.exe -m src.models.train_lstm_baseline
```

Salidas principales:

- `models/lstm/lstm_baseline_v01.keras`
- `models/lstm/lstm_baseline_v01_scaler.joblib`
- `outputs/metrics/lstm_baseline_v01_metrics.json`
- `outputs/predictions/lstm_baseline_v01_predictions.csv`
- `reports/figures/lstm_baseline_v01_training_curve.png`

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

- [docs/recovery_plan.md](C:\Users\Pedro Lopevia\Escritorio\frost_ema_igp\docs\recovery_plan.md)
- [docs/methodology.md](C:\Users\Pedro Lopevia\Escritorio\frost_ema_igp\docs\methodology.md)
- [docs/laptop_training_requirements.md](C:\Users\Pedro Lopevia\Escritorio\frost_ema_igp\docs\laptop_training_requirements.md)
- [reports/paper_notes/literature_synthesis_v01.md](C:\Users\Pedro Lopevia\Escritorio\frost_ema_igp\reports\paper_notes\literature_synthesis_v01.md)
