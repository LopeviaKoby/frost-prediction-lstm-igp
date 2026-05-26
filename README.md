# Predicción de Heladas en la Cuenca del Mantaro mediante Deep Learning

Proyecto de investigación en Deep Learning enfocado en la experimentación, desarrollo y validación de algoritmos de aprendizaje profundo para la predicción de heladas en la Cuenca del Río Mantaro, utilizando observaciones de estaciones físicas del IGP (EMA) y datos de reanálisis satelital global (ERA5).

---

## 1. Objetivos del Proyecto

El objetivo principal es investigar y evaluar arquitecturas de aprendizaje profundo (Deep Learning) aplicadas al pronóstico a corto plazo de heladas radiativas extremas en zonas de alta montaña de los Andes Centrales Peruanos.

*   **Modelado Secuencial**: Modelar la dinámica meteorológica local como un problema de series temporales utilizando redes recurrentes (LSTM) y otras arquitecturas avanzadas.
*   **Alineación Multifuente**: Integrar observaciones en superficie (estaciones EMA) con modelos globales de reanálisis atmosférico (ERA5).
*   **Enfoque de Alerta Temprana**: Priorizar la sensibilidad (recall) en la detección de heladas para minimizar falsos negativos que puedan impactar la toma de decisiones agrícolas.

---

## 2. Metodología de Modelado Baseline

### Definición del Problema
*   **Tipo de problema**: Clasificación binaria (ocurre helada / no ocurre helada).
*   **Horizonte de pronóstico**: Predicción a `t + 12h`.
*   **Variable objetivo**: `frost_event_t_plus_12h` (definida como una temperatura mínima menor o igual a 0 °C en el tiempo objetivo: `temp2m_min(t+12h) <= 0 °C`).
*   **Secuencia de entrada**: Ventana histórica de 24 horas consecutivas de observaciones meteorológicas.

### Arquitectura y Resultados de la Versión Incipiente (LSTM Baseline v02)
Para establecer una línea base de experimentación robusta, explicable y reproducible, se implementó una red recurrente (LSTM) como versión incipiente con la siguiente arquitectura:
1.  **Capa LSTM**: 32 unidades recurrentes para capturar la dependencia secuencial de las últimas 24 horas.
2.  **Capa de Regularización**: `Dropout(0.2)` para mitigar el sobreajuste.
3.  **Capa Densa Oculta**: 16 unidades con activación ReLU para extraer combinaciones no lineales.
4.  **Capa de Salida**: 1 neurona con función de activación Sigmoide.

*El modelo utiliza la pérdida de entropía cruzada binaria y optimización Adam, aplicando un balanceo de pesos de clase debido a que los eventos de heladas representan aproximadamente el 5.39% del histórico.*

#### Métricas de Prueba del Baseline `v02` (Versión Incipiente)
*   **Accuracy**: `0.9694`
*   **Precision**: `0.4338`
*   **Recall (Sensibilidad)**: `0.9709`
*   **F1-Score**: `0.5996`
*   **ROC-AUC**: `0.9960`
*   **Tasa Positiva en Test**: `2.36%`

#### Artefactos de Presentación y Figuras Generadas
*   `reports/figures/lstm_baseline_v02_training_curve.png` (Curva de aprendizaje)
*   `reports/figures/lstm_baseline_v02_confusion_matrix.png` (Matriz de confusión)
*   `reports/figures/lstm_baseline_v02_roc_curve.png` (Curva ROC)
*   `reports/figures/presentation_v02_frost_rate_by_month_hour.png`
*   `reports/figures/presentation_v02_frost_heatmap_month_hour.png`
*   `reports/figures/presentation_v02_yearly_frost_counts.png`
*   `reports/figures/presentation_v02_morning_frost_incidence_intensity.png`
*   `reports/figures/presentation_v02_key_feature_distributions.png`
*   `reports/figures/presentation_v02_prediction_timeline_winter_sample.png`
*   `reports/figures/presentation_v02_threshold_tradeoff.png`
*   `reports/figures/presentation_v02_metrics_summary.png`

---

## 3. Datos del Proyecto

> [!IMPORTANT]
> **Acceso a los Datos Crudos**:
> Los conjuntos de datos meteorológicos crudos originales (archivos CSV de EMA y NetCDF de ERA5) se encuentran almacenados y administrados en carpetas de **Google Drive compartidas exclusivamente con los miembros autorizados del proyecto**. Para su ejecución local, se asume que estos datos son descargados en las rutas especificadas de `data/raw/`.

### A. Estación Física EMA (LAMAR - IGP)
Observaciones locales registradas a resolución horaria por el Instituto Geofísico del Perú (IGP) en la estación LAMAR (Huancayo/Huayao):
*   **Ubicación**: Latitud `-12.03833° S`, Longitud `-75.32278° W`.
*   **Altitud**: 3350 metros sobre el nivel del mar.
*   **Variables de la estación**: Temperatura a 2 metros (`temp2m`), Humedad Relativa (`HR`), Radiación Infrarroja (`radinf`), Dirección del Viento (`dir`), Velocidad del Viento (`vel`), Precipitación acumulada (`pp`), Presión Atmosférica (`press`).

### B. Reanálisis Atmosférico Global (ERA5)
Datos de modelamiento y reanálisis satelital del Centro Europeo de Previsiones Meteorológicas a Plazo Medio (ECMWF):
*   **Parámetros de Descarga y Extensión Geográfica**:
    *   **Área Geográfica (Bounding Box)**: Latitud de `-10.0` a `-14.0° S`, Longitud de `-77.0` a `-74.0° W` (Cubre en su totalidad la Cuenca del Río Mantaro).
    *   **Resolución Espacial**: Rejilla de 0.25° x 0.25° (cada píxel cubre un área aproximada de **28 km x 28 km** en el terreno).
    *   **Estructura de Archivos**: Los archivos NetCDF (`.nc`) mensuales se descargan en formato comprimido que contiene dos archivos NetCDF internos:
        1.  `instant.nc`: Variables instantáneas como temperatura a 2m (`t2m`), componentes de viento (`u10`, `v10`), presión superficial (`sp`), cobertura nubosa (`tcc`), temperatura de suelo (`stl1`), etc.
        2.  `accum.nc`: Variables acumuladas como precipitación total (`tp`).
*   **Método de Extracción Local**:
    Se aplica **interpolación bilineal** en las coordenadas exactas de la estación LAMAR para aproximar y suavizar los valores climáticos de la rejilla de ERA5 al punto geográfico físico exacto del sensor.

---

## 4. Estructura y Contenido del Directorio

El repositorio sigue un diseño modular bajo buenas prácticas de ingeniería de software para proyectos de Ciencia de Datos y Machine Learning:

```text
.
├── data/
│   ├── raw/             # Datos meteorológicos crudos (EMA y ERA5). Compartidos vía Google Drive.
│   │   ├── ema/         # CSVs de la estación local LAMAR (2018-2025).
│   │   └── era5/        # NetCDFs de ERA5 organizados por años.
│   ├── interim/         # Datos limpios con controles de calidad aplicados.
│   └── processed/       # Tensores estructurados y alineados listos para modelado.
├── docs/                # Documentación metodológica y especificaciones de descarga/análisis.
├── models/              # Serialización de los modelos (.keras) y escaladores (.joblib).
├── notebooks/           # Notebooks de Jupyter ordenados de forma cronológica por etapas.
│   ├── 00_project_context/   # Contexto literario y alcance científico.
│   ├── 01_eda/               # Análisis exploratorio y comparación de fuentes (EMA vs. ERA5).
│   ├── 02_preprocessing/     # Pipelines de limpieza y tratamiento de NaNs.
│   ├── 03_feature_engineering/ # Generación de características ciclicas y rezagos temporales.
│   ├── 04_modeling/          # Experimentación de arquitecturas de Deep Learning.
│   └── 05_evaluation/        # Evaluación cuantitativa en test (curvas ROC, métricas).
├── outputs/             # Resultados temporales de predicción y archivos de log.
├── reports/             # Figuras, gráficos y reportes automáticos de calidad.
├── src/                 # Código fuente modularizado del proyecto.
│   ├── data/            # Módulos de carga y descompresión/extracción de ERA5 y EMA.
│   ├── features/        # Preprocesamiento, transformaciones y generación de secuencias.
│   ├── models/          # Definición de arquitecturas de Deep Learning y entrenamiento.
│   ├── visualization/   # Generación de gráficos analíticos y de presentación.
│   ├── config.py        # Configuración de rutas estáticas y parámetros del modelo.
│   └── settings.py      # Módulo para inyectar constantes y variables de entorno desde .env.
├── .env                 # Variables de entorno (API keys, Bounding Boxes, validaciones).
├── .gitignore           # Archivo de exclusión de archivos temporales, entornos y datos pesados.
├── requirements.txt     # Dependencias de Python necesarias (xarray, netCDF4, tensorflow, pandas, etc.).
└── README.md            # Guía del proyecto (este archivo).
```

---

## 5. Pipeline de Ejecución Reproducible

### 1. Construir Dataset de EMA Limpio y Procesado
Lee y valida físicamente los CSVs crudos de la estación local:
```bash
.\ema-venv\Scripts\python.exe -m src.data.make_dataset
```

### 2. Entrenar la Línea Base LSTM v02
Entrena la red LSTM utilizando la segmentación temporal estricta (entrenamiento hasta 2023, validación durante 2024, prueba en 2025):
```bash
.\ema-venv\Scripts\python.exe -m src.models.train_lstm_baseline
```

---

## 6. Documentación Clave

*   [docs/methodology.md](docs/methodology.md): Metodología científica adoptada, justificación de variables y revisión bibliográfica.
