# Plan de Recuperacion del Proyecto

## 1. Diagnostico del repositorio recuperado

Estado observado al inicio de esta sesion:

- El proyecto conservaba 7 CSV horarios del IGP/EMA para 2018-01-01 a 2025-08-31.
- Existia un notebook monolitico (`ema_analysis.ipynb`) con EDA, modelos tabulares y una MLP, pero no una estructura reproducible por etapas.
- El `README.md` reportaba artefactos entrenados y resultados que no estaban presentes en el repositorio actual.
- Habia scripts auxiliares (`predict_example.py`, `utils.py`) acoplados a esos artefactos faltantes.
- El entorno virtual `ema-venv/` estaba dentro del proyecto, lo cual complica versionado y limpieza.
- La carpeta del proyecto no es la raiz del repositorio Git actual; la raiz visible esta en `C:\Users\Pedro Lopevia`. Por seguridad no se reestructuro Git automaticamente.

## 2. Hallazgos tecnicos clave

- Dataset integrado actual: 67,200 registros horarios y 28 variables numericas.
- Variable objetivo contemporanea disponible: `tempsup_min <= 0`.
- Desbalance de clases actual: aproximadamente 5.39% de eventos de helada.
- Filas con faltantes: 561.
- Existen valores fisicamente sospechosos en algunas columnas, por ejemplo:
  - `HR_min < 0`
  - `tempsup_min` extremadamente bajo para el contexto
  - algunos maximos/minimos de presion fuera del rango operativo esperado

Conclusiones:

- Si se entrena directamente sobre el notebook legado, hay riesgo alto de fuga de informacion y de reproducibilidad deficiente.
- El proyecto necesita separar: inventario de datos, limpieza, definicion del target, generacion de secuencias, entrenamiento y evaluacion.

## 3. Decisiones de recuperacion tomadas

Se aplico una reorganizacion no destructiva:

- Los CSV crudos se movieron a `data/raw/`.
- El notebook y scripts legados se preservaron en `archive/legacy/notebooks/`.
- Se creo una estructura modular en `src/`.
- Se creo la carpeta `.codex/` con `AGENT.md`.

## 4. Estrategia metodologica hasta el primer LSTM completo

### Fase A. Inventario y limpieza reproducible

Objetivo:

- Integrar los 7 CSV de forma consistente.
- Aplicar controles fisicos simples y transparentes.
- Guardar un dataset intermedio limpio.

Salida esperada:

- `data/interim/frost_hourly_clean_v01.csv`
- `reports/tables/data_quality_report_v01.csv`
- `reports/tables/dataset_inventory_v01.md`

### Fase B. Dataset procesado para modelado

Objetivo:

- Agregar variables temporales.
- Definir una variable objetivo de pronostico y no de contemporaneidad.

Decision baseline:

- Target de clasificacion: `tempsup_min(t+12h) <= 0 °C`
- Variable binaria: `frost_event_t_plus_12h`

Salida esperada:

- `data/processed/frost_dataset_v01.csv`

### Fase C. Generacion de secuencias supervisadas

Objetivo:

- Construir tensores para LSTM usando una ventana historica corta y segura.

Decision baseline:

- Ventana de entrada: 24 horas
- Horizonte de pronostico: 12 horas

Razon:

- Mantiene complejidad razonable para una laptop sin GPU.
- Es consistente con una primera linea base operativa y explicable.

### Fase D. Entrenamiento baseline LSTM

Objetivo:

- Llegar a un primer entrenamiento completo, documentado y reproducible.

Configuracion baseline:

- Arquitectura LSTM simple
- 32 unidades LSTM
- Dropout ligero de 0.2
- Capa densa oculta pequena
- `binary_crossentropy`
- `Adam(1e-3)`
- `batch_size=32`
- hasta 20 epocas
- `EarlyStopping` con paciencia 5

### Fase E. Evaluacion y artefactos

Objetivo:

- Guardar todo lo necesario para demostrar una primera linea base completa.

Salidas esperadas:

- `models/lstm/lstm_baseline_v01.keras`
- `models/lstm/lstm_baseline_v01_scaler.joblib`
- `outputs/metrics/lstm_baseline_v01_metrics.json`
- `outputs/predictions/lstm_baseline_v01_predictions.csv`
- `reports/figures/lstm_baseline_v01_training_curve.png`
- `reports/figures/lstm_baseline_v01_confusion_matrix.png`
- `reports/figures/lstm_baseline_v01_roc_curve.png`

## 5. Riesgos controlados

### Riesgo de fuga de informacion

Mitigacion:

- Split cronologico por tiempo objetivo, no aleatorio.
- Escalado ajustado solo con entrenamiento.
- Target desplazado a futuro.
- Imputacion basada en propagacion hacia adelante y estadisticos del conjunto de entrenamiento.

### Riesgo de sobrecarga de laptop

Mitigacion:

- Modelo pequeno.
- Sin busqueda de hiperparametros.
- Secuencias de 24 pasos.
- Lotes pequenos.
- Limite manual de hilos de TensorFlow.

### Riesgo de confundir avance real con avance declarado

Mitigacion:

- El `README` y la documentacion deben reflejar solamente artefactos realmente presentes.

## 6. Checkpoints Git recomendados

Debido a que la raiz Git visible no corresponde solo a este proyecto, estos checkpoints se dejan como disciplina recomendada y no se ejecutan automaticamente:

1. `repo-cleanup-and-inventory`
2. `data-pipeline-v01`
3. `eda-notebooks-v01`
4. `preprocessing-and-sequences-v01`
5. `lstm-baseline-v01`

## 7. Estado esperado al cierre de esta etapa

La etapa quedara completa cuando existan:

- estructura limpia del repositorio
- dataset procesado versionado
- notebooks por etapa con markdown tecnico
- documento metodologico y de capacidad de laptop
- entrenamiento baseline LSTM ejecutado y guardado
- plan claro para la siguiente iteracion sin entrar aun a optimizacion
