# Requisitos de Laptop para el Primer Entrenamiento LSTM

## 1. Evaluacion de la laptop observada en esta sesion

Se pudo medir directamente lo siguiente desde el entorno Python del proyecto:

- CPU logicos detectados: 8
- CPU fisicos detectados: 4
- RAM total detectada: ~11.7 GB
- TensorFlow detectado: 2.20.0
- GPU visible para TensorFlow: ninguna
- `nvidia-smi`: no disponible

Limitacion:

- Windows bloqueo algunas consultas de hardware mas detalladas, por lo que el modelo exacto de CPU/GPU y la VRAM no pudieron confirmarse desde esta sesion.

Conclusion practica:

- La laptop es apta para un primer LSTM pequeno en CPU.
- No es una configuracion adecuada para entrenamiento pesado, multiples corridas extensas o busqueda agresiva de hiperparametros.

## 2. Hardware minimo y recomendado

### Minimo viable para esta etapa

- CPU: 4 nucleos fisicos
- RAM: 8 GB
- GPU: no obligatoria
- VRAM: 0 GB si se trabaja solo en CPU
- almacenamiento libre: al menos 5 GB

### Recomendado para trabajar con margen

- CPU: 6 a 8 nucleos fisicos
- RAM: 16 GB
- GPU: opcional pero util si TensorFlow la reconoce
- VRAM: 4 a 8 GB si se usa GPU local
- almacenamiento libre: 10 GB o mas

## 3. Factibilidad por tamano de dataset

### Dataset pequeno

Perfil orientativo:

- menos de 50 mil filas
- menos de 20 variables
- ventana de hasta 24 pasos

Factibilidad:

- Entrenable localmente en CPU sin mayor problema.

### Dataset mediano

Perfil orientativo:

- alrededor de 50 mil a 200 mil filas
- 20 a 50 variables
- ventana de 24 a 48 pasos

Factibilidad:

- Viable en laptop local si el modelo es pequeno y el batch size es moderado.
- Este proyecto cae en esta zona baja/media.

### Dataset grande

Perfil orientativo:

- mas de 200 mil filas
- muchas variables derivadas
- ventanas largas o varios experimentos encadenados

Factibilidad:

- Poco recomendable en esta laptop sin GPU utilizable.
- Mejor usar Colab, Kaggle o nube.

## 4. Restricciones practicas para este proyecto

Configuracion segura baseline:

- longitud de secuencia: 24 horas
- batch size: 32
- numero de epocas: hasta 20
- numero de features: aproximadamente 30 a 40
- split: train hasta 2023, validacion 2024, prueba 2025

Riesgos de memoria y estabilidad:

- aumentar mucho el `batch_size` puede disparar uso de RAM;
- ventanas de 48-72 horas incrementan memoria y tiempo por epoca;
- muchas variables derivadas multiplican el tamano del tensor;
- correr varias notebooks pesadas en paralelo puede colgar la laptop;
- busqueda de hiperparametros con muchas corridas es especialmente riesgosa en este hardware.

## 5. Donde conviene entrenar

### CPU local

Si:

- se usa el baseline actual;
- el objetivo es validar pipeline y obtener una primera referencia reproducible.

### GPU local

Si:

- TensorFlow detecta GPU correctamente;
- hay al menos 4 GB de VRAM reales.

En esta sesion:

- no hay GPU utilizable detectada por TensorFlow.

### Google Colab

Conviene si:

- se quiere acelerar iteraciones;
- la laptop vuelve a mostrar inestabilidad;
- se va a probar una ventana mayor o mas epocas.

### Kaggle

Conviene si:

- se necesita una sesion gratuita con GPU/T4;
- los datos ya estan organizados y el flujo es reproducible.

### Cloud VM

Conviene si:

- el proyecto pasa a una fase mas operacional;
- se necesita control total del entorno;
- se incorpora optimizacion o varios modelos comparativos.

## 6. Matriz de decision

### Opcion: entrenar localmente

Usar si:

- una sola corrida baseline;
- secuencia 24;
- batch size 32;
- hasta 20 epocas;
- sin tuning.

### Opcion: entrenar localmente con batch/ventana reducidos

Usar si:

- la laptop se vuelve inestable;
- hay poca RAM libre;
- el sistema empieza a intercambiar memoria o congelarse.

Parametros sugeridos:

- secuencia 12 o 18
- batch size 16
- epocas 10 a 15

### Opcion: usar Colab/Kaggle

Usar si:

- se necesita repetir experimentos varias veces;
- se desea ventana 48;
- se quieren comparar mas de un modelo neuronal.

### Opcion: usar nube

Usar si:

- se planea un pipeline continuo;
- se integraran datos externos pesados;
- se entra a optimizacion sistematica.

## 7. Primera configuracion de entrenamiento segura para una laptop tipica de estudiante

Configuracion recomendada:

- framework: TensorFlow/Keras
- horizonte: 12 horas
- ventana: 24 horas
- LSTM units: 32
- dropout: 0.2
- dense hidden: 16
- batch size: 32
- epocas maximas: 20
- early stopping: si
- numero de notebooks abiertos simultaneamente: minimo

## 8. Checklist de reproducibilidad

- fijar semillas aleatorias (`python`, `numpy`, `tensorflow`)
- registrar versiones del entorno (`requirements.txt`)
- versionar el dataset procesado (`frost_dataset_v01.csv`)
- versionar notebooks por etapa (`*_v01_*.ipynb`)
- versionar artefactos del modelo (`lstm_baseline_v01.keras`, metadata, scaler)
- guardar metricas y predicciones con nombres versionados
- documentar split temporal y definicion exacta del target

## 9. Advertencia importante

No se debe saltar directamente a optimizacion de hiperparametros.

Primero hay que comprobar:

- que la definicion del target sea coherente;
- que no haya fuga de informacion;
- que el pipeline se pueda reproducir;
- que el modelo baseline entrene y evalua sin colgar la laptop.

Solo despues de eso tiene sentido ajustar ventana, batch size, numero de capas o learning rate.
