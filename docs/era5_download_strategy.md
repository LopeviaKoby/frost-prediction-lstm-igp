# Estrategia de Descarga ERA5 v02

## 1. Correccion de alcance

Se elimina completamente el bloque de descarga diaria del punto del Valle del Mantaro.

La estrategia queda restringida a:

- Cuenca del Mantaro, resolucion horaria, bloque prioritario.
- Sudamerica, resolucion mensual, bloque opcional y posterior.

### Justificacion

Desde el punto de vista de tesis:

- el bloque horario de Mantaro es el que mas valor aporta al pipeline LSTM de heladas;
- mantiene coherencia con la escala temporal del problema;
- permite integrar mejor variables de radiacion, viento, superficie y suelo con tus observaciones EMA.

Desde el punto de vista de recursos:

- evita abrir una tercera rama de datos con poco retorno inmediato;
- reduce complejidad operativa;
- simplifica almacenamiento, trazabilidad y mantenimiento en laptop.

## 2. Parametros interpretados del documento tecnico

### A. Cuenca del Mantaro

- Resolucion temporal: `horaria`
- Periodo: `2018-2025`
- Area: shapefile de la cuenca del Mantaro
- Bounding box de trabajo para descarga previa y recorte posterior:
  - `[ -10, -77, -14, -74 ]` en formato ERA5 `[N, W, S, E]`
- Variables del bloque:
  - `2m_temperature`
  - `10m_u_component_of_wind`
  - `10m_v_component_of_wind`
  - `mean_sea_level_pressure`
  - `surface_pressure`
  - `evaporation`
  - `potential_evaporation`
  - `total_sky_direct_solar_radiation_at_surface`
  - `surface_net_solar_radiation_clear_sky`
  - `skin_temperature`
  - `surface_latent_heat_flux`
  - `surface_sensible_heat_flux`
  - `total_cloud_cover`
  - `runoff`
  - `surface_runoff`
  - `total_precipitation`
  - `soil_temperature_level_1`
  - `soil_temperature_level_2`
  - `soil_temperature_level_3`
  - `soil_temperature_level_4`
  - `volumetric_soil_water_layer_4`
  - `leaf_area_index_low_vegetation`
  - `leaf_area_index_high_vegetation`

### B. Sudamerica

- Resolucion temporal: `mensual`
- Periodo: `1940-2025`
- Area: `90W-15W, 15N, 60S`
- Bounding box ERA5:
  - `[15, -90, -60, -15]`
- Variables legibles del documento:
  - `2m_temperature`
  - `10m_u_component_of_wind`
  - `10m_v_component_of_wind`
  - `mean_sea_level_pressure`
  - `sea_surface_temperature`
  - `surface_pressure`
  - `evaporation`
  - `potential_evaporation`
  - `total_sky_direct_solar_radiation_at_surface`
  - `surface_net_solar_radiation_clear_sky`

## 3. Estimacion de volumen

### Cuenca del Mantaro, horario, 2018-2025

Supuestos:

- malla proxy: `221` celdas
- horas totales: `67,200`
- 23 variables

Estimacion:

- bruto: `~1.27 GiB`
- esperado en disco:
  - NetCDF: `1.0-2.0 GB`
  - GRIB: `0.6-1.1 GB`

Estrategia recomendada:

- descargar por mes
- total esperado: `96` archivos

### Sudamerica, mensual, 1940-2025

Supuestos:

- malla `301 x 301 = 90,601` celdas a `0.25°`
- 1,032 meses
- 10 variables

Estimacion:

- bruto: `~3.48 GiB`
- esperado en disco:
  - a `0.5°` y NetCDF: `~1.0-2.5 GB`
  - a `0.25°` y NetCDF: `~2.5-4.5 GB`

Estrategia recomendada:

- descargar anual
- total esperado: `86` archivos si se parte por anio

## 4. Validacion con literatura

Saavedra y Takahashi (2017) no basan su metodologia en una descarga ERA5 directa como insumo unico. El estudio se apoya en:

- observaciones in situ del observatorio de Huancayo;
- analisis estadistico;
- modelacion de suelo y transferencia radiativa.

Hallazgos fisicos que si guian la seleccion ERA5:

- nubosidad baja;
- humedad atmosferica;
- humedad/estado del suelo;
- balance radiativo;
- temperatura minima.

Fuente primaria:

- https://www.sciencedirect.com/science/article/abs/pii/S016819231730062X
- https://repositorio.igp.gob.pe/items/b2025686-5360-4a3e-ba7c-86e642dc88b1

Conclusion:

- replicar exactamente ese enfoque con ERA5 no corresponde;
- usar ERA5 para complementar el modelo local con variables fisicamente relevantes si corresponde a una tesis viable y coherente.

## 5. Factibilidad en la laptop

Hardware disponible:

- CPU: 4 nucleos
- RAM: ~12 GB
- Sin GPU

Clasificacion:

- Mantaro horario: `viable con reduccion`
- Sudamerica mensual: `viable con reduccion`

Interpretacion:

- si descargas por chunks y procesas incrementalmente, la ejecucion local es razonable;
- no es recomendable disparar todo el plan de una sola vez.

## 6. Validacion de ejecucion nocturna

### ¿El script actual soporta interrupciones de red?

Con la version mejorada, `si`, de forma parcial y controlada:

- cada chunk se descarga de manera independiente;
- si falla un chunk, se reintenta automaticamente;
- los chunks anteriores completados no se vuelven a pedir.

### ¿Se puede reanudar la descarga si falla?

`Si`.

La reanudacion ocurre a nivel de chunk, no a nivel de byte:

- si un mes ya existe y pasa validacion minima, se omite;
- si un mes falla, se vuelve a intentar solo ese archivo en la siguiente ejecucion.

### ¿Existe riesgo de corrupcion de archivos?

`Si`, si el proceso se interrumpe a mitad de escritura.

Mitigacion aplicada:

- la descarga va primero a un archivo temporal `.part`;
- solo al completarse se renombra al `.nc` final;
- si queda un `.part` huérfano, se elimina antes de reintentar.

### ¿Como evitar descargas duplicadas?

Se evita con:

- chequeo de existencia del archivo final;
- validacion minima de tamano;
- manifiesto por chunk;
- archivo de estado `_download_state.json`.

### ¿Como manejar limites del CDS API?

Medidas adoptadas:

- chunking por mes o por anio;
- pausa corta entre solicitudes;
- reintentos con espera;
- no lanzar demasiados archivos simultaneamente;
- no pedir toda la historia en una sola solicitud.

## 7. Diseño robusto de descarga

### Principios

- chunking por mes para Mantaro
- chunking por anio para Sudamerica
- guardado incremental
- logs persistentes
- manifiesto por chunk
- estado persistente de ejecucion

### Estructura dentro del repo

```text
data/raw/era5/
├── _download_state.json
├── _manifests/
├── mantaro_hourly/
│   ├── 2018/
│   │   ├── 2018_06.nc
│   │   ├── 2018_07.nc
│   │   └── 2018_08.nc
│   ├── 2019/
│   └── ...
└── sudamerica_monthly/
    ├── 1940.nc
    ├── 1941.nc
    └── ...
```

### Ventajas de naming

- orden natural cronologico;
- facil de reintentar por archivo;
- facil de mapear a manifiestos y logs;
- facilita versionado por lotes si luego aparecen `v02`.

## 8. Integracion con el pipeline

Ruta general:

- `data/raw/era5/`
- `data/interim/era5/`
- `data/processed/era5/`

Pasos recomendados:

1. descarga chunked en `raw`
2. recorte por shapefile y conversion de unidades en `interim`
3. alineacion temporal con EMA y generacion de features en `processed`

Transformaciones clave:

- `K -> °C`
- `m -> mm` donde corresponda
- control de signos de flujos
- unificacion de zona horaria
- agregacion o resample segun fase
- lags, rolling windows y features fisicamente justificadas

## 9. Flujo nocturno recomendado

### Fase piloto

Comando:

```powershell
.\ema-venv\Scripts\python.exe -m src.data.download_era5 --mode mantaro-hourly --start 2018-06 --end 2018-08 --profile pilot --retries 4 --retry-wait-seconds 180
```

Duracion estimada:

- `20-60 min`, segun cola del CDS

### Fase productiva nocturna

Comando:

```powershell
.\ema-venv\Scripts\python.exe -m src.data.download_era5 --mode mantaro-hourly --start 2018-01 --end 2025-08 --profile core --retries 4 --retry-wait-seconds 180
```

Duracion estimada:

- varias horas
- recomendable dejarlo durante la noche

### Verificacion al dia siguiente

Revisar:

- `outputs/logs/era5_download.log`
- `data/raw/era5/_download_state.json`
- cantidad de archivos por anio en `data/raw/era5/mantaro_hourly/`

Errores tipicos:

- chunks en estado `failed`
- ausencia del `.nc` final
- presencia repetida de `.part`

## 10. Recomendacion final

La estrategia correcta y segura es:

1. excluir completamente el bloque diario;
2. ejecutar primero un piloto horario de Mantaro;
3. si funciona, lanzar Mantaro completo durante la noche;
4. dejar Sudamerica mensual para una fase posterior y separada.
