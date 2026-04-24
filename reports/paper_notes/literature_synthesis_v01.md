# Sintesis Bibliografica Accionable v01

## 1. Trasmonte et al. (2008)

- Contexto: riesgo de heladas en la cuenca del Mantaro con enfasis agroclimatico y espacial.
- Variables/insumos: temperatura minima historica, altitud, geomorfologia, pendiente, uso de suelo, suelos, vegetacion, zonas de vida.
- Criterio de evento: analisis de dias de helada y categorias de riesgo.
- Idea util: documentar que el riesgo no es solo temporal sino tambien espacial.
- Adopcion: solo como contexto territorial y justificacion; no como insumo directo del baseline puntual.

## 2. Saavedra y Takahashi (2017)

- Contexto: controles fisicos de heladas en los Andes centrales del Peru.
- Variables clave: radiacion de onda larga descendente, nubosidad baja, humedad especifica, humedad del suelo, temperatura minima.
- Criterio de evento: dias de helada y variabilidad de `Tmin`.
- Idea util: una parte importante del problema es radiativo y de balance de energia.
- Adopcion: usar radiacion, humedad, precipitacion y temperatura en el baseline; declarar ausencia de nubosidad y humedad de suelo directas.

## 3. Sulca et al. (2018)

- Contexto: eventos frios extremos en Mantaro durante verano austral y su relacion con teleconexiones.
- Variables clave: `Tmin`, OLR, MJO, trenes de ondas de Rossby, adveccion de aire frio y seco.
- Criterio de evento: ECEs definidos a partir de anomalias extremas de temperatura minima.
- Idea util: hay modulacion de gran escala que puede explicar fallos del modelo local.
- Adopcion: no incorporar aun MJO/OLR; dejarlos como ampliacion posterior.

## 4. Bazo et al. (2021)

- Contexto: mecanismo de anticipacion para olas de frio/nieve en Andes del sur peruano.
- Variables clave: percentiles bajos de temperatura, altitud, vulnerabilidad, poblacion expuesta, ganado, pobreza.
- Criterio de activacion: alerta de nivel 4 con lead time de 5 dias y umbrales probabilisticos.
- Idea util: el problema operativo real combina hazard y vulnerabilidad.
- Adopcion: en esta etapa solo modelar hazard; dejar impacto/accion anticipatoria para una fase posterior.

## 5. Marengo et al. (2023)

- Contexto: ola de frio regional durante invierno austral 2021.
- Variables clave: temperaturas maximas/minimas diarias, percentiles climatologicos, configuracion sinoptica.
- Criterio de evento: maximas y minimas por debajo del percentil 10 por al menos 3 dias consecutivos.
- Idea util: distinguir entre helada puntual y ola de frio regional.
- Adopcion: no mezclar escalas; mantener el baseline como clasificacion puntual horaria.

## 6. Espinoza et al. (2013)

- Contexto: intrusiones de aire frio al este de los Andes y su propagacion.
- Variables clave: temperatura minima diaria, circulacion de gran escala, vientos de bajo nivel, subsidencia y sequedad.
- Duracion tipica: 2 a 3 dias.
- Idea util: la memoria atmosferica relevante puede exceder una sola noche.
- Adopcion: baseline con 24 horas; posible extension futura a 48-72 horas.

## 7. Decisiones concretas para este repositorio

Se adopta en v01:

- objetivo binario de helada a 12 horas;
- ventana de 24 horas;
- uso de temperatura, humedad, radiacion, viento, presion y precipitacion;
- metricas centradas en recall/F1/ROC-AUC;
- explicacion explicita de que faltan predictores externos de gran escala.
