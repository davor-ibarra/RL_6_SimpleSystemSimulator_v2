---
title: Reporte Release 1
project: RL_6_SimpleSystemSimulator_v2
type: reporte
status: emitido
release: 1
created: 2026-06-20
tags:
  - release
  - validacion
  - cartpole
  - rl-control
---

# Reporte Release 1

## Resumen ejecutivo

Release 1 establece una version funcional y verificable de un simulador CartPole RL+Control. El sistema ejecuta episodios de dinamica fisica, aplica control PID multivariable, ajusta ganancias mediante Q-learning discreto, calcula metricas por intervalo, asigna recompensas firmadas y persiste resultados experimentales en una estructura JSON por chunks.

La version se considera orientada a experimentacion cientifica controlada. Su diseno prioriza legibilidad, trazabilidad y correspondencia directa entre senales fisicas, costos, recompensas y aprendizaje.

## Alcance del release

El release contiene los siguientes modulos operativos:

- construccion explicita de runtime mediante arreglos NumPy;
- integracion no lineal de dinamica CartPole mediante `odeint`;
- control PID en modo posicional e incremental;
- saturacion global y asignacion local del esfuerzo;
- antiwindup condicional;
- Q-learning discreto para ganancias PID;
- procesamiento de metricas por intervalo;
- recompensa local y cooperativa;
- persistencia por episodios y chunks;
- exportacion de metadata, summary, estado del agente y figuras post-run.

No se incluye credito fino por agente ni recompensa estrategica activa. Estas extensiones permanecen fuera del alcance de Release 1.

## Declaracion del modelo computacional

El estado fisico se representa como:

$$
x_k =
\begin{bmatrix}
p_k &
\dot{p}_k &
\theta_k &
\dot{\theta}_k
\end{bmatrix}^T.
$$

Los controladores generan demandas locales:

$$
u^{\mathrm{demand}}_{j,k}
= k_{p,j} e_{j,k}
+ k_{i,j} I_{j,k}
+ k_{d,j} D_{j,k}.
$$

La accion aplicada al sistema se obtiene por:

$$
u_k = \operatorname{clip}
\left(
\sum_j u^{\mathrm{demand}}_{j,k},
u_{\min},
u_{\max}
\right).
$$

El aprendizaje modifica ganancias individuales mediante acciones discretas:

$$
a \in \{-1,0,+1\}.
$$

La recompensa mantiene dominio firmado unitario:

$$
r \in [-1,1].
$$

La senal `cooperative_reward_signed` representa solo la recompensa cooperativa. Las senales `reward_global_signed` y `global_interval_reward` representan la recompensa global compuesta, definida como promedio de las recompensas compuestas de controlador.

## Estado de implementacion

Release 1 cumple las siguientes propiedades estructurales:

- no utiliza carga dinamica por `importlib`;
- no depende de clases base delegadoras;
- no usa records por paso dentro del runtime;
- no usa accesos opcionales para claves de configuracion;
- no reintroduce aliases legacy de control;
- registra saturacion como fenomeno global mediante `saturation_proportion_global`;
- conserva separacion entre runtime numerico y persistencia JSON;
- valida que el `output_schema` solicite senales existentes;
- mantiene Pandas fuera del loop fisico y del loop de aprendizaje.

La persistencia conserva tres secciones por episodio:

- `step_data`;
- `interval_data`;
- `end_episode_data`.

`step_data` conserva muestras causalmente completas $x[k],u[k]$. El estado terminal producido por la ultima transicion se conserva en `end_episode_data` cuando no posee un control posterior valido en el mismo indice.

## Verificacion realizada

La verificacion funcional se ejecuto con una corrida corta de dos episodios. La corrida confirmo:

- generacion de dataset chunk;
- generacion de summary workbook;
- guardado de estado del agente;
- generacion de figuras post-run;
- consistencia de longitudes de series;
- ausencia de valores no finitos;
- acciones restringidas a `-1`, `0` y `+1`;
- costos dentro de $[0,1]$;
- recompensas dentro de $[-1,1]$.

Tambien se verifico que una senal inexistente en `output_schema` produce fallo temprano durante la inicializacion del collector. Esta condicion protege la validez de los datasets exportados.

## Criterios de aceptacion cientifica

El release es aceptable para experimentos iniciales cuando se cumplen las siguientes condiciones:

1. La configuracion declara completamente la simulacion, dinamica, control, agente, metricas, recompensa, datos y visualizacion.
2. La corrida produce artefactos persistentes suficientes para reconstruir el comportamiento por episodio.
3. Las metricas usadas por recompensa son costos normalizados.
4. La recompensa asignada a aprendizaje conserva dominio firmado unitario.
5. El agente aprende exclusivamente desde la senal `reward_for_learning_<agent_id>`.
6. La terminacion se decide en el gestor de simulacion a partir de variables fisicas crudas.

## Limitaciones conocidas

Release 1 no pretende resolver todos los problemas de control adaptativo del CartPole. Sus limitaciones principales son:

- el espacio de estado de Q-learning esta definido por indices discretos de ganancias;
- no se implementa credito fino por agente;
- la recompensa estrategica esta declarada como posibilidad conceptual, pero permanece inactiva;
- la calidad del aprendizaje depende de la discretizacion de ganancias, frecuencia de decision y pesos de recompensa;
- la estabilidad fisica no implica automaticamente optimalidad de aprendizaje.

Estas limitaciones son deliberadas. El release privilegia un nucleo experimental pequeno, inspeccionable y extensible bajo reglas estrictas.

## Recomendaciones para continuidad

Las modificaciones futuras deben seguir el [[../02_manifiesto_reglas_codificacion]]. En particular, cualquier ampliacion debe mantener la separacion entre senales runtime y esquema de salida, conservar nombres canonicos y justificar formalmente toda nueva metrica o recompensa.

Antes de incorporar credito fino por agente, conviene disponer de evidencia comparativa sobre:

- distribucion de recompensas por controlador;
- sensibilidad a pesos locales y cooperativos;
- frecuencia de saturacion global;
- evolucion de visitas en Q-tables;
- relacion entre terminacion por limite y estabilizacion.

## Estado final

Release 1 queda declarado como una base estable para experimentacion inicial en control PID adaptativo mediante Q-learning discreto sobre CartPole. Su valor principal reside en la transparencia de las senales, la persistencia reproducible y la correspondencia explicita entre fenomeno fisico, metrica, recompensa y actualizacion de aprendizaje.

## Documentos relacionados

- [[../00_declaracion_descriptiva_general]]
- [[../01_declaracion_especifica_sistema]]
- [[../02_manifiesto_reglas_codificacion]]
