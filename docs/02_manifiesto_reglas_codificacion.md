---
title: Manifiesto de reglas de codificacion del sistema
project: RL_6_SimpleSystemSimulator_v2
type: manifiesto
status: vigente
release: 1
created: 2026-06-20
tags:
  - arquitectura
  - reglas
  - mantenimiento
  - implementacion
---

# Manifiesto de reglas de codificacion del sistema

## Principio rector

Toda modificacion futura debe preservar la claridad causal del sistema. La implementacion debe permitir que un lector reconstruya el experimento desde el codigo sin inferir capas ocultas, compatibilidad historica o contratos implicitos. La prioridad tecnica es la trazabilidad cientifica del flujo:

$$
\text{estado} \rightarrow \text{control} \rightarrow \text{dinamica}
\rightarrow \text{metrica} \rightarrow \text{recompensa}
\rightarrow \text{aprendizaje}.
$$

La simplicidad se entiende como reduccion de intermediarios, no como perdida de rigor.

## Regla 1: contratos concretos

El sistema debe seguir usando componentes concretos. No deben introducirse clases base delegadoras, factories dinamicas, registros de clases ni carga por strings. Las dependencias entre componentes deben ser visibles en `main.py` y expresarse mediante importaciones directas.

No se permite introducir:

- `importlib`;
- `module_path`;
- `class_name`;
- factories genericas;
- registries dinamicos;
- clases base sin logica cientifica propia.

## Regla 2: configuracion estricta

La configuracion es un contrato experimental. Una clave ausente debe producir fallo explicito. No deben usarse defaults silenciosos para corregir configuraciones incompletas.

No se permite:

- accesos opcionales a diccionarios para claves del contrato;
- imputacion silenciosa de senales;
- correccion automatica de estructuras incompletas;
- equivalencias legacy entre nombres.

Las validaciones aceptables son invariantes fisicas o matematicas inevitables, por ejemplo positividad de masas, rangos crecientes y suma de pesos cuando corresponda.

## Regla 3: arrays como runtime

El runtime debe operar sobre arreglos NumPy preasignados. Cada senal escalar registrada debe escribirse por indice:

$$
y[i] \leftarrow v.
$$

No deben circular objetos tipo `step_record`, `interval_record`, `actions_dict` o diccionarios equivalentes dentro del loop fisico o del loop de intervalo. La construccion de diccionarios esta permitida unicamente en la frontera de persistencia.

## Regla 4: separacion entre runtime y salida

El runtime define las senales que existen. El `output_schema` define las senales que se exportan. Estos conceptos no deben confundirse.

La regla operacional es:

- si una senal existe en runtime, puede ser exportada;
- si una senal solicitada por el `output_schema` no existe, el sistema debe fallar al inicializar el collector;
- el `output_schema` no debe crear senales.

Esta separacion protege la validez del dataset: una columna persistida debe provenir de una senal efectivamente calculada.

## Regla 5: nombres canonicos

Cada concepto fisico o computacional debe tener un unico nombre canonico. No deben coexistir aliases para la misma magnitud. En particular:

- `u_total_raw` representa la suma no saturada de demandas locales;
- `u_total` representa la accion final saturada aplicada al sistema;
- `u_demand_<controller>` representa la demanda local del PID;
- `u_alloc_<controller>` representa el esfuerzo local efectivamente asignado;
- `u_unallocated_<controller>` representa la demanda local no asignada por saturacion global.

No deben reintroducirse nombres como `u_eff`, `control_action_eff` o `u_total_saturated`.

## Regla 6: responsabilidades por archivo

Cada archivo debe conservar una responsabilidad dominante:

- `main.py`: carga configuracion, crea runtime, instancia componentes y ejecuta la corrida.
- `runtime_arrays.py`: declara y construye las senales runtime.
- `dynamic_system/cart_pole_dynamic_system.py`: integra la dinamica y escribe estado.
- `controllers/pid_controller.py`: calcula senales PID locales.
- `controllers/controller_manager.py`: aplica acciones del agente, satura y asigna control.
- `agents/pid_qlearning_agent.py`: selecciona acciones discretas y actualiza Q-tables.
- `metrics/metric_processing.py`: agrega metricas por intervalo.
- `rewards/reward_calculator.py`: calcula y asigna recompensas.
- `metrics/metric_collector.py`: convierte arrays en salida persistente.
- `result_handler.py`: escribe chunks, metadata, resumen y estado del agente.
- `visualization_manager.py`: genera figuras post-run.

Una modificacion que cruce estas fronteras debe justificarse por una necesidad cientifica o computacional clara.

## Regla 7: reward interpretable

Toda metrica usada por la recompensa debe ser un costo normalizado en $[0,1]$. Toda recompensa final debe permanecer en $[-1,1]$. La transformacion canonica es:

$$
r = 1 - 2c.
$$

La recompensa local debe depender de metricas propias del lazo, como error y variacion de esfuerzo asignado. La cooperacion debe combinar costos locales normalizados mediante un modo declarado y verificable, sin introducir metricas cruzadas entre lazos. Los modos de potencial admitidos son `norm-p` y `geometric_consensus`; los modos de recompensa cooperativa admitidos son `absolute`, `progress` y `progress_residual`. La ruta `norm-p` + `absolute` debe conservar compatibilidad numerica con la transformacion canonica $r=1-2c$. La memoria temporal cooperativa debe limitarse al potencial anterior del episodio y permanecer dentro de `RewardCalculator`. La recompensa estrategica solo puede activarse si existe una implementacion explicita y verificable.

## Regla 8: aprendizaje sin contexto oculto

El agente de Q-learning solo debe seleccionar acciones discretas:

$$
a \in \{-1,0,+1\}.
$$

El agente no debe escribir `delta_gain_requested`, `delta_gain_applied` ni `gain_value`. La traduccion desde accion discreta hacia variacion de ganancia pertenece al gestor de control. El agente debe aprender desde `reward_for_learning_<agent_id>` y debe obtener la transicion de estado desde `gain_value_<agent_id>`.

## Regla 9: control con saturacion observable

Toda saturacion debe dejar rastro en las series:

- `u_total_raw`;
- `u_total`;
- `delta_u_total`;
- `is_saturated_global`;
- `saturation_excess`;
- `saturation_proportion_global`;
- `u_alloc_<controller>`;
- `u_unallocated_<controller>`.

El antiwindup debe ser explicito. Si se declara `conditional`, la integral solo debe avanzar cuando la asignacion efectiva no refuerza el crecimiento del error bajo saturacion.

## Regla 10: persistencia fuera del loop critico

El collector no debe participar por paso ni por intervalo. Su funcion es persistir slices al cierre del episodio:

$$
\text{array}[0:n] \rightarrow \text{lista JSON}.
$$

Pandas puede usarse para resumenes, planillas o visualizacion post-run, pero no para computo dentro del loop de simulacion.

## Regla 11: lectura lineal

Los metodos principales deben nombrar acciones del dominio:

- `run_episode`;
- `run_interval`;
- `integrate_step`;
- `compute_u_demand`;
- `apply_allocation`;
- `select_actions`;
- `learn`;
- `aggregate_interval`;
- `evaluate_interval`;
- `persist_episode`.

No debe usarse un metodo generico como `calculate()` para ocultar ramas de responsabilidad distintas.

## Regla 12: verificacion minima de cambios

Toda modificacion relevante debe poder superar una verificacion minima:

- compilacion del codigo Python;
- corrida corta de al menos dos episodios;
- validacion de longitudes de `step_data` e `interval_data`;
- ausencia de valores no finitos;
- acciones restringidas a `-1`, `0` y `+1`;
- costos en $[0,1]$;
- recompensas en $[-1,1]$;
- persistencia de chunk, summary y estado del agente cuando corresponda;
- busqueda de patrones prohibidos.

Esta verificacion no reemplaza la evaluacion cientifica, pero reduce el riesgo de errores estructurales.

## Regla 13: incorporacion de nuevas senales

Una nueva senal debe incorporarse mediante el siguiente procedimiento:

1. Definir su significado fisico o computacional.
2. Asignar un nombre canonico.
3. Declarar su existencia en el constructor de runtime si pertenece al estado interno.
4. Escribirla por indice en el componente responsable.
5. Agregarla al `output_schema` solo si debe persistirse.
6. Verificar que su longitud y dominio sean correctos en una corrida corta.

No debe agregarse una senal al `output_schema` como forma indirecta de crear runtime.

## Documentos relacionados

- [[00_declaracion_descriptiva_general]]
- [[01_declaracion_especifica_sistema]]
- [[reports/release_1]]
