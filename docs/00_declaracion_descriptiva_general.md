---
title: Declaracion descriptiva general del sistema
project: RL_6_SimpleSystemSimulator_v2
type: documentacion
status: vigente
release: 1
created: 2026-06-20
tags:
  - cartpole
  - rl-control
  - simulacion
  - documentacion
---

# Declaracion descriptiva general del sistema

## Proposito

El sistema constituye una plataforma computacional especifica para estudiar la interaccion entre control PID multivariable y aprendizaje por refuerzo discreto en un sistema dinamico tipo CartPole. Su finalidad es producir trayectorias episodicas, metricas normalizadas, recompensas interpretables y artefactos persistentes que permitan analizar la evolucion de una politica de ajuste de ganancias PID bajo condiciones fisicas reproducibles.

El objeto cientifico principal no es la generalidad arquitectonica, sino la observabilidad completa de un experimento de control adaptativo. El sistema privilegia un flujo de datos explicito, una semantica de senales estable y una separacion rigurosa entre calculo interno y persistencia externa.

## Naturaleza del experimento

Cada corrida se compone de un conjunto finito de episodios. En cada episodio se integra la dinamica fisica con un paso temporal fino, mientras que el agente de aprendizaje toma decisiones a una frecuencia menor. Esta diferencia de escalas permite representar dos procesos acoplados:

- la evolucion continua aproximada del sistema fisico;
- la modificacion discreta de parametros de control.

Formalmente, un episodio puede describirse como una secuencia de estados fisicos

$$
x_k = [p_k,\ \dot{p}_k,\ \theta_k,\ \dot{\theta}_k]^T,
$$

donde $p$ representa la posicion del carro, $\dot{p}$ su velocidad, $\theta$ el angulo del pendulo y $\dot{\theta}$ la velocidad angular. La integracion avanza con paso $dt$, mientras que la decision del agente ocurre cada $T_d$, con $T_d \geq dt$.

## Componentes conceptuales

El sistema se organiza alrededor de siete responsabilidades cientificas y computacionales:

1. La dinamica fisica calcula la transicion del estado del CartPole.
2. Los controladores PID calculan demandas locales de control sobre variables objetivo.
3. El gestor de control suma, satura y asigna el esfuerzo efectivo aplicado.
4. El agente de Q-learning selecciona acciones discretas de ajuste de ganancias.
5. El procesador de metricas agrega ventanas temporales de senales por intervalo.
6. El calculador de recompensa transforma costos normalizados en recompensas firmadas.
7. El recolector de metricas persiste los resultados en una estructura JSON por episodios y chunks.

Esta division no introduce capas abstractas innecesarias. Cada componente concreto opera directamente sobre arreglos NumPy preasignados y escribe sus resultados por indice.

## Escalas temporales

El sistema distingue tres escalas de datos:

- `step`: una muestra por paso fisico de integracion.
- `interval`: una muestra por intervalo de decision y aprendizaje.
- `episode`: una muestra agregada al cierre del episodio.

La relacion temporal se expresa como:

$$
N_d = \operatorname{round}\left(\frac{T_d}{dt}\right),
$$

donde $N_d$ es el numero aproximado de pasos fisicos contenidos en un intervalo de decision. El gestor de simulacion coordina esta relacion y determina los indices iniciales y finales de cada ventana.

La convencion causal de indice es:

$$
x[k] \rightarrow u[k] \rightarrow x[k+1].
$$

Por tanto, cada muestra persistida en `step_data` contiene un estado $x[k]$ y el control $u[k]$ calculado desde ese mismo estado. El estado terminal producido por la ultima transicion se conserva en `end_episode_data` cuando ya no existe un control posterior valido asociado a esa muestra.

## Flujo informacional

El flujo interno del sistema es estrictamente numerico. Las senales no circulan como diccionarios de registros por paso, sino como columnas escalares preasignadas. En consecuencia, cada componente lee y escribe sobre referencias directas:

$$
\text{array}[\text{indice}] \leftarrow \text{valor}.
$$

Esta decision reduce ambiguedad semantica, evita conversiones intermedias y facilita la verificacion posterior de las series temporales. La persistencia en JSON ocurre solo en la frontera de salida, cuando el episodio ya ha concluido.

## Interpretabilidad de metricas y recompensas

Las metricas que alimentan la recompensa se construyen como costos normalizados en el intervalo $[0,1]$. La convencion principal es que un costo bajo representa mejor desempeno. La recompensa firmada se obtiene mediante la transformacion:

$$
r = 1 - 2c,
$$

donde $c \in [0,1]$ y $r \in [-1,1]$. Esta relacion mantiene una interpretacion lineal:

- $c = 0$ implica $r = 1$;
- $c = 0.5$ implica $r = 0$;
- $c = 1$ implica $r = -1$.

El sistema conserva asi una correspondencia directa entre desempeno fisico, costo agregado y senal de aprendizaje.

El componente cooperativo puede expresar la calidad comun como potencial $\Phi \in [0,1]$ derivado de fuentes normalizadas y declaradas. La configuracion distingue las fuentes del potencial, de la necesidad adaptativa, de la salud usada por el balance, de la sincronizacion temporal y del residual adaptativo. `concurrent_progress` construye el potencial desde los cambios de salud de todos los lazos; `balance` compara sus niveles actuales y `synchrony` consume la alineacion de evolucion calculada por `MetricProcessor`. En la ruta absoluta, la recompensa cooperativa transforma el potencial final mediante $2\Phi-1$. En las rutas de progreso, la recompensa usa $\Delta\Phi$ y, opcionalmente, un termino residual proporcional a $1-\Phi$. Estas senales siguen perteneciendo al dominio firmado antes de su composicion con las recompensas de controlador.

## Persistencia y trazabilidad

Cada episodio se persiste con tres secciones:

- `step_data`: series de alta frecuencia asociadas a la integracion fisica y al control.
- `interval_data`: metricas, recompensas, acciones y variables de aprendizaje.
- `end_episode_data`: estado final, recompensas acumuladas y parametros globales del agente.

La corrida completa conserva metadata, chunks de dataset, resumen tabular, estado del agente y figuras post-run. Esta estructura permite reconstruir analisis experimentales sin depender de objetos vivos en memoria.

## Documentos relacionados

- [[01_declaracion_especifica_sistema]]
- [[02_manifiesto_reglas_codificacion]]
- [[reports/release_1]]
