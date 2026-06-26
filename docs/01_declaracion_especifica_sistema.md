---
title: Declaracion especifica del sistema CartPole RL Control
project: RL_6_SimpleSystemSimulator_v2
type: especificacion
status: vigente
release: 1
created: 2026-06-20
tags:
  - cartpole
  - pid
  - qlearning
  - recompensa
---

# Declaracion especifica del sistema

## Identidad del sistema

El sistema implementa una simulacion episodica de un CartPole controlado por una estructura PID multivariable, cuyas ganancias son ajustadas mediante Q-learning discreto. La planta fisica, los controladores, el agente, las metricas, la recompensa y la persistencia se definen de forma explicita en el codigo y en la configuracion.

La configuracion vigente declara dos controladores PID:

- `pendulum_angle`, orientado al angulo del pendulo;
- `cart_position`, orientado a la posicion del carro.

Cada controlador posee tres ganancias ajustables: $k_p$, $k_i$ y $k_d$. Por tanto, el sistema contiene seis agentes de ganancia, cada uno asociado a una unica ganancia de un unico controlador.

## Estado fisico

El estado fisico del sistema en el paso $k$ se representa como:

$$
x_k =
\begin{bmatrix}
p_k \\
\dot{p}_k \\
\theta_k \\
\dot{\theta}_k
\end{bmatrix},
$$

donde:

- $p_k$ es la posicion del carro;
- $\dot{p}_k$ es la velocidad del carro;
- $\theta_k$ es el angulo del pendulo;
- $\dot{\theta}_k$ es la velocidad angular del pendulo.

El angulo del pendulo se expresa en el sistema de referencia corregido:

$$
\theta \leftarrow (\theta + \pi) \bmod 2\pi - \pi,
$$

de modo que $\theta = 0$ corresponde a la vertical superior y el dominio angular queda contenido en $[-\pi,\pi]$.

La dinamica escribe tanto valores crudos como valores normalizados. Para una variable $z$ con rango fisico $[z_{\min}, z_{\max}]$ y rango de salida $[a,b]$, la normalizacion lineal se define como:

$$
\hat{z} = a + \frac{z - z_{\min}}{z_{\max} - z_{\min}}(b-a),
$$

con saturacion al intervalo $[a,b]$.

## Accion fisica aplicada

El gestor de control produce una accion total adimensional:

$$
u_k \in [u_{\min}, u_{\max}],
$$

que es transformada por la dinamica en fuerza aplicada al carro:

$$
F_k = u_k \frac{\tau_{\max} G}{r_p},
$$

donde $\tau_{\max}$ es el torque maximo, $G$ es la relacion de engranaje y $r_p$ es el radio del pinion. Esta fuerza se utiliza en la integracion de la dinamica del CartPole.

La evolucion temporal de la planta se integra mediante `scipy.integrate.odeint`, usando las ecuaciones no lineales del sistema CartPole.

## Control PID local

Para cada controlador $j$, el error se calcula a partir de la variable objetivo y el setpoint declarado. El dominio interno del PID puede ser crudo o normalizado; en el modo normalizado, el setpoint fisico se normaliza una unica vez y el control opera sobre $e^{\mathrm{norm}}_{j,k}$, mientras que las senales crudas se preservan con proposito diagnostico.

En modo posicional, la demanda local se define como:

$$
u^{\mathrm{demand}}_{j,k}
= k_{p,j} e_{j,k}
+ k_{i,j} I_{j,k}
+ k_{d,j} D_{j,k},
$$

donde:

$$
I_{j,k} = I_{j,k-1} + e_{j,k}dt,
$$

y

$$
D_{j,k} = \frac{e_{j,k} - e_{j,k-1}}{dt}.
$$

El controlador tambien soporta una forma incremental, en la cual la demanda se actualiza desde la accion previa y las diferencias sucesivas del error.

Para dicha forma incremental se utiliza:

$$
\Delta e_{j,k} = e_{j,k} - e_{j,k-1},
$$

$$
\Delta e_{j,k-1} = e_{j,k-1} - e_{j,k-2},
$$

$$
\Delta^2 e_{j,k} = \Delta e_{j,k} - \Delta e_{j,k-1},
$$

y

$$
\Delta u^{\mathrm{demand}}_{j,k}
= k_{p,j}\Delta e_{j,k}
+ k_{i,j}e_{j,k}dt
+ k_{d,j}\frac{\Delta^2 e_{j,k}}{dt}.
$$

Por tanto:

$$
u^{\mathrm{demand}}_{j,k}
= u^{\mathrm{demand}}_{j,k-1}
+ \Delta u^{\mathrm{demand}}_{j,k}.
$$

## Saturacion global y asignacion local

Las demandas locales se agregan mediante suma:

$$
u^{\mathrm{raw}}_k = \sum_j u^{\mathrm{demand}}_{j,k}.
$$

La accion aplicada se obtiene por saturacion global:

$$
u_k = \operatorname{clip}(u^{\mathrm{raw}}_k,\ u_{\min},\ u_{\max}).
$$

Cuando existe saturacion y $u^{\mathrm{raw}}_k \neq 0$, el sistema calcula un factor de asignacion:

$$
\alpha_k = \frac{u_k}{u^{\mathrm{raw}}_k}.
$$

La asignacion efectiva de cada controlador queda entonces:

$$
u^{\mathrm{alloc}}_{j,k} = \alpha_k u^{\mathrm{demand}}_{j,k}.
$$

La demanda local no asignada se registra como:

$$
u^{\mathrm{unallocated}}_{j,k}
= u^{\mathrm{demand}}_{j,k} - u^{\mathrm{alloc}}_{j,k}.
$$

## Antiwindup condicional

El sistema implementa antiwindup condicional. Primero se calcula una integral candidata $I^*_{j,k}$. Luego, si no hay saturacion efectiva, la integral avanza. Si existe saturacion, la integral solo avanza cuando la demanda no asignada no amplifica el error:

$$
\left(u^{\mathrm{demand}}_{j,k} - u^{\mathrm{alloc}}_{j,k}\right)e_{j,k} \leq 0.
$$

Esta regla evita que el termino integral crezca cuando el actuador ya no puede entregar la accion demandada en la direccion requerida por el error.

## Aprendizaje por refuerzo discreto

Cada agente de ganancia selecciona una accion:

$$
a \in \{-1,0,+1\},
$$

que significa disminuir, mantener o aumentar una ganancia. El gestor de control transforma esta accion en una variacion de ganancia:

$$
\Delta k = a \delta_k,
$$

y aplica limites:

$$
k^+ = \operatorname{clip}(k + \Delta k,\ k_{\min},\ k_{\max}).
$$

La ganancia efectivamente aplicada se registra en `gain_value_<agent_id>`. El agente utiliza esta senal para identificar la transicion real de estado discreto, en lugar de inferirla desde la accion solicitada. El agente aprende a partir de `reward_for_learning_<agent_id>`. La actualizacion de Q-learning se expresa como:

$$
Q(s,a) \leftarrow Q(s,a) +
\eta \left[
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\right],
$$

donde $\eta$ es la tasa de aprendizaje y $\gamma$ es el factor de descuento. En episodios terminados, el termino futuro se anula.

## Metricas por intervalo

Las metricas se agregan sobre ventanas de pasos fisicos. Para una senal normalizada $z_k$ en el intervalo $[m,n]$, la energia instantanea se define como:

$$
E_k = z_k^2.
$$

El costo agregado normalizado se obtiene como:

$$
c = \operatorname{clip}_{[0,1]}
\left(
\frac{
\frac{1}{n-m+1}\sum_{k=m}^{n} E_k
}{B}
\right),
$$

donde $B$ es el limite de normalizacion declarado para la metrica. Los costos principales son energia del error, variacion del esfuerzo asignado, energia de accion total y variacion de accion total.

Como diagnostico adicional se persiste la amplitud RMS normalizada:

$$
\mathrm{rms\_norm}(z) =
\sqrt{\frac{1}{n-m+1}\sum_{k=m}^{n} z_k^2}.
$$

Las senales `<variable>_rms_norm` permiten observar amplitud promedio, mientras que `cost_<...>_01` conserva la energia media normalizada que compone la recompensa.

## Recompensa

Para cada controlador $j$, el costo local se define como combinacion convexa de componentes de costo configurados:

$$
c^{\mathrm{local}}_j
= \sum_q w_{j,q} c_{j,q},
$$

con $w_{j,q} \geq 0$ y $\sum_q w_{j,q} = 1$. En la configuracion vigente, los componentes son `cost_error_energy_01_<controller>` y `cost_delta_u_01_<controller>`, de modo que cada costo local depende de metricas propias del lazo. La recompensa local firmada es:

$$
r^{\mathrm{local}}_j = 1 - 2c^{\mathrm{local}}_j.
$$

El costo cooperativo combina unicamente los costos locales ya normalizados. El modo se declara en `reward.cooperative.mode`; para el modo `norm-p`, el exponente se declara en `reward.cooperative.p`. Los pesos son no negativos:

$$
\qquad \sum_j \lambda_j = 1.
$$

En modo `norm-p`, con $p \geq 1$, el costo cooperativo es:

$$
c^{\mathrm{coop}}
=
\left(
\sum_j \lambda_j
\left(c^{\mathrm{local}}_j\right)^p
\right)^{1/p}.
$$

Cuando $p=1$, esta expresion equivale a una media ponderada lineal:

$$
c^{\mathrm{coop}} = \sum_j \lambda_j c^{\mathrm{local}}_j.
$$

Cuando $p=2$, equivale a una media RMS ponderada:

$$
c^{\mathrm{coop}}
=
\sqrt{
\sum_j \lambda_j
\left(c^{\mathrm{local}}_j\right)^2
}.
$$

Valores mayores de $p$ incrementan la influencia relativa de los lazos con mayor costo local. La formulacion conserva $c^{\mathrm{coop}} \in [0,1]$ cuando los costos locales pertenecen a $[0,1]$. La cooperacion no introduce metricas cruzadas entre controladores; solo agrega el desempeno local declarado para cada lazo.

La recompensa cooperativa es:

$$
r^{\mathrm{coop}} = 1 - 2c^{\mathrm{coop}}.
$$

La recompensa de controlador se compone como:

$$
r^{\mathrm{controller}}_j =
\operatorname{clip}_{[-1,1]}
\left(
w_l r^{\mathrm{local}}_j
+ w_c r^{\mathrm{coop}}
+ w_s r^{\mathrm{strategic}}
\right).
$$

La recompensa global firmada se define como el promedio de las recompensas compuestas de controlador:

$$
r^{\mathrm{global}} =
\operatorname{clip}_{[-1,1]}
\left(
\frac{1}{J}\sum_{j=1}^{J} r^{\mathrm{controller}}_j
\right).
$$

La senal `cooperative_reward_signed` conserva solo la recompensa cooperativa. Las senales `reward_global_signed` y `global_interval_reward` conservan la recompensa global compuesta. En Release 1, la recompensa estrategica permanece inactiva. La asignacion activa entrega a cada agente de ganancia la recompensa del controlador al que pertenece.

## Terminacion del episodio

El gestor de simulacion evalua la terminacion a partir de senales fisicas crudas. Existen tres condiciones conceptuales:

- exceder limites fisicos declarados;
- alcanzar una region de estabilizacion;
- agotar el tiempo maximo del episodio.

La dinamica no decide la terminacion. Esta responsabilidad pertenece al flujo temporal del `SimulationManager`.

## Persistencia experimental

El sistema persiste cada episodio como un objeto JSON con:

- `step_data`, para trayectorias fisicas y de control;
- `interval_data`, para metricas, recompensas, acciones y aprendizaje;
- `end_episode_data`, para resumen del cierre.

Adicionalmente se generan chunks de dataset, metadata de corrida, resumen tabular, estado del agente y figuras post-run. La persistencia ocurre despues de cerrar el episodio y no forma parte del loop fisico de alta frecuencia.

El cierre de episodio conserva, entre otros campos, la razon de terminacion, el tiempo final, las recompensas acumuladas por controlador y las ganancias finales de cada controlador.

## Documentos relacionados

- [[00_declaracion_descriptiva_general]]
- [[02_manifiesto_reglas_codificacion]]
- [[reports/release_1]]
