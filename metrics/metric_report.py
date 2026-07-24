import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


STEADY_STATE_WINDOW_SEC = 1.0
REWARD_ZERO_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class MetricSpec:
    canonical_name: str
    scientific_label: str
    family: str
    entity_type: str
    entity_id: str
    unit_or_domain: str
    definition: str
    applicability: str
    denominator: str
    recommended_summary: str
    interpretation: str
    number_format: str

    def catalog_row(self):
        return {
            "canonical_name": self.canonical_name,
            "scientific_label": self.scientific_label,
            "family": self.family,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "unit_or_domain": self.unit_or_domain,
            "definition": self.definition,
            "applicability": self.applicability,
            "denominator": self.denominator,
            "recommended_summary": self.recommended_summary,
            "interpretation": self.interpretation,
            "number_format": self.number_format,
        }


CATALOG_COLUMNS = (
    "canonical_name",
    "scientific_label",
    "family",
    "entity_type",
    "entity_id",
    "unit_or_domain",
    "definition",
    "applicability",
    "denominator",
    "recommended_summary",
    "interpretation",
    "number_format",
)

EPISODE_IDENTITY_COLUMNS = (
    "episode_id",
    "n_step",
    "n_interval",
    "effective_duration_sec",
    "termination_reason_code",
    "termination_reason",
)


def _spec(
    canonical_name,
    scientific_label,
    family,
    entity_type,
    entity_id,
    unit_or_domain,
    definition,
    applicability,
    denominator,
    recommended_summary,
    interpretation,
    number_format="0.000000",
):
    return MetricSpec(
        canonical_name=canonical_name,
        scientific_label=scientific_label,
        family=family,
        entity_type=entity_type,
        entity_id=entity_id,
        unit_or_domain=unit_or_domain,
        definition=definition,
        applicability=applicability,
        denominator=denominator,
        recommended_summary=recommended_summary,
        interpretation=interpretation,
        number_format=number_format,
    )


def build_metric_report_specs(config):
    specs = []

    def add(*args, **kwargs):
        specs.append(_spec(*args, **kwargs))

    add(
        "success",
        "Éxito",
        "resultado_estabilidad",
        "global",
        "global",
        "binario",
        "Indicador de terminación por estabilización exitosa.",
        "Todos los episodios terminados.",
        "episodio",
        "media",
        "Mayor es mejor.",
        "0",
    )
    add(
        "failure",
        "Fallo",
        "resultado_estabilidad",
        "global",
        "global",
        "binario",
        "Complemento del indicador de éxito.",
        "Todos los episodios terminados.",
        "episodio",
        "media",
        "Menor es mejor.",
        "0",
    )
    add(
        "stabilization_start_sec",
        "Inicio de estabilización",
        "resultado_estabilidad",
        "global",
        "global",
        "s",
        "Tiempo del primer estado del primer bloque que satisface la persistencia terminal.",
        "Existe un bloque estable de longitud suficiente.",
        "n_step",
        "mediana e IQR",
        "Menor es mejor entre episodios exitosos.",
    )
    add(
        "stable_block_max_sec",
        "Duración estable máxima",
        "resultado_estabilidad",
        "global",
        "global",
        "s",
        "Duración del bloque estable conjunto consecutivo más largo.",
        "Todos los episodios con muestras físicas.",
        "n_step",
        "mediana e IQR",
        "Mayor indica mayor mantenimiento de estabilidad.",
    )
    add(
        "stable_occupancy",
        "Ocupación estable conjunta",
        "resultado_estabilidad",
        "global",
        "global",
        "proporción",
        "Fracción de estados físicos válidos dentro de todas las bandas terminales.",
        "Todos los episodios con muestras físicas.",
        "n_step",
        "media",
        "Mayor es mejor.",
        "0.0000",
    )
    add(
        "joint_stability_loss_count",
        "Pérdidas conjuntas de estabilidad",
        "resultado_estabilidad",
        "global",
        "global",
        "conteo",
        "Transiciones de estabilidad conjunta a inestabilidad posteriores a la primera entrada.",
        "Todos los episodios con muestras físicas.",
        "joint_post_entry_duration_sec",
        "mediana e IQR",
        "Menor es mejor.",
        "0",
    )
    add(
        "joint_post_entry_duration_sec",
        "Tiempo conjunto observado tras la entrada",
        "resultado_estabilidad",
        "global",
        "global",
        "s",
        "Tiempo con oportunidades de pérdida después de la primera entrada conjunta.",
        "Existe al menos una entrada conjunta.",
        "n_step",
        "suma",
        "Denominador de la frecuencia de pérdidas conjuntas.",
    )
    add(
        "joint_stability_loss_rate_hz",
        "Frecuencia de pérdidas conjuntas",
        "resultado_estabilidad",
        "global",
        "global",
        "s^-1",
        "Pérdidas conjuntas divididas por el tiempo observado después de la primera entrada.",
        "Tiempo posterior a la primera entrada mayor que cero.",
        "joint_post_entry_duration_sec",
        "mediana e IQR",
        "Menor es mejor.",
    )
    add(
        "stability_persistence_samples",
        "Muestras de persistencia",
        "resultado_estabilidad",
        "global",
        "global",
        "conteo",
        "Número de muestras físicas exigidas por el criterio terminal de persistencia.",
        "Todos los episodios.",
        "dt_sec",
        "valor único",
        "Parámetro del criterio de éxito.",
        "0",
    )

    objective_units = {"pendulum_angle": "rad", "cart_position": "m"}
    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        objective = controller_config["objective"]
        unit = objective_units[objective]
        squared_unit = f"{unit}^2"
        add(
            f"iae_{controller_id}",
            f"IAE — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            f"{unit}·s",
            "Integral discreta del valor absoluto del error físico y-r.",
            "Todos los episodios con muestras físicas.",
            "effective_duration_sec",
            "mediana e IQR",
            "Menor es mejor; interpretar junto con la duración.",
        )
        add(
            f"ise_{controller_id}",
            f"ISE — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            f"{squared_unit}·s",
            "Integral discreta del cuadrado del error físico y-r.",
            "Todos los episodios con muestras físicas.",
            "effective_duration_sec",
            "mediana e IQR",
            "Menor es mejor; penaliza picos.",
        )
        add(
            f"rms_error_{controller_id}",
            f"RMS del error — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            unit,
            "Raíz de ISE dividida por la duración efectiva.",
            "Duración efectiva mayor que cero.",
            "effective_duration_sec",
            "mediana e IQR",
            "Menor es mejor.",
        )
        add(
            f"max_abs_error_{controller_id}",
            f"Error absoluto máximo — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            unit,
            "Máximo valor absoluto del error físico y-r.",
            "Todos los episodios con muestras físicas.",
            "n_step",
            "mediana e IQR",
            "Menor es mejor.",
        )
        add(
            f"overshoot_{controller_id}",
            f"Sobreimpulso — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            unit,
            "Excursión orientada máxima más allá de la referencia después del primer cruce.",
            "El lazo cruza la referencia o comienza exactamente en ella.",
            "reference_crossed_" + controller_id,
            "mediana e IQR",
            "Menor es mejor.",
        )
        add(
            f"undershoot_{controller_id}",
            f"Subimpulso — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            unit,
            "Excursión orientada máxima hacia el lado inicial después del primer cruce.",
            "El lazo cruza la referencia o comienza exactamente en ella.",
            "reference_crossed_" + controller_id,
            "mediana e IQR",
            "Menor es mejor.",
        )
        add(
            f"reference_crossed_{controller_id}",
            f"Cruce de referencia — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            "binario",
            "Indica si la trayectoria alcanzó o cruzó la referencia orientada.",
            "Todos los episodios con muestras físicas.",
            "episodio",
            "media",
            "Determina la aplicabilidad de sobreimpulso y subimpulso.",
            "0",
        )
        add(
            f"local_stability_loss_count_{controller_id}",
            f"Reapariciones locales — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            "conteo",
            "Transiciones desde la banda local de salida y velocidad hacia fuera de ella.",
            "Todos los episodios con muestras físicas.",
            f"local_post_entry_duration_sec_{controller_id}",
            "mediana e IQR",
            "Menor es mejor.",
            "0",
        )
        add(
            f"local_post_entry_duration_sec_{controller_id}",
            f"Tiempo local observado tras la entrada — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            "s",
            "Tiempo con oportunidades de pérdida después de la primera entrada local.",
            "Existe al menos una entrada local.",
            "n_step",
            "suma",
            "Denominador de la frecuencia de reaparición.",
        )
        add(
            f"local_stability_loss_rate_hz_{controller_id}",
            f"Frecuencia de reaparición — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            "s^-1",
            "Reapariciones locales divididas por el tiempo observado tras la primera entrada.",
            "Tiempo posterior a la primera entrada mayor que cero.",
            f"local_post_entry_duration_sec_{controller_id}",
            "mediana e IQR",
            "Menor es mejor.",
        )
        add(
            f"steady_state_error_signed_{controller_id}",
            f"Error final firmado — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            unit,
            "Media firmada del error físico en la ventana final de 1 s.",
            "La trayectoria contiene la ventana final completa.",
            f"steady_state_window_applicable_{controller_id}",
            "mediana e IQR",
            "Cero indica ausencia de sesgo; en fallos describe error terminal.",
        )
        add(
            f"steady_state_error_abs_{controller_id}",
            f"Error final absoluto — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            unit,
            "Media del valor absoluto del error físico en la ventana final de 1 s.",
            "La trayectoria contiene la ventana final completa.",
            f"steady_state_window_applicable_{controller_id}",
            "mediana e IQR",
            "Menor es mejor; evita cancelación por oscilaciones.",
        )
        add(
            f"steady_state_window_applicable_{controller_id}",
            f"Ventana final aplicable — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            "binario",
            "Indica disponibilidad de la ventana final completa de 1 s.",
            "Todos los episodios.",
            "n_step",
            "media",
            "Control de aplicabilidad.",
            "0",
        )
        add(
            f"steady_state_confirmed_{controller_id}",
            f"Régimen estable confirmado — {controller_id}",
            "seguimiento",
            "lazo",
            controller_id,
            "binario",
            "Indica que el error final pertenece a un episodio con éxito terminal confirmado.",
            "Todos los episodios.",
            "episodio",
            "media",
            "Distingue régimen estacionario confirmado de error terminal.",
            "0",
        )
        add(
            f"max_abs_u_demand_{controller_id}",
            f"Demanda máxima — {controller_id}",
            "control_actuador",
            "controlador",
            controller_id,
            "dominio de control",
            "Máximo valor absoluto de la demanda local en el dominio activo del controlador.",
            "Todos los episodios con muestras de control.",
            "n_step",
            "mediana e IQR",
            "Mayor indica mayor actividad interna del controlador.",
        )

    global_control_specs = (
        ("saturation_count", "Muestras saturadas", "conteo", "Número de oportunidades físicas con saturación global.", "n_step", "Menor es mejor.", "0"),
        ("saturation_frequency", "Frecuencia de saturación", "proporción", "Fracción de oportunidades físicas con saturación global.", "n_step", "Menor es mejor.", "0.0000"),
        ("saturation_duration_sec", "Duración saturada", "s", "Tiempo físico acumulado bajo saturación global.", "n_step", "Menor es mejor.", "0.000000"),
        ("min_pre_saturation_margin", "Margen mínimo previo a saturación", "dominio de control", "Mínimo margen de la demanda total no saturada respecto de ambos límites.", "n_step", "Positivo es holgura; negativo es exceso.", "0.000000"),
        ("cancellation_index", "Índice de cancelación", "proporción", "Uno menos la actividad absoluta neta dividida por la actividad absoluta local.", "cancellation_denominator", "Mayor indica más cancelación interna.", "0.0000"),
        ("cancellation_denominator", "Actividad local absoluta", "dominio de control·s", "Integral de la suma de demandas locales absolutas.", "effective_duration_sec", "Denominador del índice de cancelación.", "0.000000"),
        ("control_effort_u2", "Esfuerzo cuadrático aplicado", "dominio de control^2·s", "Integral de la actuación aplicada al cuadrado.", "effective_duration_sec", "Menor es mejor condicionado al resultado.", "0.000000"),
        ("control_mean_square", "Media cuadrática aplicada", "dominio de control^2", "Esfuerzo cuadrático aplicado dividido por la duración.", "effective_duration_sec", "Menor indica menor intensidad media.", "0.000000"),
        ("demand_effort", "Esfuerzo cuadrático demandado", "dominio de control^2·s", "Integral de la suma de demandas locales al cuadrado.", "effective_duration_sec", "Caracteriza actividad interna conjunta.", "0.000000"),
        ("actuation_energy_abs", "Energía mecánica absoluta", "J", "Integral de |F·v| usando fuerza aplicada y velocidad del carro.", "effective_duration_sec", "Menor es mejor condicionado al resultado.", "0.000000"),
        ("actuation_power_abs_mean", "Potencia mecánica absoluta media", "W", "Energía mecánica absoluta dividida por la duración efectiva.", "effective_duration_sec", "Permite comparar duraciones distintas.", "0.000000"),
        ("control_total_variation", "Variación total de actuación", "dominio de control", "Suma de cambios absolutos consecutivos de la actuación aplicada.", "n_step - 1", "Menor indica actuación más suave.", "0.000000"),
        ("max_abs_u_applied", "Actuación máxima aplicada", "dominio de control", "Máximo valor absoluto de la actuación saturada aplicada.", "n_step", "Menor indica menor pico aplicado.", "0.000000"),
        ("post_stabilization_control_effort_u2", "Esfuerzo posterior a estabilización", "dominio de control^2·s", "Integral de u² desde el inicio de estabilización hasta el final.", "post_stabilization_duration_sec", "Menor es mejor entre episodios exitosos.", "0.000000"),
        ("post_stabilization_actuation_energy_abs", "Energía posterior a estabilización", "J", "Integral de |F·v| desde el inicio de estabilización hasta el final.", "post_stabilization_duration_sec", "Menor es mejor entre episodios exitosos.", "0.000000"),
        ("post_stabilization_duration_sec", "Duración posterior a estabilización", "s", "Duración del tramo desde el inicio estable hasta la terminación.", "success", "Denominador de métricas posteriores a estabilización.", "0.000000"),
    )
    for name, label, unit, definition, denominator, interpretation, number_format in global_control_specs:
        add(
            name,
            label,
            "control_actuador",
            "global",
            "global",
            unit,
            definition,
            "Todos los episodios con muestras físicas, salvo métricas condicionales indicadas.",
            denominator,
            "mediana e IQR",
            interpretation,
            number_format,
        )

    transition_specs = (
        ("health_transition_count", "Transiciones de salud", "conteo", "Número de transiciones entre intervalos válidos.", "n_interval - 1", "Denominador común de progreso y deterioro.", "0"),
        ("concurrent_progress_count", "Progresos concurrentes", "conteo", "Transiciones donde todas las saludes locales aumentan.", "health_transition_count", "Mayor indica progreso simultáneo.", "0"),
        ("concurrent_progress_frequency", "Frecuencia de progreso concurrente", "proporción", "Progresos concurrentes divididos por transiciones de salud.", "health_transition_count", "Mayor indica progreso simultáneo frecuente.", "0.0000"),
        ("deterioration_count", "Transiciones con deterioro", "conteo", "Transiciones donde al menos una salud local disminuye.", "health_transition_count", "Menor es mejor.", "0"),
        ("deterioration_frequency", "Frecuencia de deterioro", "proporción", "Transiciones con deterioro divididas por transiciones de salud.", "health_transition_count", "Menor es mejor.", "0.0000"),
        ("compensatory_transition_count", "Transiciones compensatorias", "conteo", "Mejoras aritméticas agregadas que ocultan deterioro local.", "health_transition_count", "Menor indica cooperación más equilibrada.", "0"),
        ("compensatory_transition_frequency", "Frecuencia compensatoria", "proporción", "Transiciones compensatorias divididas por transiciones de salud.", "health_transition_count", "Menor indica cooperación más equilibrada.", "0.0000"),
        ("worst_health_mean", "Peor salud media", "[0,1]", "Media temporal de la menor salud local de cada intervalo.", "n_interval", "Mayor es mejor.", "0.0000"),
        ("worst_health_min", "Peor salud mínima", "[0,1]", "Mínimo temporal de la menor salud local.", "n_interval", "Mayor es mejor.", "0.0000"),
        ("worst_health_final", "Peor salud final", "[0,1]", "Menor salud local en el último intervalo.", "n_interval", "Mayor es mejor.", "0.0000"),
        ("health_gap_mean", "Brecha aritmético-geométrica media", "[0,1]", "Media de W_A-W_G calculada con los pesos efectivos.", "n_interval", "Menor indica saludes más semejantes.", "0.0000"),
        ("health_gap_max", "Brecha aritmético-geométrica máxima", "[0,1]", "Máximo de W_A-W_G calculado con los pesos efectivos.", "n_interval", "Menor indica menor desigualdad máxima.", "0.0000"),
        ("health_balance_mean", "Balance relativo medio", "[0,1]", "Media temporal de W_G/W_A.", "n_interval", "Más próximo a uno indica mayor equilibrio.", "0.0000"),
        ("health_balance_min", "Balance relativo mínimo", "[0,1]", "Mínimo temporal de W_G/W_A.", "n_interval", "Más próximo a uno indica mayor equilibrio.", "0.0000"),
        ("health_balance_final", "Balance relativo final", "[0,1]", "Valor final de W_G/W_A.", "n_interval", "Más próximo a uno indica mayor equilibrio.", "0.0000"),
    )
    for name, label, unit, definition, denominator, interpretation, number_format in transition_specs:
        add(
            name,
            label,
            "mecanismo_salud",
            "global",
            "global",
            unit,
            definition,
            "Existen intervalos o transiciones suficientes para la reducción.",
            denominator,
            "media o mediana según diseño experimental",
            interpretation,
            number_format,
        )

    reward_specs = (
        ("cooperative_reward_positive_proportion", "Recompensa cooperativa positiva", "proporción", "Fracción de recompensas cooperativas mayores que τr.", "Mayor indica predominio positivo."),
        ("cooperative_reward_zero_proportion", "Recompensa cooperativa nula", "proporción", "Fracción de recompensas cooperativas con |r|≤τr.", "Describe la masa alrededor de cero."),
        ("cooperative_reward_negative_proportion", "Recompensa cooperativa negativa", "proporción", "Fracción de recompensas cooperativas menores que -τr.", "Menor indica menos señales negativas."),
        ("cooperative_reward_variance", "Varianza cooperativa sin ponderar", "recompensa^2", "Varianza poblacional de la recompensa cooperativa ordinaria.", "Describe dispersión sin efecto del peso."),
        ("cooperative_reward_effective_scale", "Escala cooperativa sin ponderar", "recompensa", "RMS de la recompensa cooperativa ordinaria.", "Describe amplitud sin efecto del peso."),
        ("cooperative_contribution_variance", "Varianza de contribución cooperativa", "recompensa^2", "Varianza poblacional de wC·rC.", "Describe dispersión que llegó a la composición."),
        ("cooperative_contribution_effective_scale", "Escala efectiva cooperativa", "recompensa", "RMS de wC·rC.", "Describe amplitud que llegó a la composición."),
    )
    for name, label, unit, definition, interpretation in reward_specs:
        add(
            name,
            label,
            "recompensa",
            "global",
            "cooperativa",
            unit,
            definition,
            "Existe al menos un intervalo válido.",
            "n_interval",
            "media o mediana según diseño experimental",
            interpretation,
            "0.000000",
        )

    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        for stem, label, unit, definition, interpretation in (
            ("local_reward_variance", "Varianza local sin ponderar", "recompensa^2", "Varianza poblacional de la recompensa local ordinaria.", "Describe dinámica local sin efecto del peso."),
            ("local_reward_effective_scale", "Escala local sin ponderar", "recompensa", "RMS de la recompensa local ordinaria.", "Describe amplitud local sin efecto del peso."),
            ("local_contribution_variance", "Varianza de contribución local", "recompensa^2", "Varianza poblacional de wL·rL.", "Describe dispersión que llegó a la composición."),
            ("local_contribution_effective_scale", "Escala efectiva local", "recompensa", "RMS de wL·rL.", "Describe amplitud que llegó a la composición."),
        ):
            add(
                f"{stem}_{controller_id}",
                f"{label} — {controller_id}",
                "recompensa",
                "controlador",
                controller_id,
                unit,
                definition,
                "Existe al menos un intervalo válido.",
                "n_interval",
                "media o mediana según diseño experimental",
                interpretation,
                "0.000000",
            )

    for gain_agent_config in config["agent"]["gain_agents"]:
        agent_id = gain_agent_config["id"]
        agent_specs = (
            ("gain_change_count", "Cambios efectivos de ganancia", "conteo", "Solicitudes con ΔK aplicada distinta de cero.", "n_interval", "Mayor indica mayor actividad de adaptación.", "0"),
            ("gain_change_frequency", "Frecuencia de cambio de ganancia", "proporción", "Cambios efectivos divididos por decisiones.", "n_interval", "Mayor indica adaptación más frecuente.", "0.0000"),
            ("nonzero_gain_request_count", "Solicitudes no nulas", "conteo", "Solicitudes de cambio de ganancia distintas de cero.", "n_interval", "Denominador de acciones bloqueadas.", "0"),
            ("blocked_gain_action_count", "Acciones de ganancia bloqueadas", "conteo", "Solicitudes no nulas cuya variación aplicada difiere de la solicitada.", f"nonzero_gain_request_count_{agent_id}", "Menor es mejor.", "0"),
            ("blocked_gain_action_frequency", "Frecuencia de bloqueo", "proporción", "Acciones bloqueadas divididas por solicitudes no nulas.", f"nonzero_gain_request_count_{agent_id}", "Menor es mejor.", "0.0000"),
            ("gain_initial", "Ganancia inicial", "ganancia PID", "Valor de ganancia al inicio del episodio.", "episodio", "Contexto de la trayectoria.", "0.000000"),
            ("gain_final", "Ganancia final", "ganancia PID", "Valor de ganancia después de la última decisión.", "n_interval", "Resultado final de adaptación.", "0.000000"),
            ("gain_min", "Ganancia mínima", "ganancia PID", "Mínimo valor aplicado durante la trayectoria intervalar.", "n_interval", "Extremo inferior observado.", "0.000000"),
            ("gain_max", "Ganancia máxima", "ganancia PID", "Máximo valor aplicado durante la trayectoria intervalar.", "n_interval", "Extremo superior observado.", "0.000000"),
            ("gain_std", "Desviación temporal de ganancia", "ganancia PID", "Desviación estándar poblacional de la trayectoria intervalar de ganancia.", "n_interval", "Mayor indica mayor dispersión temporal.", "0.000000"),
            ("gain_total_variation", "Variación total de ganancia", "ganancia PID", "Suma de |ΔK aplicada| durante el episodio.", "n_interval", "Mayor indica mayor recorrido acumulado.", "0.000000"),
            ("action_change_count", "Cambios de acción", "conteo", "Transiciones entre acciones discretas distintas.", "n_interval - 1", "Mayor indica política ejecutada menos estable.", "0"),
            ("action_change_frequency", "Frecuencia de cambio de acción", "proporción", "Cambios de acción divididos por transiciones entre intervalos.", "n_interval - 1", "Menor indica acción ejecutada más estable.", "0.0000"),
            ("q_delta_rms", "Variación RMS de Q", "valor Q", "RMS por entrada entre las tablas Q inicial y final del episodio.", "número de entradas Q", "Menor indica menor actualización típica.", "0.000000"),
            ("policy_stability", "Estabilidad de política", "proporción", "Fracción de estados visitados cuyo conjunto de acciones codiciosas no cambia.", f"visited_state_count_{agent_id}", "Mayor indica política codiciosa más estable.", "0.0000"),
            ("visited_state_count", "Estados visitados", "conteo", "Número de estados únicos visitados durante el episodio.", "episodio", "Denominador de estabilidad de política.", "0"),
            ("learning_metrics_applicable", "Métricas de aprendizaje aplicables", "binario", "Indica que el episodio ejecutó selección y aprendizaje del agente.", "episodio", "Control de aplicabilidad.", "0"),
        )
        for stem, label, unit, definition, denominator, interpretation, number_format in agent_specs:
            add(
                f"{stem}_{agent_id}",
                f"{label} — {agent_id}",
                "aprendizaje_ganancia",
                "agente",
                agent_id,
                unit,
                definition,
                "Episodio de entrenamiento, salvo métricas de ganancia que también describen ejecución fija.",
                denominator,
                "media o mediana según diseño experimental",
                interpretation,
                number_format,
            )

    add(
        "q_delta_rms_agent_mean",
        "Variación Q media entre agentes",
        "aprendizaje",
        "global",
        "agentes",
        "valor Q",
        "Media de las variaciones RMS de Q de los agentes.",
        "Episodio de entrenamiento con agentes aplicables.",
        "número de agentes",
        "mediana e IQR",
        "Menor indica menor actualización media.",
    )
    add(
        "policy_stability_agent_mean",
        "Estabilidad de política media entre agentes",
        "aprendizaje",
        "global",
        "agentes",
        "proporción",
        "Media de la estabilidad de política sobre agentes.",
        "Episodio de entrenamiento con agentes aplicables.",
        "número de agentes",
        "media",
        "Mayor indica mayor estabilidad media.",
        "0.0000",
    )

    return tuple(specs)


def build_metric_report_signal_names(config):
    return [spec.canonical_name for spec in build_metric_report_specs(config)]


def _longest_true_run(indicator):
    longest = 0
    current = 0
    for is_true in indicator:
        if bool(is_true):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _first_persistent_start(indicator, persistence_samples):
    current = 0
    for sample_idx, is_true in enumerate(indicator):
        if bool(is_true):
            current += 1
            if current >= persistence_samples:
                return sample_idx - persistence_samples + 1
        else:
            current = 0
    return None


def _loss_metrics(indicator, dt_sec):
    true_indices = np.flatnonzero(indicator)
    if true_indices.size == 0:
        return 0, 0.0, None
    first_entry = int(true_indices[0])
    losses = int(np.sum(indicator[first_entry:-1] & ~indicator[first_entry + 1 :]))
    post_entry_duration_sec = float(max(0, len(indicator) - first_entry - 1) * dt_sec)
    if post_entry_duration_sec == 0.0:
        return losses, post_entry_duration_sec, None
    return losses, post_entry_duration_sec, float(losses / post_entry_duration_sec)


def _population_variance(values):
    mean_value = float(np.mean(values))
    return float(np.mean((values - mean_value) ** 2))


def _effective_scale(values):
    return float(math.sqrt(float(np.mean(values**2))))


def _config_fingerprint(run_config_row):
    scientific_config = dict(run_config_row)
    del scientific_config["run_id"]
    canonical = json.dumps(scientific_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_run_config_row(config, run_id):
    simulation = config["simulation"]
    termination = config["termination"]
    dynamic_system = config["dynamic_system"]
    control = config["control"]
    agent = config["agent"]
    reward = config["reward"]
    stabilization = termination["stabilization"]
    agent_params = agent["params"]
    composition_weights = reward["composition"]["weights"]
    cooperative = reward["cooperative"]
    potential = cooperative["potential"]

    row = {
        "run_id": run_id,
        "simulation_mode": simulation["mode"],
        "n_episodes": int(simulation["n_episodes"]),
        "episode_duration_sec": float(simulation["episode_duration_sec"]),
        "dt_sec": float(simulation["dt_sec"]),
        "decision_interval_sec": float(simulation["decision_interval_sec"]),
        "dynamic_system": dynamic_system["name"],
        "boundary_cart_position_min": float(termination["boundary"]["cart_position"][0]),
        "boundary_cart_position_max": float(termination["boundary"]["cart_position"][1]),
        "boundary_pendulum_angle_min": float(termination["boundary"]["pendulum_angle"][0]),
        "boundary_pendulum_angle_max": float(termination["boundary"]["pendulum_angle"][1]),
        "stabilization_mode": stabilization["mode"],
        "stabilization_persistence_sec": float(stabilization["persistence_sec"]),
        "stabilization_pendulum_angle_min": float(stabilization["pendulum_angle"][0]),
        "stabilization_pendulum_angle_max": float(stabilization["pendulum_angle"][1]),
        "stabilization_pendulum_velocity_min": float(stabilization["pendulum_velocity"][0]),
        "stabilization_pendulum_velocity_max": float(stabilization["pendulum_velocity"][1]),
        "stabilization_cart_position_min": float(stabilization["cart_position"][0]),
        "stabilization_cart_position_max": float(stabilization["cart_position"][1]),
        "stabilization_cart_velocity_min": float(stabilization["cart_velocity"][0]),
        "stabilization_cart_velocity_max": float(stabilization["cart_velocity"][1]),
        "steady_state_window_sec": STEADY_STATE_WINDOW_SEC,
        "reward_zero_tolerance": REWARD_ZERO_TOLERANCE,
        "initial_cart_position": float(dynamic_system["initial_state"]["cart_position"]),
        "initial_cart_velocity": float(dynamic_system["initial_state"]["cart_velocity"]),
        "initial_pendulum_angle": float(dynamic_system["initial_state"]["pendulum_angle"]),
        "initial_pendulum_velocity": float(dynamic_system["initial_state"]["pendulum_velocity"]),
        "control_mixing_policy": control["mixing_policy"],
        "control_objectives_in": control["objectives_in"],
        "global_actuator_min": float(control["global_actuator_limits"][0]),
        "global_actuator_max": float(control["global_actuator_limits"][1]),
        "agent_type": agent["type"],
        "agent_algorithm": agent_params["algorithm"],
        "discount_factor": float(agent_params["discount_factor"]),
        "epsilon_initial": float(agent_params["epsilon"]),
        "epsilon_decay_enabled": bool(agent_params["epsilon_decay"]["enabled"]),
        "epsilon_decay_factor": float(agent_params["epsilon_decay"]["decay_factor"]),
        "epsilon_min": float(agent_params["epsilon_decay"]["epsilon_min"]),
        "learning_rate_initial": float(agent_params["learning_rate"]),
        "learning_rate_decay_enabled": bool(agent_params["learning_rate_decay"]["enabled"]),
        "learning_rate_decay_factor": float(agent_params["learning_rate_decay"]["decay_factor"]),
        "learning_rate_min": float(agent_params["learning_rate_decay"]["learning_rate_min"]),
        "reward_assignment": reward["assignment"],
        "reward_domain": reward["domain"],
        "reward_weight_local": float(composition_weights["local"]),
        "reward_weight_cooperative": float(composition_weights["cooperative"]),
        "reward_weight_strategic": float(composition_weights["strategic"]),
        "cooperative_enabled": bool(cooperative["enabled"]),
        "cooperative_reward_mode": cooperative["reward_mode"],
        "cooperative_method": potential["method"],
        "cooperative_residual_weight": float(cooperative["residual_weight"]),
    }

    for parameter_name, parameter_value in dynamic_system["params"].items():
        row[f"dynamic_{parameter_name}"] = float(parameter_value)
    normalization = dynamic_system["state_normalization"]
    row["state_normalization_output_min"] = float(normalization["output_limits"][0])
    row["state_normalization_output_max"] = float(normalization["output_limits"][1])
    for state_name, state_range in normalization["ranges"].items():
        row[f"state_normalization_{state_name}_min"] = float(state_range[0])
        row[f"state_normalization_{state_name}_max"] = float(state_range[1])

    for controller_config in control["controllers"]:
        controller_id = controller_config["id"]
        row[f"controller_objective_{controller_id}"] = controller_config["objective"]
        row[f"controller_setpoint_raw_{controller_id}"] = float(controller_config["setpoint_raw"])
        row[f"controller_setpoint_norm_{controller_id}"] = float(controller_config["setpoint_norm"])
        row[f"controller_error_sign_{controller_id}"] = controller_config["error_sign"]
        row[f"controller_calculation_mode_{controller_id}"] = controller_config["calculation_mode"]
        row[f"controller_antiwindup_{controller_id}"] = controller_config["antiwindup"]["method"]
        for gain_name in ("kp", "ki", "kd"):
            row[f"initial_{gain_name}_{controller_id}"] = float(controller_config["initial_gains"][gain_name])
            row[f"min_{gain_name}_{controller_id}"] = float(controller_config["gain_update"][gain_name]["min"])
            row[f"max_{gain_name}_{controller_id}"] = float(controller_config["gain_update"][gain_name]["max"])
            row[f"delta_{gain_name}_{controller_id}"] = float(
                controller_config["gain_update"][gain_name]["delta_gain"]
            )

    for local_config in reward["local"]["controllers"]:
        controller_id = local_config["controller"]
        for component in local_config["components"]:
            row[f"local_reward_weight_{controller_id}_{component['cost']}"] = float(component["weight"])

    for variable_config in potential["variables"]:
        variable_id = variable_config["id"]
        row[f"cooperative_source_{variable_id}"] = variable_config["source"]
        row[f"cooperative_weight_{variable_id}"] = float(variable_config["weight"])

    row["cooperative_norm_order"] = float(potential["p"]) if potential["method"] == "norm-p" else None
    row["cooperative_geometric_epsilon"] = (
        float(potential["epsilon_geom_conc"])
        if potential["method"] == "geometric_consensus"
        else None
    )
    row["cooperative_deterioration_order"] = (
        float(potential["deterioration_order"])
        if potential["method"] == "concurrent_progress"
        else None
    )

    adaptive_weights = cooperative["adaptive_weights"]
    row["adaptive_cooperative_weights_enabled"] = bool(adaptive_weights["enabled"])
    row["adaptive_weight_floor"] = (
        float(adaptive_weights["weight_floor"]) if adaptive_weights["enabled"] else None
    )
    row["adaptive_deterioration_weight"] = (
        float(adaptive_weights["deterioration_weight"]) if adaptive_weights["enabled"] else None
    )
    row["adaptive_weight_smoothing_alpha"] = (
        float(adaptive_weights["smoothing_alpha"]) if adaptive_weights["enabled"] else None
    )
    for variable_config in adaptive_weights["need_variables"]:
        variable_id = variable_config["id"]
        row[f"adaptive_need_source_{variable_id}"] = (
            variable_config["source"] if adaptive_weights["enabled"] else None
        )

    balance = cooperative["balance"]
    row["cooperative_balance_enabled"] = bool(balance["enabled"])
    row["cooperative_balance_strength"] = float(balance["strength"]) if balance["enabled"] else None
    for variable_config in balance["health_variables"]:
        variable_id = variable_config["id"]
        row[f"cooperative_balance_source_{variable_id}"] = (
            variable_config["source"] if balance["enabled"] else None
        )

    synchrony = cooperative["synchrony"]
    row["cooperative_synchrony_enabled"] = bool(synchrony["enabled"])
    row["cooperative_synchrony_source"] = synchrony["source"] if synchrony["enabled"] else None
    row["cooperative_synchrony_strength"] = (
        float(synchrony["strength"]) if synchrony["enabled"] else None
    )

    adaptive_residual = cooperative["adaptive_residual"]
    row["adaptive_residual_enabled"] = bool(adaptive_residual["enabled"])
    row["adaptive_residual_source"] = (
        adaptive_residual["source"] if adaptive_residual["enabled"] else None
    )
    row["adaptive_residual_weight_min"] = (
        float(adaptive_residual["residual_weight_min"]) if adaptive_residual["enabled"] else None
    )
    row["adaptive_residual_weight_max"] = (
        float(adaptive_residual["residual_weight_max"]) if adaptive_residual["enabled"] else None
    )
    row["adaptive_residual_gamma"] = (
        float(adaptive_residual["gamma"]) if adaptive_residual["enabled"] else None
    )

    for gain_agent_config in agent["gain_agents"]:
        agent_id = gain_agent_config["id"]
        row[f"agent_controller_{agent_id}"] = gain_agent_config["controller"]
        row[f"agent_gain_{agent_id}"] = gain_agent_config["gain"]

    terminal_bonus = reward["terminal_learning_bonus"]
    row["terminal_bonus_enabled"] = bool(terminal_bonus["enabled"])
    for reason_name, bonus in terminal_bonus["bonuses"].items():
        row[f"terminal_bonus_{reason_name}"] = float(bonus)

    row["config_id"] = _config_fingerprint(row)
    ordered_row = {"run_id": row["run_id"], "config_id": row["config_id"]}
    for field_name, value in row.items():
        if field_name not in ("run_id", "config_id"):
            ordered_row[field_name] = value
    return ordered_row


class MetricReport:
    def __init__(self, runtime, config, agent, run_id):
        self.step = runtime.step
        self.interval = runtime.interval
        self.episode = runtime.episode
        self.config = config
        self.agent = agent
        self.dt_sec = float(config["simulation"]["dt_sec"])
        self.controllers = tuple(config["control"]["controllers"])
        self.gain_agents = tuple(config["agent"]["gain_agents"])
        self.metric_specs = build_metric_report_specs(config)
        self.metric_signal_names = tuple(spec.canonical_name for spec in self.metric_specs)
        self.episode_columns = EPISODE_IDENTITY_COLUMNS + self.metric_signal_names
        self.catalog_columns = CATALOG_COLUMNS
        self.catalog_rows = tuple(spec.catalog_row() for spec in self.metric_specs)
        self.run_config_row = build_run_config_row(config, run_id)
        self.run_config_columns = tuple(self.run_config_row)

    def compute_episode(self, episode_idx, n_steps, n_intervals, termination_reason_code, training_enabled):
        del episode_idx
        values = {}
        success_code = int(self.config["termination"]["reason_codes"]["stabilization_success"])
        success = int(termination_reason_code) == success_code
        values["success"] = int(success)
        values["failure"] = int(not success)

        stabilization = self.config["termination"]["stabilization"]
        if stabilization["mode"] == "instantaneous":
            persistence_samples = 1
        else:
            persistence_samples = int(math.ceil(float(stabilization["persistence_sec"]) / self.dt_sec))
        values["stability_persistence_samples"] = persistence_samples

        state_slice = slice(1, n_steps + 1)
        control_slice = slice(0, n_steps)
        interval_slice = slice(0, n_intervals)
        duration_sec = float(n_steps * self.dt_sec)

        local_stability = {
            "pendulum_angle": self._stability_indicator(
                self.step.pendulum_angle_raw[state_slice],
                self.step.pendulum_velocity_raw[state_slice],
                stabilization["pendulum_angle"],
                stabilization["pendulum_velocity"],
            ),
            "cart_position": self._stability_indicator(
                self.step.cart_position_raw[state_slice],
                self.step.cart_velocity_raw[state_slice],
                stabilization["cart_position"],
                stabilization["cart_velocity"],
            ),
        }
        joint_stability = local_stability["pendulum_angle"] & local_stability["cart_position"]
        stable_start_idx = _first_persistent_start(joint_stability, persistence_samples)
        if stable_start_idx is None:
            values["stabilization_start_sec"] = None
        else:
            values["stabilization_start_sec"] = float(self.step.t_sec[stable_start_idx + 1])
        values["stable_block_max_sec"] = float(_longest_true_run(joint_stability) * self.dt_sec)
        values["stable_occupancy"] = float(np.mean(joint_stability))
        joint_losses, joint_duration, joint_rate = _loss_metrics(joint_stability, self.dt_sec)
        values["joint_stability_loss_count"] = joint_losses
        values["joint_post_entry_duration_sec"] = joint_duration
        values["joint_stability_loss_rate_hz"] = joint_rate

        steady_state_samples = int(math.ceil(STEADY_STATE_WINDOW_SEC / self.dt_sec))
        control_domain = self.config["control"]["objectives_in"]
        demand_arrays = []
        for controller_config in self.controllers:
            controller_id = controller_config["id"]
            objective = controller_config["objective"]
            error = np.asarray(getattr(self.step, f"error_{controller_id}_raw")[control_slice], dtype=np.float64)
            if controller_config["error_sign"] == "setpoint_minus_measurement":
                error = -error
            iae = float(np.sum(np.abs(error)) * self.dt_sec)
            ise = float(np.sum(error**2) * self.dt_sec)
            values[f"iae_{controller_id}"] = iae
            values[f"ise_{controller_id}"] = ise
            values[f"rms_error_{controller_id}"] = float(math.sqrt(ise / duration_sec))
            values[f"max_abs_error_{controller_id}"] = float(np.max(np.abs(error)))

            measurement = np.asarray(getattr(self.step, f"{objective}_raw")[control_slice], dtype=np.float64)
            setpoint = float(controller_config["setpoint_raw"])
            overshoot, undershoot, crossed = self._overshoot_undershoot(measurement, setpoint)
            values[f"overshoot_{controller_id}"] = overshoot
            values[f"undershoot_{controller_id}"] = undershoot
            values[f"reference_crossed_{controller_id}"] = crossed

            indicator = local_stability[objective]
            losses, post_entry_duration, loss_rate = _loss_metrics(indicator, self.dt_sec)
            values[f"local_stability_loss_count_{controller_id}"] = losses
            values[f"local_post_entry_duration_sec_{controller_id}"] = post_entry_duration
            values[f"local_stability_loss_rate_hz_{controller_id}"] = loss_rate

            steady_state_applicable = n_steps >= steady_state_samples
            values[f"steady_state_window_applicable_{controller_id}"] = int(steady_state_applicable)
            values[f"steady_state_confirmed_{controller_id}"] = int(success)
            if steady_state_applicable:
                final_error = error[-steady_state_samples:]
                values[f"steady_state_error_signed_{controller_id}"] = float(np.mean(final_error))
                values[f"steady_state_error_abs_{controller_id}"] = float(np.mean(np.abs(final_error)))
            else:
                values[f"steady_state_error_signed_{controller_id}"] = None
                values[f"steady_state_error_abs_{controller_id}"] = None

            demand = np.asarray(
                getattr(self.step, f"u_demand_{controller_id}_{control_domain}")[control_slice],
                dtype=np.float64,
            )
            demand_arrays.append(demand)
            values[f"max_abs_u_demand_{controller_id}"] = float(np.max(np.abs(demand)))

        u_total_raw = np.asarray(self.step.u_total_raw[control_slice], dtype=np.float64)
        u_total = np.asarray(self.step.u_total[control_slice], dtype=np.float64)
        saturated = np.asarray(self.step.is_saturated_global[control_slice], dtype=np.float64) != 0.0
        saturation_count = int(np.sum(saturated))
        values["saturation_count"] = saturation_count
        values["saturation_frequency"] = float(saturation_count / n_steps)
        values["saturation_duration_sec"] = float(saturation_count * self.dt_sec)
        actuator_limits = self.config["control"]["global_actuator_limits"]
        lower_limit = float(actuator_limits[0])
        upper_limit = float(actuator_limits[1])
        values["min_pre_saturation_margin"] = float(
            np.min(np.minimum(upper_limit - u_total_raw, u_total_raw - lower_limit))
        )

        demand_matrix = np.vstack(demand_arrays)
        cancellation_denominator = float(np.sum(np.abs(demand_matrix)) * self.dt_sec)
        values["cancellation_denominator"] = cancellation_denominator
        if cancellation_denominator == 0.0:
            values["cancellation_index"] = None
        else:
            values["cancellation_index"] = float(
                1.0 - np.sum(np.abs(u_total_raw)) * self.dt_sec / cancellation_denominator
            )

        control_effort = float(np.sum(u_total**2) * self.dt_sec)
        values["control_effort_u2"] = control_effort
        values["control_mean_square"] = float(control_effort / duration_sec)
        values["demand_effort"] = float(np.sum(demand_matrix**2) * self.dt_sec)
        force = np.asarray(self.step.cart_force[1 : n_steps + 1], dtype=np.float64)
        cart_velocity = np.asarray(self.step.cart_velocity_raw[1 : n_steps + 1], dtype=np.float64)
        mechanical_power_abs = np.abs(force * cart_velocity)
        actuation_energy = float(np.sum(mechanical_power_abs) * self.dt_sec)
        values["actuation_energy_abs"] = actuation_energy
        values["actuation_power_abs_mean"] = float(actuation_energy / duration_sec)
        values["control_total_variation"] = float(np.sum(np.abs(np.diff(u_total))))
        values["max_abs_u_applied"] = float(np.max(np.abs(u_total)))

        if success and stable_start_idx is not None:
            post_start_control_idx = stable_start_idx + 1
            post_u_total = u_total[post_start_control_idx:]
            post_power = mechanical_power_abs[post_start_control_idx:]
            values["post_stabilization_control_effort_u2"] = float(np.sum(post_u_total**2) * self.dt_sec)
            values["post_stabilization_actuation_energy_abs"] = float(np.sum(post_power) * self.dt_sec)
            values["post_stabilization_duration_sec"] = float(len(post_u_total) * self.dt_sec)
        else:
            values["post_stabilization_control_effort_u2"] = None
            values["post_stabilization_actuation_energy_abs"] = None
            values["post_stabilization_duration_sec"] = None

        self._compute_health_metrics(values, interval_slice, n_intervals)
        self._compute_reward_metrics(values, interval_slice, n_intervals)
        self._compute_agent_metrics(values, interval_slice, n_intervals, training_enabled)

        for signal_name in self.metric_signal_names:
            setattr(self.episode, signal_name, values[signal_name])

    @staticmethod
    def _stability_indicator(output, velocity, output_band, velocity_band):
        return (
            (output >= float(output_band[0]))
            & (output <= float(output_band[1]))
            & (velocity >= float(velocity_band[0]))
            & (velocity <= float(velocity_band[1]))
        )

    @staticmethod
    def _overshoot_undershoot(measurement, setpoint):
        initial_displacement = setpoint - float(measurement[0])
        error = measurement - setpoint
        if initial_displacement == 0.0:
            return float(max(0.0, np.max(error))), float(max(0.0, np.max(-error))), 1

        orientation = math.copysign(1.0, initial_displacement)
        oriented = orientation * error
        crossing_indices = np.flatnonzero(oriented >= 0.0)
        if crossing_indices.size == 0:
            return None, None, 0
        crossed = oriented[int(crossing_indices[0]) :]
        overshoot = float(max(0.0, np.max(crossed)))
        undershoot = float(max(0.0, np.max(-crossed)))
        return overshoot, undershoot, 1

    def _compute_health_metrics(self, values, interval_slice, n_intervals):
        variable_configs = self.config["reward"]["cooperative"]["potential"]["variables"]
        health_rows = []
        weight_rows = []
        for variable_config in variable_configs:
            variable_id = variable_config["id"]
            health_rows.append(np.asarray(getattr(self.interval, variable_config["source"])[interval_slice]))
            weight_rows.append(
                np.asarray(getattr(self.interval, f"cooperative_weight_effective_{variable_id}")[interval_slice])
            )
        health = np.vstack(health_rows)
        weights = np.vstack(weight_rows)
        transition_count = max(0, n_intervals - 1)
        values["health_transition_count"] = transition_count
        if transition_count > 0:
            health_delta = health[:, 1:] - health[:, :-1]
            concurrent = np.all(health_delta > 0.0, axis=0)
            deterioration = np.any(health_delta < 0.0, axis=0)
        else:
            concurrent = np.asarray([], dtype=bool)
            deterioration = np.asarray([], dtype=bool)

        arithmetic_health = np.sum(weights * health, axis=0)
        geometric_health = np.prod(health**weights, axis=0)
        health_gap = arithmetic_health - geometric_health
        health_balance = np.ones(n_intervals, dtype=np.float64)
        nonzero_arithmetic = arithmetic_health != 0.0
        health_balance[nonzero_arithmetic] = geometric_health[nonzero_arithmetic] / arithmetic_health[nonzero_arithmetic]
        worst_health = np.min(health, axis=0)

        if transition_count > 0:
            arithmetic_delta = arithmetic_health[1:] - arithmetic_health[:-1]
            compensatory = (arithmetic_delta > 0.0) & deterioration
            concurrent_count = int(np.sum(concurrent))
            deterioration_count = int(np.sum(deterioration))
            compensatory_count = int(np.sum(compensatory))
            values["concurrent_progress_count"] = concurrent_count
            values["concurrent_progress_frequency"] = float(concurrent_count / transition_count)
            values["deterioration_count"] = deterioration_count
            values["deterioration_frequency"] = float(deterioration_count / transition_count)
            values["compensatory_transition_count"] = compensatory_count
            values["compensatory_transition_frequency"] = float(compensatory_count / transition_count)
        else:
            values["concurrent_progress_count"] = 0
            values["concurrent_progress_frequency"] = None
            values["deterioration_count"] = 0
            values["deterioration_frequency"] = None
            values["compensatory_transition_count"] = 0
            values["compensatory_transition_frequency"] = None

        values["worst_health_mean"] = float(np.mean(worst_health))
        values["worst_health_min"] = float(np.min(worst_health))
        values["worst_health_final"] = float(worst_health[-1])
        values["health_gap_mean"] = float(np.mean(health_gap))
        values["health_gap_max"] = float(np.max(health_gap))
        values["health_balance_mean"] = float(np.mean(health_balance))
        values["health_balance_min"] = float(np.min(health_balance))
        values["health_balance_final"] = float(health_balance[-1])

    def _compute_reward_metrics(self, values, interval_slice, n_intervals):
        cooperative_reward = np.asarray(self.interval.cooperative_reward_signed[interval_slice], dtype=np.float64)
        values["cooperative_reward_positive_proportion"] = float(
            np.sum(cooperative_reward > REWARD_ZERO_TOLERANCE) / n_intervals
        )
        values["cooperative_reward_zero_proportion"] = float(
            np.sum(np.abs(cooperative_reward) <= REWARD_ZERO_TOLERANCE) / n_intervals
        )
        values["cooperative_reward_negative_proportion"] = float(
            np.sum(cooperative_reward < -REWARD_ZERO_TOLERANCE) / n_intervals
        )
        values["cooperative_reward_variance"] = _population_variance(cooperative_reward)
        values["cooperative_reward_effective_scale"] = _effective_scale(cooperative_reward)

        composition_weights = self.config["reward"]["composition"]["weights"]
        local_weight = float(composition_weights["local"])
        cooperative_weight = float(composition_weights["cooperative"])
        cooperative_contribution = cooperative_weight * cooperative_reward
        values["cooperative_contribution_variance"] = _population_variance(cooperative_contribution)
        values["cooperative_contribution_effective_scale"] = _effective_scale(cooperative_contribution)

        for controller_config in self.controllers:
            controller_id = controller_config["id"]
            local_reward = np.asarray(
                getattr(self.interval, f"local_reward_signed_{controller_id}")[interval_slice],
                dtype=np.float64,
            )
            local_contribution = local_weight * local_reward
            values[f"local_reward_variance_{controller_id}"] = _population_variance(local_reward)
            values[f"local_reward_effective_scale_{controller_id}"] = _effective_scale(local_reward)
            values[f"local_contribution_variance_{controller_id}"] = _population_variance(local_contribution)
            values[f"local_contribution_effective_scale_{controller_id}"] = _effective_scale(local_contribution)

    def _compute_agent_metrics(self, values, interval_slice, n_intervals, training_enabled):
        learning_by_agent = {}
        if training_enabled:
            learning_by_agent = {
                item["agent_id"]: item for item in self.agent.episode_learning_metrics()
            }

        q_delta_values = []
        policy_stability_values = []
        controller_config_by_id = {}
        for controller_config in self.controllers:
            controller_config_by_id[controller_config["id"]] = controller_config

        for gain_agent_config in self.gain_agents:
            agent_id = gain_agent_config["id"]
            requested = np.asarray(
                getattr(self.interval, f"delta_gain_requested_{agent_id}")[interval_slice],
                dtype=np.float64,
            )
            applied = np.asarray(
                getattr(self.interval, f"delta_gain_applied_{agent_id}")[interval_slice],
                dtype=np.float64,
            )
            gain_values = np.asarray(
                getattr(self.interval, f"gain_value_{agent_id}")[interval_slice],
                dtype=np.float64,
            )
            changed = applied != 0.0
            nonzero_requested = requested != 0.0
            blocked = nonzero_requested & (applied != requested)
            change_count = int(np.sum(changed))
            nonzero_request_count = int(np.sum(nonzero_requested))
            blocked_count = int(np.sum(blocked))
            values[f"gain_change_count_{agent_id}"] = change_count
            values[f"gain_change_frequency_{agent_id}"] = float(change_count / n_intervals)
            values[f"nonzero_gain_request_count_{agent_id}"] = nonzero_request_count
            values[f"blocked_gain_action_count_{agent_id}"] = blocked_count
            if nonzero_request_count == 0:
                values[f"blocked_gain_action_frequency_{agent_id}"] = None
            else:
                values[f"blocked_gain_action_frequency_{agent_id}"] = float(
                    blocked_count / nonzero_request_count
                )
            controller_config = controller_config_by_id[gain_agent_config["controller"]]
            values[f"gain_initial_{agent_id}"] = float(
                controller_config["initial_gains"][gain_agent_config["gain"]]
            )
            values[f"gain_final_{agent_id}"] = float(gain_values[-1])
            values[f"gain_min_{agent_id}"] = float(np.min(gain_values))
            values[f"gain_max_{agent_id}"] = float(np.max(gain_values))
            values[f"gain_std_{agent_id}"] = float(np.std(gain_values))
            values[f"gain_total_variation_{agent_id}"] = float(np.sum(np.abs(applied)))

            values[f"learning_metrics_applicable_{agent_id}"] = int(training_enabled)
            if training_enabled:
                actions = np.asarray(
                    getattr(self.interval, f"action_{agent_id}")[interval_slice],
                    dtype=np.int64,
                )
                action_change_count = int(np.sum(actions[1:] != actions[:-1]))
                values[f"action_change_count_{agent_id}"] = action_change_count
                if n_intervals > 1:
                    values[f"action_change_frequency_{agent_id}"] = float(
                        action_change_count / (n_intervals - 1)
                    )
                else:
                    values[f"action_change_frequency_{agent_id}"] = None
                learning = learning_by_agent[agent_id]
                values[f"q_delta_rms_{agent_id}"] = learning["q_delta_rms"]
                values[f"policy_stability_{agent_id}"] = learning["policy_stability"]
                values[f"visited_state_count_{agent_id}"] = learning["visited_state_count"]
                q_delta_values.append(learning["q_delta_rms"])
                policy_stability_values.append(learning["policy_stability"])
            else:
                values[f"action_change_count_{agent_id}"] = None
                values[f"action_change_frequency_{agent_id}"] = None
                values[f"q_delta_rms_{agent_id}"] = None
                values[f"policy_stability_{agent_id}"] = None
                values[f"visited_state_count_{agent_id}"] = None

        if training_enabled:
            values["q_delta_rms_agent_mean"] = float(np.mean(q_delta_values))
            values["policy_stability_agent_mean"] = float(np.mean(policy_stability_values))
        else:
            values["q_delta_rms_agent_mean"] = None
            values["policy_stability_agent_mean"] = None
