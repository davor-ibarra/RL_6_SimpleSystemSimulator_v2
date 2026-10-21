import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


METRIC_REGISTRY_VERSION = "2.4.0"
METRIC_SOURCE_SEPARATOR = " | "

_COMPONENT_STAT_NAMES = (
    "mean",
    "variance",
    "rms",
    "abs_mean",
    "positive_proportion",
    "negative_proportion",
    "zero_proportion",
)


def _sources(*source_names):
    if len(source_names) == 0:
        raise ValueError("A metric must declare at least one canonical source")
    for source_name in source_names:
        if not isinstance(source_name, str) or source_name == "":
            raise ValueError("Canonical metric sources must be non-empty strings")
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"Repeated canonical metric source: {source_names}")
    return tuple(source_names)


def _controller_config(config, controller_id):
    for controller_config in config["control"]["controllers"]:
        if controller_config["id"] == controller_id:
            return controller_config
    raise KeyError(f"Unknown metric controller source: {controller_id}")


def _gain_agent_config(config, agent_id):
    for gain_agent_config in config["agent"]["gain_agents"]:
        if gain_agent_config["id"] == agent_id:
            return gain_agent_config
    raise KeyError(f"Unknown metric agent source: {agent_id}")


def _metric_stem(canonical_name, entity_id):
    suffix = f"_{entity_id}"
    if not canonical_name.endswith(suffix):
        raise KeyError(
            f"Metric {canonical_name} does not use its canonical entity suffix {suffix}"
        )
    return canonical_name[: -len(suffix)]


def _controller_state_sources(config, controller_id):
    controller_config = _controller_config(config, controller_id)
    return _sources(
        *(f"step.{variable}_raw" for variable in controller_config["vars"]),
        *(f"config.termination.stabilization.{variable}" for variable in controller_config["vars"]),
    )


def _controller_demand_sources(config):
    control_domain = config["control"]["objectives_in"]
    return _sources(
        *(
            f"step.u_demand_{controller_config['id']}_{control_domain}"
            for controller_config in config["control"]["controllers"]
        )
    )


def _local_health_sources():
    return _sources(
        "interval.local_health_01_pendulum_angle",
        "interval.local_health_01_cart_position",
    )


def _joint_state_sources():
    return _sources(
        "step.pendulum_angle_raw",
        "step.pendulum_velocity_raw",
        "step.cart_position_raw",
        "step.cart_velocity_raw",
        "config.termination.stabilization.pendulum_angle",
        "config.termination.stabilization.pendulum_velocity",
        "config.termination.stabilization.cart_position",
        "config.termination.stabilization.cart_velocity",
    )


def _controller_reward_sources():
    return _sources(
        "interval.reward_controller_pendulum_angle",
        "interval.reward_controller_cart_position",
    )


def _resolve_agent_metric_sources(canonical_name, family, agent_id, config):
    stem = _metric_stem(canonical_name, agent_id)
    if family == "acciones":
        if stem in (
            "action_decrease_frequency",
            "action_maintain_frequency",
            "action_increase_frequency",
            "action_entropy",
        ):
            return _sources(f"interval.action_{agent_id}")
        if stem == "exploratory_action_frequency":
            return _sources(f"interval.is_exploratory_{agent_id}")

    if family == "aprendizaje_ganancia":
        if stem in ("gain_change_count", "gain_change_frequency", "gain_total_variation"):
            return _sources(f"interval.delta_gain_applied_{agent_id}")
        if stem == "nonzero_gain_request_count":
            return _sources(f"interval.delta_gain_requested_{agent_id}")
        if stem in ("blocked_gain_action_count", "blocked_gain_action_frequency"):
            return _sources(
                f"interval.delta_gain_requested_{agent_id}",
                f"interval.delta_gain_applied_{agent_id}",
            )
        if stem == "gain_initial":
            gain_agent_config = _gain_agent_config(config, agent_id)
            controller_id = gain_agent_config["controller"]
            gain_name = gain_agent_config["gain"]
            return _sources(
                f"config.control.controllers[id={controller_id}].initial_gains.{gain_name}"
            )
        if stem in ("gain_final", "gain_min", "gain_max", "gain_std"):
            return _sources(f"interval.gain_value_{agent_id}")
        if stem in ("action_change_count", "action_change_frequency"):
            return _sources(f"interval.action_{agent_id}")
        if stem in (
            "q_delta_rms",
            "policy_stability",
            "visited_state_count",
        ):
            return _sources(f"agent.episode_learning_metrics[{agent_id}].{stem}")
        if stem == "learning_metrics_applicable":
            return _sources("config.simulation.mode")

    if family == "aprendizaje":
        if stem in ("td_error_mean_abs", "td_error_rms", "td_error_max_abs"):
            return _sources(f"interval.td_error_{agent_id}")
        if stem in (
            "q_value_mean",
            "q_value_std",
            "q_value_min",
            "q_value_max",
            "visited_state_frequency",
        ):
            return _sources(f"agent.episode_learning_metrics[{agent_id}].{stem}")

    if family == "bono_terminal":
        controller_id = _gain_agent_config(config, agent_id)["controller"]
        if stem == "ordinary_learning_return":
            return _sources(
                f"interval.reward_for_learning_{agent_id}",
                f"interval.terminal_bonus_{controller_id}",
            )
        if stem == "total_learning_return":
            return _sources(f"interval.reward_for_learning_{agent_id}")
        if stem in (
            "terminal_bonus_abs_return_fraction",
            "terminal_bonus_separated_abs_return_fraction",
        ):
            return _sources(
                f"interval.reward_for_learning_{agent_id}",
                f"interval.terminal_bonus_{controller_id}",
            )
        if stem == "terminal_bonus_discounted_abs_return_fraction":
            return _sources(
                f"interval.reward_for_learning_{agent_id}",
                f"interval.terminal_bonus_{controller_id}",
                "config.agent.params.discount_factor",
            )

    raise KeyError(
        f"No canonical sources declared for agent metric {canonical_name} ({family})"
    )


def _resolve_controller_metric_sources(
    canonical_name, family, controller_id, config
):
    stem = _metric_stem(canonical_name, controller_id)
    controller_config = _controller_config(config, controller_id)
    objective = controller_config["objective"]

    if family == "bono_terminal":
        return _sources(f"interval.{stem.removesuffix('_final')}_{controller_id}")

    if family == "seguimiento":
        error_source = f"step.error_{controller_id}_raw"
        if stem in ("iae", "ise", "rms_error", "mean_abs_error"):
            return _sources(error_source, "config.simulation.dt_sec")
        if stem in ("max_abs_error", "terminal_error_signed", "terminal_error_abs"):
            return _sources(error_source)
        if stem in ("overshoot", "undershoot", "reference_crossed"):
            return _sources(
                f"step.{objective}_raw",
                f"config.control.controllers[id={controller_id}].setpoint_raw",
            )
        if stem in (
            "local_stability_loss_count",
            "local_post_entry_duration_sec",
            "local_stability_loss_rate_hz",
        ):
            state_sources = _controller_state_sources(config, controller_id)
            if stem == "local_stability_loss_count":
                return state_sources
            return _sources(*state_sources, "config.simulation.dt_sec")
        if stem in (
            "steady_state_error_signed",
            "steady_state_error_abs",
            "steady_state_window_applicable",
        ):
            return _sources(
                error_source,
                "config.metrics.episode_report.steady_state_window_sec",
                "config.simulation.dt_sec",
            )
        if stem == "steady_state_confirmed":
            return _sources(
                "episode.end_termination_reason_code",
                "config.termination.reason_codes.stabilization_success",
            )

    if family == "resultado_estabilidad" and stem == "local_stable_occupancy":
        return _controller_state_sources(config, controller_id)

    if family == "control_actuador" and stem == "max_abs_u_demand":
        control_domain = config["control"]["objectives_in"]
        return _sources(f"step.u_demand_{controller_id}_{control_domain}")

    if family == "interaccion_componentes":
        if stem == "local_cooperative_contribution_covariance":
            return _sources(
                f"interval.local_reward_signed_{controller_id}",
                f"interval.cooperative_contribution_signed_{controller_id}",
                "config.reward.composition.weights.local",
            )
        if stem in (
            "local_cooperative_opposite_sign_frequency",
            "local_cooperative_same_sign_frequency",
        ):
            return _sources(
                f"interval.local_reward_signed_{controller_id}",
                f"interval.cooperative_contribution_signed_{controller_id}",
                "config.reward.composition.weights.local",
                "config.metrics.episode_report.reward_zero_tolerance",
            )

    if family == "fronteras":
        if stem in (
            "local_cost_lower_boundary_frequency",
            "local_cost_upper_boundary_frequency",
        ):
            return _sources(
                f"interval.local_cost_01_{controller_id}",
                "config.metrics.episode_report.relational_thresholds.boundary",
            )
        if stem in (
            "local_health_lower_boundary_frequency",
            "local_health_upper_boundary_frequency",
        ):
            return _sources(
                f"interval.local_health_01_{controller_id}",
                "config.metrics.episode_report.relational_thresholds.boundary",
            )
        if stem in (
            "reward_negative_boundary_frequency",
            "reward_positive_boundary_frequency",
        ):
            return _sources(
                f"interval.reward_controller_{controller_id}",
                "config.metrics.episode_report.relational_thresholds.boundary",
            )
        if stem == "reward_clipped_frequency":
            return _sources(f"interval.reward_clipped_{controller_id}")
        if stem == "reward_clip_excess_max":
            return _sources(f"interval.reward_clip_excess_{controller_id}")

    if family == "recompensa":
        if stem in ("local_reward_variance", "local_reward_effective_scale"):
            return _sources(f"interval.local_reward_signed_{controller_id}")
        if stem in (
            "local_contribution_variance",
            "local_contribution_effective_scale",
        ):
            return _sources(
                f"interval.local_reward_signed_{controller_id}",
                "config.reward.composition.weights.local",
            )

    if family == "recompensa_aprendizaje":
        if stem == "total_reward":
            return _sources(f"episode.total_reward_{controller_id}")
        if stem == "total_learning_return":
            return _sources(f"episode.total_learning_return_{controller_id}")

    raise KeyError(
        f"No canonical sources declared for controller metric {canonical_name} ({family})"
    )


def _resolve_global_metric_sources(canonical_name, family, config):
    local_health_sources = _local_health_sources()
    joint_state_sources = _joint_state_sources()
    controller_reward_sources = _controller_reward_sources()

    if family == "resultado_estabilidad":
        if canonical_name in ("success", "failure"):
            return _sources(
                "episode.end_termination_reason_code",
                "config.termination.reason_codes.stabilization_success",
            )
        if canonical_name == "stability_persistence_samples":
            return _sources(
                "config.termination.stabilization.mode",
                "config.termination.stabilization.persistence_sec",
                "config.simulation.dt_sec",
            )
        if canonical_name in (
            "stabilization_start_sec",
            "stabilization_start_step",
            "stable_block_max_sec",
            "stable_occupancy",
            "joint_stability_loss_count",
            "joint_post_entry_duration_sec",
            "joint_stability_loss_rate_hz",
        ):
            sources = list(joint_state_sources)
            if canonical_name == "stabilization_start_sec":
                sources.append("step.t_sec")
            if canonical_name in (
                "stabilization_start_sec",
                "stabilization_start_step",
            ):
                sources.append("config.termination.stabilization.mode")
                sources.append("config.termination.stabilization.persistence_sec")
                sources.append("config.simulation.dt_sec")
            if canonical_name in (
                "stable_block_max_sec",
                "joint_post_entry_duration_sec",
                "joint_stability_loss_rate_hz",
            ):
                sources.append("config.simulation.dt_sec")
            return _sources(*sources)

    if family == "velocidad":
        velocity_sources = {
            "rms_pendulum_velocity": "step.pendulum_velocity_raw",
            "max_abs_pendulum_velocity": "step.pendulum_velocity_raw",
            "rms_cart_velocity": "step.cart_velocity_raw",
            "max_abs_cart_velocity": "step.cart_velocity_raw",
        }
        if canonical_name in velocity_sources:
            return _sources(velocity_sources[canonical_name])

    if family == "control_actuador":
        if canonical_name in (
            "saturation_count",
            "saturation_frequency",
        ):
            return _sources("step.is_saturated_global")
        if canonical_name in ("saturation_duration_sec", "saturation_block_max_sec"):
            return _sources(
                "step.is_saturated_global", "config.simulation.dt_sec"
            )
        if canonical_name == "min_pre_saturation_margin":
            return _sources(
                "step.u_total_raw", "config.control.global_actuator_limits"
            )
        if canonical_name in ("cancellation_index", "cancellation_denominator"):
            return _sources(
                "step.u_total_raw",
                *_controller_demand_sources(config),
                "config.simulation.dt_sec",
            )
        if canonical_name in ("control_effort_u2", "control_mean_square"):
            return _sources("step.u_total", "config.simulation.dt_sec")
        if canonical_name == "demand_effort":
            return _sources(
                *_controller_demand_sources(config), "config.simulation.dt_sec"
            )
        if canonical_name in ("actuation_energy_abs", "actuation_power_abs_mean"):
            return _sources(
                "step.cart_force",
                "step.cart_velocity_raw",
                "config.simulation.dt_sec",
            )
        if canonical_name in ("control_total_variation", "max_abs_u_applied"):
            return _sources("step.u_total")
        if canonical_name == "post_stabilization_control_effort_u2":
            return _sources(
                "step.u_total",
                *joint_state_sources,
                "config.termination.stabilization.mode",
                "config.termination.stabilization.persistence_sec",
                "config.simulation.dt_sec",
            )
        if canonical_name == "post_stabilization_actuation_energy_abs":
            return _sources(
                "step.cart_force",
                *joint_state_sources,
                "config.termination.stabilization.mode",
                "config.termination.stabilization.persistence_sec",
                "config.simulation.dt_sec",
            )
        if canonical_name == "post_stabilization_duration_sec":
            return _sources(
                *joint_state_sources,
                "config.termination.stabilization.mode",
                "config.termination.stabilization.persistence_sec",
                "config.simulation.dt_sec",
            )

    if family == "calidad_comun":
        if canonical_name in (
            "mean_health_mean",
            "mean_health_min",
            "mean_health_median",
            "mean_health_final",
            "absolute_common_utility_mean",
        ):
            return local_health_sources
        if canonical_name == "mean_health_above_threshold_frequency":
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.common_health",
            )

    if family == "peor_salud":
        if canonical_name in (
            "worst_health_lower_quantile",
            "worst_health_lower_tail_mean",
        ):
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.lower_tail_quantile",
            )
        if canonical_name == "worst_health_below_threshold_frequency":
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.worst_health",
            )

    if family == "brecha_estatica":
        if canonical_name in (
            "health_gap_abs_mean",
            "health_gap_square_mean",
            "health_gap_rms",
            "health_gap_abs_max",
            "health_gap_abs_final",
        ):
            return local_health_sources
        if canonical_name == "health_gap_within_tolerance_frequency":
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.balance",
            )

    if family == "sincronia":
        if canonical_name in (
            "synchrony_mismatch_abs_mean",
            "synchrony_mismatch_square_mean",
            "synchrony_mismatch_abs_max",
            "synchrony_mismatch_abs_final",
        ):
            return local_health_sources
        if canonical_name == "synchrony_within_tolerance_frequency":
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.synchrony",
            )

    if family == "progreso_comun":
        if canonical_name in (
            "common_progress_mean",
            "common_progress_positive_mean",
            "common_progress_negative_mean",
        ):
            return local_health_sources
        if canonical_name in (
            "common_progress_positive_frequency",
            "common_progress_negative_frequency",
            "common_progress_zero_frequency",
        ):
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.progress",
            )
        if canonical_name == "common_progress_return_discounted":
            return _sources(
                *local_health_sources, "config.agent.params.discount_factor"
            )
        if canonical_name == "common_progress_weighted_return_discounted":
            return _sources(
                "interval.factorial_common_progress_contribution_signed",
                "config.agent.params.discount_factor",
            )
        if canonical_name == "common_progress_given_synchrony_mean":
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.synchrony",
            )

    if family == "dinamica_brecha":
        if canonical_name == "health_gap_oriented_change_mean":
            return local_health_sources
        if canonical_name in (
            "health_gap_closing_frequency",
            "health_gap_expanding_frequency",
            "health_gap_unchanged_frequency",
        ):
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.boundary",
            )
        if canonical_name in (
            "health_gap_zero_crossing_count",
        ):
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.equivalence",
            )
        if canonical_name == "health_gap_sign_change_count":
            return local_health_sources

    if family == "taxonomia_transiciones":
        valid_names = tuple(
            f"transition_{category}_frequency"
            for category in (
                "joint_improvement",
                "joint_deterioration",
                "compensation_pendulum_improves",
                "compensation_cart_improves",
                "unilateral_pendulum_change",
                "unilateral_cart_change",
                "stagnation",
            )
        )
        if canonical_name in valid_names:
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.progress",
            )

    if family == "cumplimiento_conjunto" and canonical_name == "joint_health_threshold_frequency":
        return _sources(
            *local_health_sources,
            "config.metrics.episode_report.joint_health_thresholds",
        )

    if family == "mecanismo_salud":
        if canonical_name in (
            "health_transition_count",
            "worst_health_mean",
            "worst_health_min",
            "worst_health_final",
            "health_gap_mean",
            "health_gap_max",
            "health_balance_mean",
            "health_balance_min",
            "health_balance_final",
        ):
            return local_health_sources
        if canonical_name in (
            "concurrent_progress_count",
            "concurrent_progress_frequency",
            "deterioration_count",
            "deterioration_frequency",
            "compensatory_transition_count",
            "compensatory_transition_frequency",
        ):
            return _sources(
                *local_health_sources,
                "config.metrics.episode_report.relational_thresholds.progress",
            )

    if family == "compensabilidad":
        if canonical_name == "collective_noncompensability_loss_mean":
            return _sources("interval.collective_noncompensability_loss_01")
        if canonical_name == "collective_noncompensability_weighted_loss_mean":
            return _sources(
                "interval.collective_noncompensability_loss_01",
                "config.reward.composition.weights.cooperative",
            )
        if canonical_name in (
            "generalized_welfare_mean",
            "generalized_welfare_min",
            "generalized_welfare_final",
        ):
            return _sources("interval.generalized_welfare_01")

    if family == "cooperacion_monotona":
        source_by_metric = {
            "monotonic_weighted_mean_health_mean": "monotonic_weighted_mean_health_01",
            "monotonic_minimum_health_mean": "monotonic_minimum_health_01",
            "monotonic_health_gap_mean": "monotonic_health_gap_01",
            "monotonic_utility_mean": "monotonic_utility_01",
            "monotonic_utility_min": "monotonic_utility_01",
            "monotonic_utility_final": "monotonic_utility_01",
            "monotonic_utility_post_penalty_mean": "monotonic_utility_post_penalty_01",
            "monotonic_potential_shaping_mean": "monotonic_potential_shaping_signed",
            "monotonic_base_reward_mean": "monotonic_base_reward_signed",
            "monotonic_post_penalty_reward_mean": "monotonic_post_penalty_reward_signed",
            "monotonic_post_compose_penalty_activation_mean": "monotonic_post_compose_penalty_activation_01",
            "monotonic_post_compose_penalty_cost_mean": "monotonic_post_compose_penalty_cost_01",
            "monotonic_post_compose_penalty_mean": "monotonic_post_compose_penalty_01",
        }
        if canonical_name in source_by_metric:
            return _sources(f"interval.{source_by_metric[canonical_name]}")
        prefix = "monotonic_post_compose_penalty_contribution_mean_"
        if canonical_name.startswith(prefix):
            variable_id = canonical_name[len(prefix) :]
            return _sources(
                "interval.monotonic_post_compose_penalty_contribution_01_"
                f"{variable_id}"
            )

    if family == "modos_recompensa":
        if canonical_name in (
            "reward_common_mean",
            "reward_differential_abs_mean",
            "reward_controller_separation_abs_mean",
        ):
            return controller_reward_sources
        if canonical_name == "reward_differential_gain_observed":
            return _sources(
                *controller_reward_sources, "interval.health_gap_signed"
            )
        if canonical_name in (
            "reward_common_return_discounted",
            "reward_differential_return_discounted",
            "reward_differential_abs_return_discounted",
        ):
            return _sources(
                *controller_reward_sources, "config.agent.params.discount_factor"
            )
        if canonical_name in (
            "health_advantage_pendulum_frequency",
            "health_advantage_cart_frequency",
            "health_equivalence_frequency",
        ):
            return _sources(
                "interval.health_gap_signed",
                "config.metrics.episode_report.relational_thresholds.equivalence",
            )

    if family == "equilibrio_aparente" and canonical_name in (
        "apparent_equilibrium_frequency",
        "apparent_equilibrium_block_max_sec",
    ):
        sources = list(
            _sources(
                *local_health_sources,
                *joint_state_sources,
                "interval.interval_end_step_idx",
                "config.metrics.episode_report.relational_thresholds.progress",
                "config.metrics.episode_report.relational_thresholds.synchrony",
            )
        )
        if canonical_name == "apparent_equilibrium_block_max_sec":
            sources.append("interval.interval_duration_sec")
        return _sources(*sources)

    if family == "recompensa":
        if canonical_name in (
            "cooperative_reward_positive_proportion",
            "cooperative_reward_zero_proportion",
            "cooperative_reward_negative_proportion",
        ):
            return _sources(
                "interval.cooperative_reward_signed",
                "config.metrics.episode_report.reward_zero_tolerance",
            )
        if canonical_name in (
            "cooperative_reward_variance",
            "cooperative_reward_effective_scale",
        ):
            return _sources("interval.cooperative_reward_signed")
        if canonical_name in (
            "cooperative_contribution_variance",
            "cooperative_contribution_effective_scale",
        ):
            return _sources(
                "interval.cooperative_reward_signed",
                "config.reward.composition.weights.cooperative",
            )

    if family == "bono_terminal" and canonical_name == "terminal_bonus":
        return _sources("interval.terminal_bonus")

    if family == "recompensa_historica" and canonical_name == "total_reward":
        return _sources("episode.total_reward")

    if family == "recompensa_aprendizaje":
        if canonical_name == "total_learning_return":
            return _sources("episode.total_learning_return")
        if canonical_name == "learning_return_rate":
            return _sources(
                "episode.total_learning_return", "episode.final_t_sec"
            )

    if family == "parametros_aprendizaje":
        if canonical_name == "epsilon":
            return _sources("episode.epsilon")
        if canonical_name == "learning_rate":
            return _sources("episode.learning_rate")

    if family == "aprendizaje":
        if canonical_name == "q_delta_rms_agent_mean":
            return _sources(
                *(
                    f"agent.episode_learning_metrics[{agent_config['id']}].q_delta_rms"
                    for agent_config in config["agent"]["gain_agents"]
                )
            )
        if canonical_name == "policy_stability_agent_mean":
            return _sources(
                *(
                    f"agent.episode_learning_metrics[{agent_config['id']}].policy_stability"
                    for agent_config in config["agent"]["gain_agents"]
                )
            )

    if family == "comprobacion_algebraica":
        if canonical_name == "cooperative_mode_sum_identity_max_error":
            return _sources(
                "interval.cooperative_reward_signed_pendulum_angle",
                "interval.cooperative_reward_signed_cart_position",
                "interval.cooperative_reward_signed",
            )
        if canonical_name == "cooperative_mode_difference_identity_max_error":
            return _sources(
                "interval.cooperative_reward_signed_pendulum_angle",
                "interval.cooperative_reward_signed_cart_position",
                "interval.cooperative_reward_differential_signed",
            )
        if canonical_name == "factorial_common_identity_max_error":
            return _sources(
                "interval.cooperative_reward_signed",
                "interval.factorial_absolute_contribution_signed",
                "interval.factorial_balance_contribution_signed",
                "interval.factorial_synchrony_contribution_signed",
                "interval.factorial_common_progress_contribution_signed",
            )
        if canonical_name == "factorial_differential_identity_max_error":
            return _sources(
                "interval.cooperative_reward_differential_signed",
                "interval.factorial_differential_contribution_signed",
            )
        if canonical_name == "intensity_differential_identity_max_error":
            return _sources(
                *controller_reward_sources,
                "interval.health_gap_signed",
                "config.reward.composition.weights.local",
            )

    raise KeyError(
        f"No canonical sources declared for global metric {canonical_name} ({family})"
    )


def _resolve_metric_sources(
    canonical_name, family, entity_type, entity_id, config
):
    if family == "escala_componentes":
        if entity_type != "componente":
            raise KeyError(
                f"Component metric {canonical_name} has invalid entity type {entity_type}"
            )
        for stat_name in _COMPONENT_STAT_NAMES:
            if canonical_name == f"{entity_id}_{stat_name}":
                if stat_name in (
                    "positive_proportion",
                    "negative_proportion",
                    "zero_proportion",
                ):
                    return _sources(
                        f"interval.{entity_id}",
                        "config.metrics.episode_report.reward_zero_tolerance",
                    )
                return _sources(f"interval.{entity_id}")
        raise KeyError(f"Unknown component statistic source: {canonical_name}")

    if entity_type == "agente":
        return _resolve_agent_metric_sources(
            canonical_name, family, entity_id, config
        )
    if entity_type in ("lazo", "controlador"):
        return _resolve_controller_metric_sources(
            canonical_name, family, entity_id, config
        )
    if entity_type == "global":
        return _resolve_global_metric_sources(canonical_name, family, config)
    raise KeyError(
        f"No canonical source resolver for metric entity type {entity_type}: {canonical_name}"
    )


@dataclass(frozen=True)
class MetricSpec:
    canonical_name: str
    scientific_label: str
    family: str
    entity_type: str
    entity_id: str
    unit_or_domain: str
    definition: str
    sources: tuple
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
            "sources": METRIC_SOURCE_SEPARATOR.join(self.sources),
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
    "sources",
    "applicability",
    "denominator",
    "recommended_summary",
    "interpretation",
    "number_format",
)

EPISODE_IDENTITY_COLUMNS = (
    "run_id",
    "config_id",
    "seed",
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
    *,
    sources,
):
    return MetricSpec(
        canonical_name=canonical_name,
        scientific_label=scientific_label,
        family=family,
        entity_type=entity_type,
        entity_id=entity_id,
        unit_or_domain=unit_or_domain,
        definition=definition,
        sources=sources,
        applicability=applicability,
        denominator=denominator,
        recommended_summary=recommended_summary,
        interpretation=interpretation,
        number_format=number_format,
    )


def build_metric_report_specs(config):
    specs = []
    steady_state_window_sec = float(
        config["metrics"]["episode_report"]["steady_state_window_sec"]
    )

    def add(*args):
        sources = _resolve_metric_sources(
            args[0],
            args[2],
            args[3],
            args[4],
            config,
        )
        specs.append(_spec(*args, sources=sources))

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
            f"Media firmada del error físico en la ventana final de {steady_state_window_sec:g} s.",
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
            f"Media del valor absoluto del error físico en la ventana final de {steady_state_window_sec:g} s.",
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
            f"Indica disponibilidad de la ventana final completa de {steady_state_window_sec:g} s.",
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

    monotonic_specs = (
        ("monotonic_weighted_mean_health_mean", "Salud media ponderada monotónica", "[0,1]", "Media temporal de la combinación ponderada de las saludes declaradas.", "Mayor es mejor."),
        ("monotonic_minimum_health_mean", "Salud mínima monotónica", "[0,1]", "Media temporal de la menor salud declarada por intervalo.", "Mayor es mejor; representa el cuello de botella."),
        ("monotonic_health_gap_mean", "Brecha de salud monotónica", "[0,1]", "Media temporal de la diferencia entre la mayor y la menor salud declarada.", "Menor indica menor separación relativa."),
        ("monotonic_utility_mean", "Utilidad monotónica media", "[0,1]", "Media temporal de la utilidad cooperativa previa a penalizaciones posteriores.", "Mayor es mejor."),
        ("monotonic_utility_min", "Utilidad monotónica mínima", "[0,1]", "Mínimo temporal de la utilidad cooperativa previa a penalizaciones posteriores.", "Mayor es mejor."),
        ("monotonic_utility_final", "Utilidad monotónica final", "[0,1]", "Utilidad cooperativa previa a penalizaciones en el último intervalo.", "Mayor es mejor."),
        ("monotonic_utility_post_penalty_mean", "Utilidad monotónica posterior a penalización", "[0,1]", "Media temporal de la utilidad cooperativa después de la penalización condicionada.", "Mayor es mejor condicionado a la eficiencia."),
        ("monotonic_potential_shaping_mean", "Moldeado potencial monotónico medio", "recompensa", "Media de γG(s')-G(s) calculada con la utilidad monotónica sin penalización de esfuerzo.", "Describe la guía potencial añadida a la base absoluta."),
        ("monotonic_base_reward_mean", "Recompensa base monotónica media", "recompensa", "Media de la transformación firmada de la utilidad monotónica previa a penalización.", "Describe el objetivo cooperativo absoluto."),
        ("monotonic_post_penalty_reward_mean", "Recompensa monotónica posterior a penalización", "recompensa", "Media de la transformación firmada posterior a penalización.", "Describe el objetivo absoluto tras eficiencia condicionada."),
        ("monotonic_post_compose_penalty_activation_mean", "Activación media de penalización posterior", "[0,1]", "Media del factor de activación definido sobre la utilidad monotónica.", "Cero desactiva el costo de eficiencia en baja calidad."),
        ("monotonic_post_compose_penalty_cost_mean", "Costo medio de penalización posterior", "[0,1]", "Media de la suma ponderada de costos declarados antes de aplicar la activación.", "Menor es mejor condicionado a la calidad."),
        ("monotonic_post_compose_penalty_mean", "Penalización posterior media", "[0,1]", "Media del producto entre activación y costo posterior compuesto.", "Cuantifica la deducción aplicada a la utilidad."),
    )
    for name, label, unit, definition, interpretation in monotonic_specs:
        add(
            name,
            label,
            "cooperacion_monotona",
            "global",
            "monotonic_cooperative",
            unit,
            definition,
            "Familia monotonic_cooperative activa.",
            "n_interval",
            "media o mediana según diseño experimental",
            interpretation,
            "0.000000",
        )

    for variable_config in config["reward"]["cooperative"]["monotonic_cooperative"][
        "post_compose_penalty"
    ]["variables"]:
        variable_id = variable_config["id"]
        add(
            f"monotonic_post_compose_penalty_contribution_mean_{variable_id}",
            f"Contribución media de penalización — {variable_id}",
            "cooperacion_monotona",
            "global",
            "monotonic_cooperative",
            "[0,1]",
            f"Media de la contribución ponderada del costo {variable_id}.",
            "Familia monotonic_cooperative activa con penalización posterior habilitada.",
            "n_interval",
            "media o mediana según diseño experimental",
            "Cuantifica la contribución previa al factor de activación.",
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

    additional_global_specs = (
        ("stabilization_start_step", "Paso inicial de estabilización", "resultado_estabilidad", "conteo", "Índice físico del primer estado del primer bloque estable persistente.", "Existe un bloque estable persistente.", "n_step", "Menor es mejor.", "0"),
        ("rms_pendulum_velocity", "RMS de velocidad angular", "velocidad", "rad/s", "RMS de la velocidad angular física.", "Existen pasos físicos válidos.", "n_step", "Menor es mejor.", "0.000000"),
        ("max_abs_pendulum_velocity", "Velocidad angular máxima", "velocidad", "rad/s", "Máximo absoluto de la velocidad angular física.", "Existen pasos físicos válidos.", "n_step", "Menor es mejor.", "0.000000"),
        ("rms_cart_velocity", "RMS de velocidad del carro", "velocidad", "m/s", "RMS de la velocidad lineal física.", "Existen pasos físicos válidos.", "n_step", "Menor es mejor.", "0.000000"),
        ("max_abs_cart_velocity", "Velocidad máxima del carro", "velocidad", "m/s", "Máximo absoluto de la velocidad lineal física.", "Existen pasos físicos válidos.", "n_step", "Menor es mejor.", "0.000000"),
        ("saturation_block_max_sec", "Bloque máximo de saturación", "control_actuador", "s", "Duración del bloque consecutivo más largo bajo saturación.", "Existen pasos físicos válidos.", "n_step", "Menor es mejor.", "0.000000"),
        ("mean_health_mean", "Salud media temporal", "calidad_comun", "[0,1]", "Media episódica de μ.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("mean_health_min", "Salud media mínima", "calidad_comun", "[0,1]", "Mínimo episódico de μ.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("mean_health_median", "Mediana de salud media", "calidad_comun", "[0,1]", "Mediana episódica de μ.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("mean_health_final", "Salud media final", "calidad_comun", "[0,1]", "Valor de μ en el último intervalo.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("mean_health_above_threshold_frequency", "Cumplimiento de salud común", "calidad_comun", "proporción", "Fracción de intervalos con μ por encima del umbral declarado.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.0000"),
        ("absolute_common_utility_mean", "Utilidad absoluta común media", "calidad_comun", "[-1,1]", "Media episódica de ρA.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("worst_health_lower_quantile", "Cuantil inferior de peor salud", "peor_salud", "[0,1]", "Cuantil inferior declarado de la menor salud intervalar.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("worst_health_lower_tail_mean", "Media de cola inferior de peor salud", "peor_salud", "[0,1]", "Media de valores de peor salud menores o iguales al cuantil inferior.", "Existe al menos un intervalo.", "valores en cola inferior", "Mayor es mejor.", "0.000000"),
        ("worst_health_below_threshold_frequency", "Incumplimiento de peor salud", "peor_salud", "proporción", "Fracción de intervalos con la peor salud por debajo del umbral declarado.", "Existe al menos un intervalo.", "n_interval", "Menor es mejor.", "0.0000"),
        ("health_gap_abs_mean", "Brecha absoluta media", "brecha_estatica", "[0,1]", "Media de |ξ|.", "Existe al menos un intervalo.", "n_interval", "Menor es mejor.", "0.000000"),
        ("health_gap_square_mean", "Brecha cuadrática media", "brecha_estatica", "[0,1]", "Media de ξ².", "Existe al menos un intervalo.", "n_interval", "Menor es mejor.", "0.000000"),
        ("health_gap_rms", "RMS de brecha", "brecha_estatica", "[0,1]", "Raíz de la media de ξ².", "Existe al menos un intervalo.", "n_interval", "Menor es mejor.", "0.000000"),
        ("health_gap_abs_max", "Brecha absoluta máxima", "brecha_estatica", "[0,1]", "Máximo de |ξ|.", "Existe al menos un intervalo.", "n_interval", "Menor es mejor.", "0.000000"),
        ("health_gap_abs_final", "Brecha absoluta final", "brecha_estatica", "[0,1]", "Valor final de |ξ|.", "Existe al menos un intervalo.", "n_interval", "Menor es mejor.", "0.000000"),
        ("health_gap_within_tolerance_frequency", "Balance dentro de tolerancia", "brecha_estatica", "proporción", "Fracción de intervalos con |ξ| dentro de la tolerancia declarada.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.0000"),
        ("synchrony_mismatch_abs_mean", "Desajuste temporal absoluto medio", "sincronia", "[0,1]", "Media de |Δξ|/2 sobre transiciones válidas.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.000000"),
        ("synchrony_mismatch_square_mean", "Desajuste temporal cuadrático medio", "sincronia", "[0,1]", "Media de (Δξ/2)² sobre transiciones válidas.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.000000"),
        ("synchrony_mismatch_abs_max", "Desajuste temporal máximo", "sincronia", "[0,1]", "Máximo de |Δξ|/2.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.000000"),
        ("synchrony_mismatch_abs_final", "Desajuste temporal final", "sincronia", "[0,1]", "Último valor de |Δξ|/2.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.000000"),
        ("synchrony_within_tolerance_frequency", "Sincronía dentro de tolerancia", "sincronia", "proporción", "Fracción de transiciones con |Δξ|/2 dentro de tolerancia.", "Existe al menos una transición.", "health_transition_count", "Mayor es mejor.", "0.0000"),
        ("common_progress_mean", "Progreso común medio", "progreso_comun", "[-1,1]", "Media de P sobre transiciones válidas.", "Existe al menos una transición.", "health_transition_count", "Mayor indica orientación neta favorable.", "0.000000"),
        ("common_progress_positive_mean", "Mejora común media", "progreso_comun", "[0,1]", "Media de max(P,0).", "Existe al menos una transición.", "health_transition_count", "Mayor indica más volumen de mejora.", "0.000000"),
        ("common_progress_negative_mean", "Deterioro común medio", "progreso_comun", "[0,1]", "Media de max(-P,0).", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.000000"),
        ("common_progress_positive_frequency", "Frecuencia de mejora común", "progreso_comun", "proporción", "Fracción de transiciones con P mayor que la tolerancia.", "Existe al menos una transición.", "health_transition_count", "Mayor indica mejora más frecuente.", "0.0000"),
        ("common_progress_negative_frequency", "Frecuencia de deterioro común", "progreso_comun", "proporción", "Fracción de transiciones con P menor que menos la tolerancia.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.0000"),
        ("common_progress_zero_frequency", "Frecuencia de estancamiento común", "progreso_comun", "proporción", "Fracción de transiciones con |P| dentro de tolerancia.", "Existe al menos una transición.", "health_transition_count", "Describe estancamiento.", "0.0000"),
        ("common_progress_return_discounted", "Retorno descontado de progreso común", "progreso_comun", "recompensa", "Suma descontada diagnóstica de P.", "Existe al menos una transición.", "health_transition_count", "Mayor indica progreso temprano.", "0.000000"),
        ("common_progress_weighted_return_discounted", "Retorno ponderado de progreso común", "progreso_comun", "recompensa", "Suma descontada de la contribución factorial de progreso común.", "Existe al menos una transición y la familia factorial está activa en modo common_progress.", "health_transition_count", "Describe la señal efectivamente ponderada.", "0.000000"),
        ("common_progress_given_synchrony_mean", "Progreso condicionado por sincronía", "progreso_comun", "[-1,1]", "Media de P en transiciones dentro de la tolerancia de sincronía.", "Existe al menos una transición sincronizada.", "transiciones sincronizadas", "Distingue sincronía favorable y desfavorable.", "0.000000"),
        ("health_gap_oriented_change_mean", "Cambio orientado de brecha", "dinamica_brecha", "salud²", "Media de ξ(k-1)Δξ(k).", "Existe al menos una transición.", "health_transition_count", "Negativo es compatible con cierre.", "0.000000"),
        ("health_gap_closing_frequency", "Frecuencia de cierre de brecha", "dinamica_brecha", "proporción", "Fracción de transiciones que reducen |ξ|.", "Existe al menos una transición.", "health_transition_count", "Mayor indica cierre más frecuente.", "0.0000"),
        ("health_gap_expanding_frequency", "Frecuencia de expansión de brecha", "dinamica_brecha", "proporción", "Fracción de transiciones que aumentan |ξ|.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.0000"),
        ("health_gap_unchanged_frequency", "Frecuencia de brecha inalterada", "dinamica_brecha", "proporción", "Fracción de transiciones sin cambio de |ξ| dentro de tolerancia.", "Existe al menos una transición.", "health_transition_count", "Describe persistencia relacional.", "0.0000"),
        ("health_gap_zero_crossing_count", "Cruces por brecha nula", "dinamica_brecha", "conteo", "Número de transiciones que cruzan o tocan la banda de equivalencia.", "Existe al menos una transición.", "health_transition_count", "Describe alternancia.", "0"),
        ("health_gap_sign_change_count", "Cambios estrictos de ventaja", "dinamica_brecha", "conteo", "Número de transiciones con signos estrictamente opuestos de ξ.", "Existe al menos una transición.", "health_transition_count", "Describe alternancia de liderazgo.", "0"),
        ("joint_health_threshold_frequency", "Cumplimiento conjunto de salud", "cumplimiento_conjunto", "proporción", "Fracción de intervalos donde ambas saludes superan sus umbrales.", "Existe al menos un intervalo.", "n_interval", "Mayor es mejor.", "0.0000"),
        (
            "collective_noncompensability_loss_mean",
            "Contraste firmado medio de compensabilidad",
            "compensabilidad",
            "[-2,2]",
            "Media de 2(μ-Wη): positiva para una penalización por no compensabilidad, nula para la referencia aritmética y negativa para una ganancia supercompensatoria.",
            "Familia intensidad-compensabilidad activa.",
            "n_interval",
            "El signo determina la dirección respecto de la referencia aritmética.",
            "0.000000",
        ),
        (
            "collective_noncompensability_weighted_loss_mean",
            "Contraste firmado efectivo de compensabilidad",
            "compensabilidad",
            "[-2,2]",
            "Media de wC·2(μ-Wη), incluida la intensidad cooperativa aplicada en la composición final.",
            "Familia intensidad-compensabilidad activa.",
            "n_interval",
            "Valores positivos reducen el modo común y valores negativos lo elevan respecto de la referencia aritmética.",
            "0.000000",
        ),
        ("generalized_welfare_mean", "Bienestar generalizado medio", "compensabilidad", "[0,1]", "Media de Wη.", "Familia intensidad-compensabilidad activa.", "n_interval", "Mayor es mejor condicionado a la geometría.", "0.000000"),
        ("generalized_welfare_min", "Bienestar generalizado mínimo", "compensabilidad", "[0,1]", "Mínimo de Wη.", "Familia intensidad-compensabilidad activa.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("generalized_welfare_final", "Bienestar generalizado final", "compensabilidad", "[0,1]", "Valor final de Wη.", "Familia intensidad-compensabilidad activa.", "n_interval", "Mayor es mejor.", "0.000000"),
        ("reward_common_mean", "Modo común medio de recompensa", "modos_recompensa", "recompensa", "Media de (Rθ+Rx)/2 de la recompensa ordinaria.", "Existe al menos un intervalo.", "n_interval", "Describe el canal común.", "0.000000"),
        ("reward_differential_abs_mean", "Modo diferencial absoluto medio", "modos_recompensa", "recompensa", "Media de |(Rθ-Rx)/2|.", "Existe al menos un intervalo.", "n_interval", "Describe separación de crédito.", "0.000000"),
        ("reward_differential_gain_observed", "Ganancia diferencial observada", "modos_recompensa", "razón", "Suma de |Rdif| dividida por suma de |ξ|.", "La suma de |ξ| es positiva.", "suma de |ξ|", "Verificación mecanística.", "0.000000"),
        ("reward_controller_separation_abs_mean", "Separación media entre controladores", "modos_recompensa", "recompensa", "Media de |Rθ-Rx|.", "Existe al menos un intervalo.", "n_interval", "Equivale a dos veces el modo diferencial absoluto.", "0.000000"),
        ("reward_common_return_discounted", "Retorno común descontado", "modos_recompensa", "recompensa", "Suma descontada del modo común ordinario.", "Existe al menos un intervalo.", "n_interval", "Describe el canal común.", "0.000000"),
        ("reward_differential_return_discounted", "Retorno diferencial descontado", "modos_recompensa", "recompensa", "Suma descontada del modo diferencial firmado.", "Existe al menos un intervalo.", "n_interval", "Puede cancelarse por cambios de signo.", "0.000000"),
        ("reward_differential_abs_return_discounted", "Retorno diferencial absoluto descontado", "modos_recompensa", "recompensa", "Suma descontada del valor absoluto del modo diferencial.", "Existe al menos un intervalo.", "n_interval", "Evita cancelación temporal.", "0.000000"),
        ("health_advantage_pendulum_frequency", "Ventaja angular", "modos_recompensa", "proporción", "Fracción de intervalos con ξ mayor que la banda de equivalencia.", "Existe al menos un intervalo.", "n_interval", "Describe ocupación de ventaja.", "0.0000"),
        ("health_advantage_cart_frequency", "Ventaja de posición", "modos_recompensa", "proporción", "Fracción de intervalos con ξ menor que menos la banda de equivalencia.", "Existe al menos un intervalo.", "n_interval", "Describe ocupación de ventaja.", "0.0000"),
        ("health_equivalence_frequency", "Equivalencia de salud", "modos_recompensa", "proporción", "Fracción de intervalos con |ξ| dentro de la banda de equivalencia.", "Existe al menos un intervalo.", "n_interval", "Describe igualdad práctica.", "0.0000"),
        ("apparent_equilibrium_frequency", "Frecuencia de equilibrio aparente", "equilibrio_aparente", "proporción", "Fracción de transiciones sin progreso ni desajuste, fuera de estabilidad física conjunta.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.0000"),
        ("apparent_equilibrium_block_max_sec", "Bloque máximo de equilibrio aparente", "equilibrio_aparente", "s", "Duración máxima consecutiva compatible con equilibrio aparente.", "Existe al menos una transición.", "health_transition_count", "Menor es mejor.", "0.000000"),
        ("terminal_bonus", "Evaluación terminal media", "bono_terminal", "recompensa", "Media entre los importes finales asignados a los controladores, incluyendo bonos y penalizaciones.", "Todos los episodios.", "controladores", "Es un resumen; la asignación al agente utiliza el valor de su controlador.", "0.000000"),
        ("total_reward", "Retorno histórico de recompensa", "recompensa_historica", "recompensa", "Media entre los retornos acumulados ordinarios de controlador más el bono terminal aplicado una sola vez.", "Existe al menos un intervalo.", "controladores", "Se conserva para trazabilidad; no representa la señal normalizada recibida por el aprendizaje.", "0.000000"),
        ("total_learning_return", "Retorno total de aprendizaje", "recompensa_aprendizaje", "recompensa", "Media entre los retornos acumulados de aprendizaje de los controladores; incluye la recompensa ordinaria normalizada y el bono terminal recibido por los agentes.", "Episodio de entrenamiento con al menos un intervalo.", "controladores", "Representa la señal acumulada que condiciona directamente las actualizaciones de aprendizaje.", "0.000000"),
        ("learning_return_rate", "Tasa de retorno de aprendizaje", "recompensa_aprendizaje", "recompensa/s", "Retorno total de aprendizaje dividido por la duración efectiva del episodio.", "Episodio de entrenamiento con duración efectiva positiva.", "final_t_sec", "Permite comparar episodios de distinta duración sin confundir retorno acumulado con supervivencia.", "0.000000"),
        ("epsilon", "Epsilon del episodio", "parametros_aprendizaje", "proporción", "Valor de exploración utilizado en el episodio antes del decaimiento posterior.", "Episodios de entrenamiento.", "episodio", "Contextualiza la exploración.", "0.000000"),
        ("learning_rate", "Tasa de aprendizaje del episodio", "parametros_aprendizaje", "proporción", "Tasa de aprendizaje utilizada en el episodio antes del decaimiento posterior.", "Episodios de entrenamiento.", "episodio", "Contextualiza las actualizaciones Q.", "0.000000"),
        ("cooperative_mode_sum_identity_max_error", "Error de identidad modal de suma", "comprobacion_algebraica", "error absoluto", "Máximo error de RCθ+RCx=2RCcom.", "Existe al menos un intervalo.", "n_interval", "Debe ser numéricamente cero.", "0.000000000000"),
        ("cooperative_mode_difference_identity_max_error", "Error de identidad modal diferencial", "comprobacion_algebraica", "error absoluto", "Máximo error de RCθ-RCx=2RCdif.", "Existe al menos un intervalo.", "n_interval", "Debe ser numéricamente cero.", "0.000000000000"),
        ("factorial_common_identity_max_error", "Error de identidad factorial común", "comprobacion_algebraica", "error absoluto", "Máximo error de reconstrucción de la suma factorial común.", "Familia factorial activa.", "n_interval", "Debe ser numéricamente cero.", "0.000000000000"),
        ("factorial_differential_identity_max_error", "Error de identidad factorial diferencial", "comprobacion_algebraica", "error absoluto", "Máximo error de reconstrucción del modo factorial diferencial.", "Familia factorial activa con componente diferencial habilitada.", "n_interval", "Debe ser numéricamente cero.", "0.000000000000"),
        ("intensity_differential_identity_max_error", "Error diferencial de intensidad", "comprobacion_algebraica", "error absoluto", "Máximo error de Rdif=wL·ξ.", "Familia intensidad-compensabilidad activa.", "n_interval", "Debe ser numéricamente cero.", "0.000000000000"),
    )
    for (
        name,
        label,
        family,
        unit,
        definition,
        applicability,
        denominator,
        interpretation,
        number_format,
    ) in additional_global_specs:
        add(
            name,
            label,
            family,
            "global",
            "global",
            unit,
            definition,
            applicability,
            denominator,
            "media o mediana según diseño experimental",
            interpretation,
            number_format,
        )

    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        for stem, label, family, unit, definition, interpretation, number_format in (
            ("terminal_inside_bands_final", "Variables físicas en banda al cierre", "bono_terminal", "indicador", "Uno si todas las variables declaradas en vars del controlador están dentro de sus bandas físicas en la última muestra.", "Describe el cumplimiento instantáneo propio.", "0"),
            ("terminal_band_duration_sec_final", "Permanencia final en bandas físicas", "bono_terminal", "s", "Número de pasos físicos consecutivos con todas las variables del lazo en banda multiplicado por dt; cero si termina fuera.", "Utiliza la misma convención temporal que la estabilización física.", "0.000000"),
            ("terminal_band_qualified_final", "Cumplimiento físico local al cierre", "bono_terminal", "indicador", "Uno si las variables del lazo satisfacen el modo y persistencia de termination.stabilization al cierre, independientemente del modo del bono.", "Describe el logro físico propio.", "0"),
            ("terminal_bonus_eligible_final", "Elegibilidad al cierre", "bono_terminal", "indicador", "Cumplimiento propio en modo local o cumplimiento conjunto en modo cooperative, antes de aplicar enabled.", "Identifica qué lazos cumplen el requisito positivo del modo configurado.", "0"),
            ("local_stable_occupancy", "Ocupación estable local", "resultado_estabilidad", "proporción", "Fracción de estados físicos dentro de las bandas locales de salida y velocidad.", "Mayor es mejor.", "0.0000"),
            ("mean_abs_error", "Error absoluto medio", "seguimiento", "unidad física", "IAE dividida por la duración efectiva.", "Menor es mejor.", "0.000000"),
            ("terminal_error_signed", "Error terminal firmado", "seguimiento", "unidad física", "Último error físico firmado del episodio.", "Cero indica ausencia de sesgo.", "0.000000"),
            ("terminal_error_abs", "Error terminal absoluto", "seguimiento", "unidad física", "Valor absoluto del último error físico.", "Menor es mejor.", "0.000000"),
            ("local_cooperative_contribution_covariance", "Covarianza local-cooperativa", "interaccion_componentes", "recompensa²", "Covarianza poblacional entre contribuciones local y cooperativa.", "Describe interacción de componentes.", "0.000000"),
            ("local_cooperative_opposite_sign_frequency", "Contribuciones con signo opuesto", "interaccion_componentes", "proporción", "Fracción de intervalos donde contribuciones local y cooperativa tienen signos opuestos.", "Menor indica menos oposición.", "0.0000"),
            ("local_cooperative_same_sign_frequency", "Contribuciones con mismo signo", "interaccion_componentes", "proporción", "Fracción de intervalos donde contribuciones local y cooperativa tienen el mismo signo no nulo.", "Describe refuerzo entre bloques.", "0.0000"),
            ("local_cost_lower_boundary_frequency", "Costo local en frontera inferior", "fronteras", "proporción", "Fracción de costos locales dentro de tolerancia de cero.", "Describe ocupación de frontera.", "0.0000"),
            ("local_cost_upper_boundary_frequency", "Costo local en frontera superior", "fronteras", "proporción", "Fracción de costos locales dentro de tolerancia de uno.", "Describe ocupación de frontera.", "0.0000"),
            ("local_health_lower_boundary_frequency", "Salud local en frontera inferior", "fronteras", "proporción", "Fracción de saludes locales dentro de tolerancia de cero.", "Describe ocupación de frontera.", "0.0000"),
            ("local_health_upper_boundary_frequency", "Salud local en frontera superior", "fronteras", "proporción", "Fracción de saludes locales dentro de tolerancia de uno.", "Describe ocupación de frontera.", "0.0000"),
            ("reward_negative_boundary_frequency", "Recompensa en frontera negativa", "fronteras", "proporción", "Fracción de recompensas ordinarias dentro de tolerancia de -1.", "Describe ocupación de frontera.", "0.0000"),
            ("reward_positive_boundary_frequency", "Recompensa en frontera positiva", "fronteras", "proporción", "Fracción de recompensas ordinarias dentro de tolerancia de 1.", "Describe ocupación de frontera.", "0.0000"),
            ("reward_clipped_frequency", "Frecuencia de recorte", "fronteras", "proporción", "Fracción de intervalos marcados como recortados.", "Debe ser cero para las nuevas familias.", "0.0000"),
            ("reward_clip_excess_max", "Exceso máximo de recompensa", "fronteras", "recompensa", "Máximo exceso de |Rpre| sobre uno.", "Debe ser cero para las nuevas familias.", "0.000000"),
            ("total_reward", "Retorno acumulado del controlador", "recompensa_aprendizaje", "recompensa", "Suma intervalar de la recompensa ordinaria compuesta del controlador, conservada sin cambios.", "No incluye el bono terminal.", "0.000000"),
            ("total_learning_return", "Retorno de aprendizaje del controlador", "recompensa_aprendizaje", "recompensa", "Media entre los retornos acumulados de aprendizaje de los agentes de ganancia asociados al controlador.", "Incluye la recompensa ordinaria normalizada y el bono terminal.", "0.000000"),
        ):
            if stem == "total_learning_return":
                applicability = "Episodio de entrenamiento."
            else:
                applicability = "Existe al menos una observación válida."
            add(
                f"{stem}_{controller_id}",
                f"{label} — {controller_id}",
                family,
                "controlador",
                controller_id,
                unit,
                definition,
                applicability,
                "n_step o n_interval según la métrica",
                "media o mediana según diseño experimental",
                interpretation,
                number_format,
            )

    transition_categories = (
        "joint_improvement",
        "joint_deterioration",
        "compensation_pendulum_improves",
        "compensation_cart_improves",
        "unilateral_pendulum_change",
        "unilateral_cart_change",
        "stagnation",
    )
    for category in transition_categories:
        add(
            f"transition_{category}_frequency",
            f"Frecuencia de transición — {category}",
            "taxonomia_transiciones",
            "global",
            "global",
            "proporción",
            f"Fracción de transiciones clasificadas como {category}.",
            "Existe al menos una transición válida.",
            "health_transition_count",
            "media",
            "Las siete frecuencias suman uno.",
            "0.0000",
        )

    component_sources = (
        "absolute_common_utility_signed",
        "balance_utility_signed",
        "synchrony_utility_signed",
        "common_progress_signed",
        "factorial_absolute_contribution_signed",
        "factorial_balance_contribution_signed",
        "factorial_synchrony_contribution_signed",
        "factorial_differential_contribution_signed",
        "factorial_common_progress_contribution_signed",
        "generalized_welfare_01",
        "collective_noncompensability_loss_01",
        "collective_noncompensability_weighted_loss_01",
        "cooperative_reward_signed",
        "cooperative_reward_differential_signed",
    )
    component_stats = (
        ("mean", "media", "recompensa"),
        ("variance", "varianza poblacional", "recompensa²"),
        ("rms", "RMS", "recompensa"),
        ("abs_mean", "valor absoluto medio", "recompensa"),
        ("positive_proportion", "proporción positiva", "proporción"),
        ("negative_proportion", "proporción negativa", "proporción"),
        ("zero_proportion", "proporción nula", "proporción"),
    )
    for source_name in component_sources:
        if source_name == "factorial_absolute_contribution_signed":
            applicability = "Familia factorial activa."
        elif source_name == "factorial_balance_contribution_signed":
            applicability = "Familia factorial activa con balance habilitado."
        elif source_name == "factorial_synchrony_contribution_signed":
            applicability = "Familia factorial activa con sincronía habilitada."
        elif source_name == "factorial_differential_contribution_signed":
            applicability = (
                "Familia factorial activa con componente diferencial habilitada."
            )
        elif source_name == "factorial_common_progress_contribution_signed":
            applicability = "Familia factorial activa en modo common_progress."
        elif source_name in (
            "generalized_welfare_01",
            "collective_noncompensability_loss_01",
            "collective_noncompensability_weighted_loss_01",
        ):
            applicability = "Familia intensidad-compensabilidad activa."
        else:
            applicability = "La componente es aplicable a la familia activa."
        for stat_name, stat_label, unit in component_stats:
            add(
                f"{source_name}_{stat_name}",
                f"{stat_label} — {source_name}",
                "escala_componentes",
                "componente",
                source_name,
                unit,
                f"{stat_label} de {source_name}.",
                applicability,
                "n_interval",
                "media o mediana según diseño experimental",
                "Describe escala, signo y ocupación de la componente.",
                "0.000000",
            )

    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        for source_name in (
            f"local_reward_signed_{controller_id}",
            f"local_contribution_signed_{controller_id}",
            f"differential_utility_signed_{controller_id}",
            f"cooperative_reward_signed_{controller_id}",
            f"cooperative_contribution_signed_{controller_id}",
            f"reward_controller_{controller_id}",
        ):
            for stat_name, stat_label, unit in component_stats:
                add(
                    f"{source_name}_{stat_name}",
                    f"{stat_label} — {source_name}",
                    "escala_componentes",
                    "componente",
                    source_name,
                    unit,
                    f"{stat_label} de {source_name}.",
                    "Existe al menos un intervalo válido.",
                    "n_interval",
                    "media o mediana según diseño experimental",
                    "Describe escala, signo y ocupación de la componente.",
                    "0.000000",
                )

    for gain_agent_config in config["agent"]["gain_agents"]:
        agent_id = gain_agent_config["id"]
        for stem, label, family, unit, definition, denominator, interpretation, number_format in (
            ("action_decrease_frequency", "Frecuencia de disminuir", "acciones", "proporción", "Fracción de decisiones con acción -1.", "n_interval", "Describe la política ejecutada.", "0.0000"),
            ("action_maintain_frequency", "Frecuencia de mantener", "acciones", "proporción", "Fracción de decisiones con acción 0.", "n_interval", "Describe la política ejecutada.", "0.0000"),
            ("action_increase_frequency", "Frecuencia de aumentar", "acciones", "proporción", "Fracción de decisiones con acción +1.", "n_interval", "Describe la política ejecutada.", "0.0000"),
            ("action_entropy", "Entropía de acciones", "acciones", "nat", "Entropía natural de la distribución de acciones.", "n_interval", "Mayor indica mayor diversidad.", "0.000000"),
            ("exploratory_action_frequency", "Frecuencia de exploración", "acciones", "proporción", "Fracción de decisiones marcadas como exploratorias por el agente.", "n_interval", "Describe exposición a exploración.", "0.0000"),
            ("td_error_mean_abs", "Error TD absoluto medio", "aprendizaje", "valor Q", "Media de |δTD|.", "n_interval", "Describe magnitud media de actualización.", "0.000000"),
            ("td_error_rms", "RMS del error TD", "aprendizaje", "valor Q", "RMS de δTD.", "n_interval", "Describe escala de actualización.", "0.000000"),
            ("td_error_max_abs", "Error TD máximo", "aprendizaje", "valor Q", "Máximo de |δTD|.", "n_interval", "Detecta actualizaciones extremas.", "0.000000"),
            ("q_value_mean", "Media de Q", "aprendizaje", "valor Q", "Media de la tabla Q final mantenida en memoria.", "entradas Q", "Describe nivel de valor.", "0.000000"),
            ("q_value_std", "Desviación de Q", "aprendizaje", "valor Q", "Desviación estándar de la tabla Q final.", "entradas Q", "Describe dispersión.", "0.000000"),
            ("q_value_min", "Mínimo de Q", "aprendizaje", "valor Q", "Mínimo de la tabla Q final.", "entradas Q", "Extremo inferior.", "0.000000"),
            ("q_value_max", "Máximo de Q", "aprendizaje", "valor Q", "Máximo de la tabla Q final.", "entradas Q", "Extremo superior.", "0.000000"),
            ("visited_state_frequency", "Cobertura relativa de estados", "aprendizaje", "proporción", "Estados únicos visitados divididos por estados posibles.", "estados posibles", "Mayor indica mayor cobertura.", "0.0000"),
            ("terminal_bonus_abs_return_fraction", "Fracción absoluta del bono terminal", "bono_terminal", "proporción", "|bT| dividido por la suma de |Rlearn|.", "suma de |Rlearn|", "Mayor indica mayor dominancia terminal.", "0.000000"),
            ("terminal_bonus_separated_abs_return_fraction", "Fracción absoluta separada del bono terminal", "bono_terminal", "proporción", "Suma de |bT| dividida por la suma de |bT| y de |Rlearn ordinaria|.", "suma de |bT| y |Rlearn ordinaria|", "Cuantifica la dominancia terminal sin cancelación con la recompensa ordinaria final.", "0.000000"),
            ("terminal_bonus_discounted_abs_return_fraction", "Fracción absoluta descontada del bono terminal", "bono_terminal", "proporción", "Masa absoluta descontada de bT dividida por la masa absoluta descontada total, separando la recompensa ordinaria.", "suma descontada de |bT| y |Rlearn ordinaria|", "Cuantifica la influencia terminal desde el inicio del episodio según gamma.", "0.000000"),
            ("ordinary_learning_return", "Retorno ordinario de aprendizaje", "bono_terminal", "recompensa", "Suma de la recompensa asignada ordinaria.", "n_interval", "Excluye el bono terminal.", "0.000000"),
            ("total_learning_return", "Retorno total de aprendizaje", "bono_terminal", "recompensa", "Suma de la recompensa entregada al aprendizaje.", "n_interval", "Incluye el bono terminal.", "0.000000"),
        ):
            add(
                f"{stem}_{agent_id}",
                f"{label} — {agent_id}",
                family,
                "agente",
                agent_id,
                unit,
                definition,
                "Episodio de entrenamiento.",
                denominator,
                "media o mediana según diseño experimental",
                interpretation,
                number_format,
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


def _longest_true_duration(indicator, durations):
    longest = 0.0
    current = 0.0
    for is_true, duration in zip(indicator, durations):
        if bool(is_true):
            current += float(duration)
            longest = max(longest, current)
        else:
            current = 0.0
    return float(longest)


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
    experiment = config["experiment"]
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
    factorial = cooperative["factorial_cooperative"]
    intensity = cooperative["intensity_compensability_cooperative"]
    monotonic = cooperative["monotonic_cooperative"]
    monotonic_active = cooperative["type"] == "monotonic_cooperative"
    potential_cooperative = cooperative["potential_cooperative"]
    potential = potential_cooperative["potential"]
    potential_active = cooperative["type"] == "potential_cooperative"
    episode_report = config["metrics"]["episode_report"]
    relational_thresholds = episode_report["relational_thresholds"]

    row = {
        "run_id": run_id,
        "seed": int(experiment["seed"]),
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
        "steady_state_window_sec": float(episode_report["steady_state_window_sec"]),
        "reward_zero_tolerance": float(episode_report["reward_zero_tolerance"]),
        "lower_tail_quantile": float(episode_report["lower_tail_quantile"]),
        "common_health_threshold": float(relational_thresholds["common_health"]),
        "worst_health_threshold": float(relational_thresholds["worst_health"]),
        "balance_tolerance": float(relational_thresholds["balance"]),
        "synchrony_tolerance": float(relational_thresholds["synchrony"]),
        "progress_tolerance": float(relational_thresholds["progress"]),
        "health_equivalence_tolerance": float(relational_thresholds["equivalence"]),
        "boundary_tolerance": float(relational_thresholds["boundary"]),
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
        "cooperative_type": cooperative["type"],
        "cooperative_reward_mode": (
            potential_cooperative["reward_mode"]
            if cooperative["type"] == "potential_cooperative"
            else monotonic["reward_mode"] if monotonic_active else None
        ),
        "cooperative_method": (
            potential["method"] if cooperative["type"] == "potential_cooperative" else None
        ),
        "cooperative_residual_weight": (
            float(potential_cooperative["residual_weight"])
            if cooperative["type"] == "potential_cooperative"
            else None
        ),
    }

    for controller_id, threshold in episode_report["joint_health_thresholds"].items():
        row[f"joint_health_threshold_{controller_id}"] = float(threshold)

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
        row[f"potential_cost_source_{variable_id}"] = variable_config["cost_source"]
        row[f"potential_health_source_{variable_id}"] = variable_config["health_source"]
        row[f"potential_weight_{variable_id}"] = float(variable_config["weight"])

    row["monotonic_minimum_health_weight"] = float(
        monotonic["minimum_health_weight"]
    )
    row["monotonic_potential_weight"] = float(monotonic["potential_weight"])
    for variable_config in monotonic["variables"]:
        variable_id = variable_config["id"]
        row[f"monotonic_cost_source_{variable_id}"] = variable_config["cost_source"]
        row[f"monotonic_health_source_{variable_id}"] = variable_config[
            "health_source"
        ]
        row[f"monotonic_weight_{variable_id}"] = float(variable_config["weight"])

    post_compose_penalty = monotonic["post_compose_penalty"]
    row["monotonic_post_compose_penalty_enabled"] = bool(
        post_compose_penalty["enabled"]
    )
    row["monotonic_post_compose_penalty_activation_value"] = float(
        post_compose_penalty["activation_value"]
    )
    for variable_config in post_compose_penalty["variables"]:
        variable_id = variable_config["id"]
        row[
            f"monotonic_post_compose_penalty_cost_source_{variable_id}"
        ] = variable_config["cost_source"]
        row[f"monotonic_post_compose_penalty_weight_{variable_id}"] = float(
            variable_config["weight"]
        )

    row["cooperative_norm_order"] = (
        float(potential["p"])
        if potential_active and potential["method"] == "norm-p"
        else None
    )
    row["cooperative_geometric_epsilon"] = (
        float(potential["epsilon_geom_conc"])
        if potential_active and potential["method"] == "geometric_consensus"
        else None
    )
    row["cooperative_deterioration_order"] = (
        float(potential["deterioration_order"])
        if potential_active and potential["method"] == "concurrent_progress"
        else None
    )

    adaptive_weights = potential_cooperative["adaptive_weights"]
    adaptive_weights_active = potential_active and bool(adaptive_weights["enabled"])
    row["adaptive_cooperative_weights_enabled"] = adaptive_weights_active
    row["adaptive_weight_floor"] = (
        float(adaptive_weights["weight_floor"]) if adaptive_weights_active else None
    )
    row["adaptive_deterioration_weight"] = (
        float(adaptive_weights["deterioration_weight"]) if adaptive_weights_active else None
    )
    row["adaptive_weight_smoothing_alpha"] = (
        float(adaptive_weights["smoothing_alpha"]) if adaptive_weights_active else None
    )
    for variable_config in adaptive_weights["need_variables"]:
        variable_id = variable_config["id"]
        row[f"adaptive_need_source_{variable_id}"] = (
            variable_config["source"] if adaptive_weights_active else None
        )

    balance = potential_cooperative["balance"]
    balance_active = potential_active and bool(balance["enabled"])
    row["cooperative_balance_enabled"] = balance_active
    row["cooperative_balance_strength"] = (
        float(balance["strength"]) if balance_active else None
    )
    for variable_config in balance["health_variables"]:
        variable_id = variable_config["id"]
        row[f"cooperative_balance_source_{variable_id}"] = (
            variable_config["source"] if balance_active else None
        )

    synchrony = potential_cooperative["synchrony"]
    synchrony_active = potential_active and bool(synchrony["enabled"])
    row["cooperative_synchrony_enabled"] = synchrony_active
    row["cooperative_synchrony_source"] = (
        synchrony["source"] if synchrony_active else None
    )
    row["cooperative_synchrony_strength"] = (
        float(synchrony["strength"]) if synchrony_active else None
    )

    adaptive_residual = potential_cooperative["adaptive_residual"]
    adaptive_residual_active = potential_active and bool(adaptive_residual["enabled"])
    row["adaptive_residual_enabled"] = adaptive_residual_active
    row["adaptive_residual_source"] = (
        adaptive_residual["source"] if adaptive_residual_active else None
    )
    row["adaptive_residual_weight_min"] = (
        float(adaptive_residual["residual_weight_min"])
        if adaptive_residual_active
        else None
    )
    row["adaptive_residual_weight_max"] = (
        float(adaptive_residual["residual_weight_max"])
        if adaptive_residual_active
        else None
    )
    row["adaptive_residual_gamma"] = (
        float(adaptive_residual["gamma"]) if adaptive_residual_active else None
    )

    for gain_agent_config in agent["gain_agents"]:
        agent_id = gain_agent_config["id"]
        row[f"agent_controller_{agent_id}"] = gain_agent_config["controller"]
        row[f"agent_gain_{agent_id}"] = gain_agent_config["gain"]

    learning_normalization = reward["learning_reward_normalization"]
    row["learning_reward_normalization_enabled"] = bool(
        learning_normalization["enabled"]
    )
    row["learning_reward_normalization_value"] = float(
        learning_normalization["normalization_value"]
    )

    terminal_bonus = reward["terminal_learning_bonus"]
    row["terminal_bonus_enabled"] = bool(terminal_bonus["enabled"])
    row["terminal_bonus_mode"] = terminal_bonus["mode"]
    for reason_name, bonus in terminal_bonus["bonus_values"].items():
        row[f"terminal_bonus_{reason_name}"] = float(bonus)
    stabilization = config["termination"]["stabilization"]
    row["terminal_bonus_stabilization_mode"] = stabilization["mode"]
    row["terminal_bonus_persistence_sec"] = float(stabilization["persistence_sec"])
    for controller_config in control["controllers"]:
        controller_id = controller_config["id"]
        row[f"terminal_bonus_vars_{controller_id}"] = json.dumps(controller_config["vars"])
        for variable in controller_config["vars"]:
            row[f"terminal_bonus_band_lower_{variable}"] = float(stabilization[variable][0])
            row[f"terminal_bonus_band_upper_{variable}"] = float(stabilization[variable][1])

    row["factorial_composition_mode"] = factorial["composition_mode"]
    row["factorial_absolute_weight"] = float(factorial["absolute"]["weight"])
    for factor_name in ("balance", "synchrony", "differential"):
        row[f"factorial_{factor_name}_enabled"] = bool(factorial[factor_name]["enabled"])
        row[f"factorial_{factor_name}_weight"] = float(factorial[factor_name]["weight"])
    row["factorial_common_progress_weight"] = float(factorial["common_progress"]["weight"])
    for variable_config in factorial["variables"]:
        row[f"factorial_health_source_{variable_config['id']}"] = variable_config["health_source"]

    row["compensability_order"] = float(intensity["compensability_order"])
    row["compensability_health_epsilon"] = float(intensity["health_epsilon"])
    for variable_config in intensity["variables"]:
        variable_id = variable_config["id"]
        row[f"compensability_health_source_{variable_id}"] = variable_config["health_source"]
        row[f"compensability_weight_{variable_id}"] = float(variable_config["weight"])

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
        episode_report = config["metrics"]["episode_report"]
        self.steady_state_window_sec = float(episode_report["steady_state_window_sec"])
        self.reward_zero_tolerance = float(episode_report["reward_zero_tolerance"])
        self.lower_tail_quantile = float(episode_report["lower_tail_quantile"])
        if not 0.0 < self.lower_tail_quantile <= 1.0:
            raise ValueError("lower_tail_quantile must be greater than 0 and at most 1")
        self.relational_thresholds = episode_report["relational_thresholds"]
        self.joint_health_thresholds = episode_report["joint_health_thresholds"]
        self.controllers = tuple(config["control"]["controllers"])
        self.gain_agents = tuple(config["agent"]["gain_agents"])
        self.registry_version = METRIC_REGISTRY_VERSION
        self.metric_specs = build_metric_report_specs(config)
        self.metric_signal_names = tuple(spec.canonical_name for spec in self.metric_specs)
        self.episode_columns = EPISODE_IDENTITY_COLUMNS + self.metric_signal_names
        self.catalog_columns = CATALOG_COLUMNS
        self.catalog_rows = tuple(spec.catalog_row() for spec in self.metric_specs)
        self.run_config_row = build_run_config_row(config, run_id)
        self.run_config_columns = tuple(self.run_config_row)

    def compute_episode(self, episode_idx, n_steps, n_intervals, termination_reason_code, training_enabled):
        del episode_idx
        if n_steps <= 0:
            raise ValueError("metric report requires at least one physical step")
        if n_intervals <= 0:
            raise ValueError("metric report requires at least one decision interval")
        values = {}
        success_code = int(self.config["termination"]["reason_codes"]["stabilization_success"])
        success = int(termination_reason_code) == success_code
        values["success"] = int(success)
        values["failure"] = int(not success)
        values["total_reward"] = float(self.episode.total_reward)
        if training_enabled:
            values["total_learning_return"] = float(
                self.episode.total_learning_return
            )
            values["learning_return_rate"] = float(
                self.episode.learning_return_rate
            )
            values["epsilon"] = float(self.episode.epsilon)
            values["learning_rate"] = float(self.episode.learning_rate)
        else:
            values["total_learning_return"] = None
            values["learning_return_rate"] = None
            values["epsilon"] = None
            values["learning_rate"] = None
        for controller_config in self.controllers:
            controller_id = controller_config["id"]
            values[f"total_reward_{controller_id}"] = float(
                getattr(self.episode, f"total_reward_{controller_id}")
            )
            if training_enabled:
                values[f"total_learning_return_{controller_id}"] = float(
                    getattr(
                        self.episode,
                        f"total_learning_return_{controller_id}",
                    )
                )
            else:
                values[f"total_learning_return_{controller_id}"] = None

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
            values["stabilization_start_step"] = None
        else:
            values["stabilization_start_sec"] = float(self.step.t_sec[stable_start_idx + 1])
            values["stabilization_start_step"] = int(stable_start_idx + 1)
        values["stable_block_max_sec"] = float(_longest_true_run(joint_stability) * self.dt_sec)
        values["stable_occupancy"] = float(np.mean(joint_stability))
        joint_losses, joint_duration, joint_rate = _loss_metrics(joint_stability, self.dt_sec)
        values["joint_stability_loss_count"] = joint_losses
        values["joint_post_entry_duration_sec"] = joint_duration
        values["joint_stability_loss_rate_hz"] = joint_rate

        velocity_values = {
            "pendulum": np.asarray(
                self.step.pendulum_velocity_raw[state_slice], dtype=np.float64
            ),
            "cart": np.asarray(self.step.cart_velocity_raw[state_slice], dtype=np.float64),
        }
        values["rms_pendulum_velocity"] = _effective_scale(velocity_values["pendulum"])
        values["max_abs_pendulum_velocity"] = float(
            np.max(np.abs(velocity_values["pendulum"]))
        )
        values["rms_cart_velocity"] = _effective_scale(velocity_values["cart"])
        values["max_abs_cart_velocity"] = float(np.max(np.abs(velocity_values["cart"])))

        steady_state_samples = int(math.ceil(self.steady_state_window_sec / self.dt_sec))
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
            values[f"mean_abs_error_{controller_id}"] = float(iae / duration_sec)
            values[f"rms_error_{controller_id}"] = float(math.sqrt(ise / duration_sec))
            values[f"max_abs_error_{controller_id}"] = float(np.max(np.abs(error)))
            values[f"terminal_error_signed_{controller_id}"] = float(error[-1])
            values[f"terminal_error_abs_{controller_id}"] = float(abs(error[-1]))

            measurement = np.asarray(getattr(self.step, f"{objective}_raw")[control_slice], dtype=np.float64)
            setpoint = float(controller_config["setpoint_raw"])
            overshoot, undershoot, crossed = self._overshoot_undershoot(measurement, setpoint)
            values[f"overshoot_{controller_id}"] = overshoot
            values[f"undershoot_{controller_id}"] = undershoot
            values[f"reference_crossed_{controller_id}"] = crossed

            indicator = local_stability[objective]
            values[f"local_stable_occupancy_{controller_id}"] = float(np.mean(indicator))
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
        values["saturation_block_max_sec"] = float(
            _longest_true_run(saturated) * self.dt_sec
        )
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

        self._compute_health_metrics(values, interval_slice, n_intervals, joint_stability)
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

    def _compute_health_metrics(self, values, interval_slice, n_intervals, joint_stability):
        controller_ids = tuple(controller_config["id"] for controller_config in self.controllers)
        if controller_ids != ("pendulum_angle", "cart_position"):
            raise ValueError(
                "relational metric report requires pendulum_angle and cart_position in canonical order"
            )
        health = np.vstack(
            [
                np.asarray(
                    getattr(self.interval, f"local_health_01_{controller_id}")[interval_slice],
                    dtype=np.float64,
                )
                for controller_id in controller_ids
            ]
        )
        if not np.all(np.isfinite(health)):
            raise ValueError("local health arrays must be finite")
        if np.any(health < 0.0) or np.any(health > 1.0):
            raise ValueError("local health arrays must stay within [0, 1]")

        mean_health = 0.5 * (health[0] + health[1])
        signed_gap = health[0] - health[1]
        absolute_utility = 2.0 * mean_health - 1.0
        worst_health = np.min(health, axis=0)

        equal_weights = np.full_like(health, 0.5)
        arithmetic_health = np.sum(equal_weights * health, axis=0)
        geometric_health = np.sqrt(health[0] * health[1])
        arithmetic_geometric_gap = arithmetic_health - geometric_health
        health_balance = np.ones(n_intervals, dtype=np.float64)
        nonzero_arithmetic = arithmetic_health != 0.0
        health_balance[nonzero_arithmetic] = (
            geometric_health[nonzero_arithmetic] / arithmetic_health[nonzero_arithmetic]
        )

        common_threshold = float(self.relational_thresholds["common_health"])
        worst_threshold = float(self.relational_thresholds["worst_health"])
        balance_tolerance = float(self.relational_thresholds["balance"])
        synchrony_tolerance = float(self.relational_thresholds["synchrony"])
        progress_tolerance = float(self.relational_thresholds["progress"])
        equivalence_tolerance = float(self.relational_thresholds["equivalence"])
        boundary_tolerance = float(self.relational_thresholds["boundary"])

        values["mean_health_mean"] = float(np.mean(mean_health))
        values["mean_health_min"] = float(np.min(mean_health))
        values["mean_health_median"] = float(np.median(mean_health))
        values["mean_health_final"] = float(mean_health[-1])
        values["mean_health_above_threshold_frequency"] = float(
            np.mean(mean_health >= common_threshold)
        )
        values["absolute_common_utility_mean"] = float(np.mean(absolute_utility))

        lower_quantile = float(np.quantile(worst_health, self.lower_tail_quantile))
        lower_tail = worst_health[worst_health <= lower_quantile]
        values["worst_health_mean"] = float(np.mean(worst_health))
        values["worst_health_min"] = float(np.min(worst_health))
        values["worst_health_final"] = float(worst_health[-1])
        values["worst_health_lower_quantile"] = lower_quantile
        values["worst_health_lower_tail_mean"] = float(np.mean(lower_tail))
        values["worst_health_below_threshold_frequency"] = float(
            np.mean(worst_health < worst_threshold)
        )

        gap_abs = np.abs(signed_gap)
        gap_square = signed_gap**2
        values["health_gap_abs_mean"] = float(np.mean(gap_abs))
        values["health_gap_square_mean"] = float(np.mean(gap_square))
        values["health_gap_rms"] = float(math.sqrt(float(np.mean(gap_square))))
        values["health_gap_abs_max"] = float(np.max(gap_abs))
        values["health_gap_abs_final"] = float(gap_abs[-1])
        values["health_gap_within_tolerance_frequency"] = float(
            np.mean(gap_abs <= balance_tolerance)
        )

        values["health_gap_mean"] = float(np.mean(arithmetic_geometric_gap))
        values["health_gap_max"] = float(np.max(arithmetic_geometric_gap))
        values["health_balance_mean"] = float(np.mean(health_balance))
        values["health_balance_min"] = float(np.min(health_balance))
        values["health_balance_final"] = float(health_balance[-1])

        joint_threshold_indicator = (
            (health[0] >= float(self.joint_health_thresholds["pendulum_angle"]))
            & (health[1] >= float(self.joint_health_thresholds["cart_position"]))
        )
        values["joint_health_threshold_frequency"] = float(
            np.mean(joint_threshold_indicator)
        )

        transition_count = max(0, n_intervals - 1)
        values["health_transition_count"] = transition_count
        if transition_count == 0:
            values["concurrent_progress_count"] = 0
            values["concurrent_progress_frequency"] = None
            values["deterioration_count"] = 0
            values["deterioration_frequency"] = None
            values["compensatory_transition_count"] = 0
            values["compensatory_transition_frequency"] = None
            temporal_names = (
                "synchrony_mismatch_abs_mean",
                "synchrony_mismatch_square_mean",
                "synchrony_mismatch_abs_max",
                "synchrony_mismatch_abs_final",
                "synchrony_within_tolerance_frequency",
                "common_progress_mean",
                "common_progress_positive_mean",
                "common_progress_negative_mean",
                "common_progress_positive_frequency",
                "common_progress_negative_frequency",
                "common_progress_zero_frequency",
                "common_progress_return_discounted",
                "common_progress_weighted_return_discounted",
                "common_progress_given_synchrony_mean",
                "health_gap_oriented_change_mean",
                "health_gap_closing_frequency",
                "health_gap_expanding_frequency",
                "health_gap_unchanged_frequency",
                "health_gap_zero_crossing_count",
                "health_gap_sign_change_count",
                "apparent_equilibrium_frequency",
                "apparent_equilibrium_block_max_sec",
            )
            for name in temporal_names:
                values[name] = None
            for category in (
                "joint_improvement",
                "joint_deterioration",
                "compensation_pendulum_improves",
                "compensation_cart_improves",
                "unilateral_pendulum_change",
                "unilateral_cart_change",
                "stagnation",
            ):
                values[f"transition_{category}_frequency"] = None
            return

        health_delta = health[:, 1:] - health[:, :-1]
        delta_gap = signed_gap[1:] - signed_gap[:-1]
        synchrony_mismatch = np.abs(delta_gap) / 2.0
        synchrony_mismatch_square = (delta_gap / 2.0) ** 2
        common_progress = 0.5 * (health_delta[0] + health_delta[1])
        concurrent = np.all(health_delta > progress_tolerance, axis=0)
        deterioration = np.any(health_delta < -progress_tolerance, axis=0)
        compensatory = health_delta[0] * health_delta[1] < -(progress_tolerance**2)

        concurrent_count = int(np.sum(concurrent))
        deterioration_count = int(np.sum(deterioration))
        compensatory_count = int(np.sum(compensatory))
        values["concurrent_progress_count"] = concurrent_count
        values["concurrent_progress_frequency"] = float(
            concurrent_count / transition_count
        )
        values["deterioration_count"] = deterioration_count
        values["deterioration_frequency"] = float(
            deterioration_count / transition_count
        )
        values["compensatory_transition_count"] = compensatory_count
        values["compensatory_transition_frequency"] = float(
            compensatory_count / transition_count
        )

        values["synchrony_mismatch_abs_mean"] = float(np.mean(synchrony_mismatch))
        values["synchrony_mismatch_square_mean"] = float(
            np.mean(synchrony_mismatch_square)
        )
        values["synchrony_mismatch_abs_max"] = float(np.max(synchrony_mismatch))
        values["synchrony_mismatch_abs_final"] = float(synchrony_mismatch[-1])
        synchronized = synchrony_mismatch <= synchrony_tolerance
        values["synchrony_within_tolerance_frequency"] = float(np.mean(synchronized))

        progress_positive = np.maximum(common_progress, 0.0)
        progress_negative = np.maximum(-common_progress, 0.0)
        values["common_progress_mean"] = float(np.mean(common_progress))
        values["common_progress_positive_mean"] = float(np.mean(progress_positive))
        values["common_progress_negative_mean"] = float(np.mean(progress_negative))
        values["common_progress_positive_frequency"] = float(
            np.mean(common_progress > progress_tolerance)
        )
        values["common_progress_negative_frequency"] = float(
            np.mean(common_progress < -progress_tolerance)
        )
        values["common_progress_zero_frequency"] = float(
            np.mean(np.abs(common_progress) <= progress_tolerance)
        )
        discount_factor = float(self.config["agent"]["params"]["discount_factor"])
        transition_discount = discount_factor ** np.arange(transition_count)
        values["common_progress_return_discounted"] = float(
            np.sum(transition_discount * common_progress)
        )
        cooperative_config = self.config["reward"]["cooperative"]
        cooperative_type = cooperative_config["type"]
        factorial_config = cooperative_config["factorial_cooperative"]
        if (
            cooperative_config["enabled"]
            and cooperative_type == "factorial_cooperative"
            and factorial_config["composition_mode"] == "common_progress"
        ):
            weighted_progress = np.asarray(
                self.interval.factorial_common_progress_contribution_signed[1:n_intervals],
                dtype=np.float64,
            )
            values["common_progress_weighted_return_discounted"] = float(
                np.sum(transition_discount * weighted_progress)
            )
        else:
            values["common_progress_weighted_return_discounted"] = None
        if np.any(synchronized):
            values["common_progress_given_synchrony_mean"] = float(
                np.mean(common_progress[synchronized])
            )
        else:
            values["common_progress_given_synchrony_mean"] = None

        values["health_gap_oriented_change_mean"] = float(
            np.mean(signed_gap[:-1] * delta_gap)
        )
        previous_gap_abs = gap_abs[:-1]
        current_gap_abs = gap_abs[1:]
        closing = current_gap_abs < previous_gap_abs - boundary_tolerance
        expanding = current_gap_abs > previous_gap_abs + boundary_tolerance
        unchanged = ~(closing | expanding)
        values["health_gap_closing_frequency"] = float(np.mean(closing))
        values["health_gap_expanding_frequency"] = float(np.mean(expanding))
        values["health_gap_unchanged_frequency"] = float(np.mean(unchanged))
        values["health_gap_zero_crossing_count"] = int(
            np.sum(
                (signed_gap[:-1] * signed_gap[1:] <= 0.0)
                & ~(
                    (np.abs(signed_gap[:-1]) <= equivalence_tolerance)
                    & (np.abs(signed_gap[1:]) <= equivalence_tolerance)
                )
            )
        )
        values["health_gap_sign_change_count"] = int(
            np.sum(signed_gap[:-1] * signed_gap[1:] < 0.0)
        )

        categories = np.full(transition_count, "stagnation", dtype=object)
        first_delta = health_delta[0]
        second_delta = health_delta[1]
        first_positive = first_delta > progress_tolerance
        first_negative = first_delta < -progress_tolerance
        second_positive = second_delta > progress_tolerance
        second_negative = second_delta < -progress_tolerance
        first_zero = ~(first_positive | first_negative)
        second_zero = ~(second_positive | second_negative)
        categories[first_positive & second_positive] = "joint_improvement"
        categories[first_negative & second_negative] = "joint_deterioration"
        categories[first_positive & second_negative] = (
            "compensation_pendulum_improves"
        )
        categories[first_negative & second_positive] = "compensation_cart_improves"
        categories[(first_positive | first_negative) & second_zero] = (
            "unilateral_pendulum_change"
        )
        categories[first_zero & (second_positive | second_negative)] = (
            "unilateral_cart_change"
        )
        for category in (
            "joint_improvement",
            "joint_deterioration",
            "compensation_pendulum_improves",
            "compensation_cart_improves",
            "unilateral_pendulum_change",
            "unilateral_cart_change",
            "stagnation",
        ):
            values[f"transition_{category}_frequency"] = float(
                np.mean(categories == category)
            )

        interval_end_indices = np.asarray(
            self.interval.interval_end_step_idx[1:n_intervals], dtype=np.int64
        )
        physical_state_indices = np.minimum(
            interval_end_indices + 1, len(joint_stability)
        )
        current_joint_stability = joint_stability[physical_state_indices - 1]
        apparent_equilibrium = (
            (np.abs(common_progress) <= progress_tolerance)
            & (synchrony_mismatch <= synchrony_tolerance)
            & ~current_joint_stability
        )
        values["apparent_equilibrium_frequency"] = float(
            np.mean(apparent_equilibrium)
        )
        interval_durations = np.asarray(
            self.interval.interval_duration_sec[1:n_intervals], dtype=np.float64
        )
        values["apparent_equilibrium_block_max_sec"] = _longest_true_duration(
            apparent_equilibrium, interval_durations
        )

    def _compute_reward_metrics(self, values, interval_slice, n_intervals):
        zero_tolerance = self.reward_zero_tolerance
        boundary_tolerance = float(self.relational_thresholds["boundary"])
        equivalence_tolerance = float(self.relational_thresholds["equivalence"])
        cooperative_reward = np.asarray(
            self.interval.cooperative_reward_signed[interval_slice], dtype=np.float64
        )
        values["cooperative_reward_positive_proportion"] = float(
            np.sum(cooperative_reward > zero_tolerance) / n_intervals
        )
        values["cooperative_reward_zero_proportion"] = float(
            np.sum(np.abs(cooperative_reward) <= zero_tolerance) / n_intervals
        )
        values["cooperative_reward_negative_proportion"] = float(
            np.sum(cooperative_reward < -zero_tolerance) / n_intervals
        )
        values["cooperative_reward_variance"] = _population_variance(cooperative_reward)
        values["cooperative_reward_effective_scale"] = _effective_scale(cooperative_reward)

        composition_weights = self.config["reward"]["composition"]["weights"]
        local_weight = float(composition_weights["local"])
        cooperative_weight = float(composition_weights["cooperative"])
        cooperative_contribution = cooperative_weight * cooperative_reward
        values["cooperative_contribution_variance"] = _population_variance(cooperative_contribution)
        values["cooperative_contribution_effective_scale"] = _effective_scale(cooperative_contribution)

        controller_rewards = {}
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

            cooperative_controller_contribution = np.asarray(
                getattr(
                    self.interval,
                    f"cooperative_contribution_signed_{controller_id}",
                )[interval_slice],
                dtype=np.float64,
            )
            covariance = float(
                np.mean(
                    (local_contribution - np.mean(local_contribution))
                    * (
                        cooperative_controller_contribution
                        - np.mean(cooperative_controller_contribution)
                    )
                )
            )
            opposite_sign = (
                local_contribution * cooperative_controller_contribution
                < -(zero_tolerance**2)
            )
            same_sign = (
                local_contribution * cooperative_controller_contribution
                > zero_tolerance**2
            )
            values[
                f"local_cooperative_contribution_covariance_{controller_id}"
            ] = covariance
            values[
                f"local_cooperative_opposite_sign_frequency_{controller_id}"
            ] = float(np.mean(opposite_sign))
            values[
                f"local_cooperative_same_sign_frequency_{controller_id}"
            ] = float(np.mean(same_sign))

            local_cost = np.asarray(
                getattr(self.interval, f"local_cost_01_{controller_id}")[interval_slice],
                dtype=np.float64,
            )
            local_health = np.asarray(
                getattr(self.interval, f"local_health_01_{controller_id}")[interval_slice],
                dtype=np.float64,
            )
            ordinary_reward = np.asarray(
                getattr(self.interval, f"reward_controller_{controller_id}")[
                    interval_slice
                ],
                dtype=np.float64,
            )
            controller_rewards[controller_id] = ordinary_reward
            reward_clip_excess = np.asarray(
                getattr(self.interval, f"reward_clip_excess_{controller_id}")[
                    interval_slice
                ],
                dtype=np.float64,
            )
            reward_clipped = np.asarray(
                getattr(self.interval, f"reward_clipped_{controller_id}")[
                    interval_slice
                ],
                dtype=np.float64,
            )
            values[f"local_cost_lower_boundary_frequency_{controller_id}"] = float(
                np.mean(local_cost <= boundary_tolerance)
            )
            values[f"local_cost_upper_boundary_frequency_{controller_id}"] = float(
                np.mean(local_cost >= 1.0 - boundary_tolerance)
            )
            values[f"local_health_lower_boundary_frequency_{controller_id}"] = float(
                np.mean(local_health <= boundary_tolerance)
            )
            values[f"local_health_upper_boundary_frequency_{controller_id}"] = float(
                np.mean(local_health >= 1.0 - boundary_tolerance)
            )
            values[f"reward_negative_boundary_frequency_{controller_id}"] = float(
                np.mean(np.abs(ordinary_reward + 1.0) <= boundary_tolerance)
            )
            values[f"reward_positive_boundary_frequency_{controller_id}"] = float(
                np.mean(np.abs(ordinary_reward - 1.0) <= boundary_tolerance)
            )
            values[f"reward_clipped_frequency_{controller_id}"] = float(
                np.mean(reward_clipped != 0.0)
            )
            values[f"reward_clip_excess_max_{controller_id}"] = float(
                np.max(reward_clip_excess)
            )

        pendulum_reward = controller_rewards["pendulum_angle"]
        cart_reward = controller_rewards["cart_position"]
        reward_common = 0.5 * (pendulum_reward + cart_reward)
        reward_differential = 0.5 * (pendulum_reward - cart_reward)
        signed_gap = np.asarray(
            self.interval.health_gap_signed[interval_slice], dtype=np.float64
        )
        values["reward_common_mean"] = float(np.mean(reward_common))
        values["reward_differential_abs_mean"] = float(
            np.mean(np.abs(reward_differential))
        )
        gap_denominator = float(np.sum(np.abs(signed_gap)))
        if gap_denominator == 0.0:
            values["reward_differential_gain_observed"] = None
        else:
            values["reward_differential_gain_observed"] = float(
                np.sum(np.abs(reward_differential)) / gap_denominator
            )
        values["reward_controller_separation_abs_mean"] = float(
            np.mean(np.abs(pendulum_reward - cart_reward))
        )
        discount_factor = float(self.config["agent"]["params"]["discount_factor"])
        interval_discount = discount_factor ** np.arange(n_intervals)
        values["reward_common_return_discounted"] = float(
            np.sum(interval_discount * reward_common)
        )
        values["reward_differential_return_discounted"] = float(
            np.sum(interval_discount * reward_differential)
        )
        values["reward_differential_abs_return_discounted"] = float(
            np.sum(interval_discount * np.abs(reward_differential))
        )
        values["health_advantage_pendulum_frequency"] = float(
            np.mean(signed_gap > equivalence_tolerance)
        )
        values["health_advantage_cart_frequency"] = float(
            np.mean(signed_gap < -equivalence_tolerance)
        )
        values["health_equivalence_frequency"] = float(
            np.mean(np.abs(signed_gap) <= equivalence_tolerance)
        )

        cooperative_config = self.config["reward"]["cooperative"]
        cooperative_type = cooperative_config["type"]
        cooperative_active = bool(cooperative_config["enabled"])
        if cooperative_active and cooperative_type == "intensity_compensability_cooperative":
            generalized_welfare = np.asarray(
                self.interval.generalized_welfare_01[interval_slice],
                dtype=np.float64,
            )
            compensability_contrast = np.asarray(
                self.interval.collective_noncompensability_loss_01[interval_slice],
                dtype=np.float64,
            )
            values["collective_noncompensability_loss_mean"] = float(
                np.mean(compensability_contrast)
            )
            values["collective_noncompensability_weighted_loss_mean"] = float(
                cooperative_weight * np.mean(compensability_contrast)
            )
            values["generalized_welfare_mean"] = float(
                np.mean(generalized_welfare)
            )
            values["generalized_welfare_min"] = float(np.min(generalized_welfare))
            values["generalized_welfare_final"] = float(generalized_welfare[-1])
        else:
            values["collective_noncompensability_loss_mean"] = None
            values["collective_noncompensability_weighted_loss_mean"] = None
            values["generalized_welfare_mean"] = None
            values["generalized_welfare_min"] = None
            values["generalized_welfare_final"] = None

        monotonic_config = cooperative_config["monotonic_cooperative"]
        monotonic_active = (
            cooperative_active and cooperative_type == "monotonic_cooperative"
        )
        monotonic_reductions = {
            "monotonic_weighted_mean_health_mean": (
                "monotonic_weighted_mean_health_01",
                "mean",
            ),
            "monotonic_minimum_health_mean": (
                "monotonic_minimum_health_01",
                "mean",
            ),
            "monotonic_health_gap_mean": ("monotonic_health_gap_01", "mean"),
            "monotonic_utility_mean": ("monotonic_utility_01", "mean"),
            "monotonic_utility_min": ("monotonic_utility_01", "min"),
            "monotonic_utility_final": ("monotonic_utility_01", "final"),
            "monotonic_utility_post_penalty_mean": (
                "monotonic_utility_post_penalty_01",
                "mean",
            ),
            "monotonic_potential_shaping_mean": (
                "monotonic_potential_shaping_signed",
                "mean",
            ),
            "monotonic_base_reward_mean": (
                "monotonic_base_reward_signed",
                "mean",
            ),
            "monotonic_post_penalty_reward_mean": (
                "monotonic_post_penalty_reward_signed",
                "mean",
            ),
        }
        for metric_name, (source_name, reduction) in monotonic_reductions.items():
            if monotonic_active:
                source_values = np.asarray(
                    getattr(self.interval, source_name)[interval_slice],
                    dtype=np.float64,
                )
                if reduction == "mean":
                    values[metric_name] = float(np.mean(source_values))
                elif reduction == "min":
                    values[metric_name] = float(np.min(source_values))
                else:
                    values[metric_name] = float(source_values[-1])
            else:
                values[metric_name] = None

        penalty_active = monotonic_active and bool(
            monotonic_config["post_compose_penalty"]["enabled"]
        )
        penalty_reductions = {
            "monotonic_post_compose_penalty_activation_mean": (
                "monotonic_post_compose_penalty_activation_01"
            ),
            "monotonic_post_compose_penalty_cost_mean": (
                "monotonic_post_compose_penalty_cost_01"
            ),
            "monotonic_post_compose_penalty_mean": (
                "monotonic_post_compose_penalty_01"
            ),
        }
        for metric_name, source_name in penalty_reductions.items():
            if penalty_active:
                values[metric_name] = float(
                    np.mean(
                        np.asarray(
                            getattr(self.interval, source_name)[interval_slice],
                            dtype=np.float64,
                        )
                    )
                )
            else:
                values[metric_name] = None

        for variable_config in monotonic_config["post_compose_penalty"][
            "variables"
        ]:
            variable_id = variable_config["id"]
            metric_name = (
                "monotonic_post_compose_penalty_contribution_mean_"
                f"{variable_id}"
            )
            if penalty_active:
                values[metric_name] = float(
                    np.mean(
                        np.asarray(
                            getattr(
                                self.interval,
                                "monotonic_post_compose_penalty_contribution_01_"
                                f"{variable_id}",
                            )[interval_slice],
                            dtype=np.float64,
                        )
                    )
                )
            else:
                values[metric_name] = None

        cooperative_differential = np.asarray(
            self.interval.cooperative_reward_differential_signed[interval_slice],
            dtype=np.float64,
        )
        cooperative_by_controller = {
            controller_id: np.asarray(
                getattr(self.interval, f"cooperative_reward_signed_{controller_id}")[
                    interval_slice
                ],
                dtype=np.float64,
            )
            for controller_id in controller_rewards
        }
        values["cooperative_mode_sum_identity_max_error"] = float(
            np.max(
                np.abs(
                    cooperative_by_controller["pendulum_angle"]
                    + cooperative_by_controller["cart_position"]
                    - 2.0 * cooperative_reward
                )
            )
        )
        values["cooperative_mode_difference_identity_max_error"] = float(
            np.max(
                np.abs(
                    cooperative_by_controller["pendulum_angle"]
                    - cooperative_by_controller["cart_position"]
                    - 2.0 * cooperative_differential
                )
            )
        )
        if cooperative_active and cooperative_type == "factorial_cooperative":
            factorial_config = cooperative_config["factorial_cooperative"]
            factorial_common = (
                np.asarray(
                    self.interval.factorial_absolute_contribution_signed[
                        interval_slice
                    ],
                    dtype=np.float64,
                )
                + np.asarray(
                    self.interval.factorial_balance_contribution_signed[
                        interval_slice
                    ],
                    dtype=np.float64,
                )
                + np.asarray(
                    self.interval.factorial_synchrony_contribution_signed[
                        interval_slice
                    ],
                    dtype=np.float64,
                )
                + np.asarray(
                    self.interval.factorial_common_progress_contribution_signed[
                        interval_slice
                    ],
                    dtype=np.float64,
                )
            )
            factorial_differential = np.asarray(
                self.interval.factorial_differential_contribution_signed[
                    interval_slice
                ],
                dtype=np.float64,
            )
            values["factorial_common_identity_max_error"] = float(
                np.max(np.abs(cooperative_reward - factorial_common))
            )
            if bool(factorial_config["differential"]["enabled"]):
                values["factorial_differential_identity_max_error"] = float(
                    np.max(
                        np.abs(
                            cooperative_differential - factorial_differential
                        )
                    )
                )
            else:
                values["factorial_differential_identity_max_error"] = None
        else:
            values["factorial_common_identity_max_error"] = None
            values["factorial_differential_identity_max_error"] = None

        if cooperative_active and cooperative_type == "intensity_compensability_cooperative":
            values["intensity_differential_identity_max_error"] = float(
                np.max(
                    np.abs(reward_differential - local_weight * signed_gap)
                )
            )
        else:
            values["intensity_differential_identity_max_error"] = None

        active_cooperative_type = cooperative_type if cooperative_active else None
        self._compute_component_scale_metrics(
            values,
            interval_slice,
            active_cooperative_type,
        )
        values["terminal_bonus"] = float(
            np.sum(np.asarray(self.interval.terminal_bonus[interval_slice], dtype=np.float64))
        )
        for controller_config in self.controllers:
            controller_id = controller_config["id"]
            for stem in (
                "terminal_inside_bands",
                "terminal_band_duration_sec",
                "terminal_band_qualified",
                "terminal_bonus_eligible",
            ):
                name = f"{stem}_{controller_id}"
                values[f"{stem}_final_{controller_id}"] = float(
                    getattr(self.interval, name)[interval_slice][-1]
                )

    def _compute_component_scale_metrics(
        self, values, interval_slice, cooperative_type
    ):
        common_sources = (
            "absolute_common_utility_signed",
            "balance_utility_signed",
            "synchrony_utility_signed",
            "common_progress_signed",
            "cooperative_reward_signed",
            "cooperative_reward_differential_signed",
        )
        factorial_sources = (
            "factorial_absolute_contribution_signed",
            "factorial_balance_contribution_signed",
            "factorial_synchrony_contribution_signed",
            "factorial_differential_contribution_signed",
            "factorial_common_progress_contribution_signed",
        )
        intensity_sources = (
            "generalized_welfare_01",
            "collective_noncompensability_loss_01",
            "collective_noncompensability_weighted_loss_01",
        )
        for source_name in common_sources:
            source_values = np.asarray(
                getattr(self.interval, source_name)[interval_slice],
                dtype=np.float64,
            )
            self._write_component_stats(values, source_name, source_values)
        for source_name in factorial_sources:
            if (
                cooperative_type == "factorial_cooperative"
                and self._factorial_component_is_applicable(source_name)
            ):
                source_values = np.asarray(
                    getattr(self.interval, source_name)[interval_slice],
                    dtype=np.float64,
                )
                self._write_component_stats(values, source_name, source_values)
            else:
                self._write_component_stats(values, source_name, None)
        for source_name in intensity_sources:
            if cooperative_type == "intensity_compensability_cooperative":
                source_values = np.asarray(
                    getattr(self.interval, source_name)[interval_slice],
                    dtype=np.float64,
                )
                self._write_component_stats(values, source_name, source_values)
            else:
                self._write_component_stats(values, source_name, None)

        for controller_config in self.controllers:
            controller_id = controller_config["id"]
            for source_name in (
                f"local_reward_signed_{controller_id}",
                f"local_contribution_signed_{controller_id}",
                f"differential_utility_signed_{controller_id}",
                f"cooperative_reward_signed_{controller_id}",
                f"cooperative_contribution_signed_{controller_id}",
                f"reward_controller_{controller_id}",
            ):
                source_values = np.asarray(
                    getattr(self.interval, source_name)[interval_slice],
                    dtype=np.float64,
                )
                self._write_component_stats(values, source_name, source_values)

    def _factorial_component_is_applicable(self, source_name):
        factorial_config = self.config["reward"]["cooperative"][
            "factorial_cooperative"
        ]
        if source_name == "factorial_absolute_contribution_signed":
            return True
        if source_name == "factorial_balance_contribution_signed":
            return bool(factorial_config["balance"]["enabled"])
        if source_name == "factorial_synchrony_contribution_signed":
            return bool(factorial_config["synchrony"]["enabled"])
        if source_name == "factorial_differential_contribution_signed":
            return bool(factorial_config["differential"]["enabled"])
        if source_name == "factorial_common_progress_contribution_signed":
            return factorial_config["composition_mode"] == "common_progress"
        raise KeyError(f"Unknown factorial component source: {source_name}")

    def _write_component_stats(self, values, source_name, source_values):
        stat_names = (
            "mean",
            "variance",
            "rms",
            "abs_mean",
            "positive_proportion",
            "negative_proportion",
            "zero_proportion",
        )
        if source_values is None:
            for stat_name in stat_names:
                values[f"{source_name}_{stat_name}"] = None
            return
        values[f"{source_name}_mean"] = float(np.mean(source_values))
        values[f"{source_name}_variance"] = _population_variance(source_values)
        values[f"{source_name}_rms"] = _effective_scale(source_values)
        values[f"{source_name}_abs_mean"] = float(np.mean(np.abs(source_values)))
        values[f"{source_name}_positive_proportion"] = float(
            np.mean(source_values > self.reward_zero_tolerance)
        )
        values[f"{source_name}_negative_proportion"] = float(
            np.mean(source_values < -self.reward_zero_tolerance)
        )
        values[f"{source_name}_zero_proportion"] = float(
            np.mean(np.abs(source_values) <= self.reward_zero_tolerance)
        )

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
            if training_enabled:
                values[f"nonzero_gain_request_count_{agent_id}"] = (
                    nonzero_request_count
                )
                values[f"blocked_gain_action_count_{agent_id}"] = blocked_count
                if nonzero_request_count == 0:
                    values[f"blocked_gain_action_frequency_{agent_id}"] = None
                else:
                    values[f"blocked_gain_action_frequency_{agent_id}"] = float(
                        blocked_count / nonzero_request_count
                    )
            else:
                values[f"nonzero_gain_request_count_{agent_id}"] = None
                values[f"blocked_gain_action_count_{agent_id}"] = None
                values[f"blocked_gain_action_frequency_{agent_id}"] = None
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
                total_learning_reward = np.asarray(
                    getattr(self.interval, f"reward_for_learning_{agent_id}")[
                        interval_slice
                    ],
                    dtype=np.float64,
                )
                terminal_bonus = np.asarray(
                    getattr(
                        self.interval,
                        f"terminal_bonus_{gain_agent_config['controller']}",
                    )[interval_slice],
                    dtype=np.float64,
                )
                ordinary_learning_reward = total_learning_reward - terminal_bonus
                values[f"ordinary_learning_return_{agent_id}"] = float(
                    np.sum(ordinary_learning_reward)
                )
                values[f"total_learning_return_{agent_id}"] = float(
                    np.sum(total_learning_reward)
                )
                absolute_learning_denominator = float(
                    np.sum(np.abs(total_learning_reward))
                )
                if absolute_learning_denominator == 0.0:
                    values[f"terminal_bonus_abs_return_fraction_{agent_id}"] = None
                else:
                    values[f"terminal_bonus_abs_return_fraction_{agent_id}"] = float(
                        abs(float(np.sum(terminal_bonus)))
                        / absolute_learning_denominator
                    )
                terminal_bonus_abs_mass = float(np.sum(np.abs(terminal_bonus)))
                ordinary_learning_abs_mass = float(
                    np.sum(np.abs(ordinary_learning_reward))
                )
                separated_absolute_denominator = (
                    terminal_bonus_abs_mass + ordinary_learning_abs_mass
                )
                if separated_absolute_denominator == 0.0:
                    values[
                        f"terminal_bonus_separated_abs_return_fraction_{agent_id}"
                    ] = None
                else:
                    values[
                        f"terminal_bonus_separated_abs_return_fraction_{agent_id}"
                    ] = float(
                        terminal_bonus_abs_mass / separated_absolute_denominator
                    )

                discount_factor = float(
                    self.config["agent"]["params"]["discount_factor"]
                )
                interval_discount = discount_factor ** np.arange(n_intervals)
                discounted_terminal_bonus_abs_mass = float(
                    np.sum(interval_discount * np.abs(terminal_bonus))
                )
                discounted_ordinary_learning_abs_mass = float(
                    np.sum(interval_discount * np.abs(ordinary_learning_reward))
                )
                discounted_absolute_denominator = (
                    discounted_terminal_bonus_abs_mass
                    + discounted_ordinary_learning_abs_mass
                )
                if discounted_absolute_denominator == 0.0:
                    values[
                        f"terminal_bonus_discounted_abs_return_fraction_{agent_id}"
                    ] = None
                else:
                    values[
                        f"terminal_bonus_discounted_abs_return_fraction_{agent_id}"
                    ] = float(
                        discounted_terminal_bonus_abs_mass
                        / discounted_absolute_denominator
                    )
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
                action_probabilities = np.asarray(
                    [
                        np.mean(actions == -1),
                        np.mean(actions == 0),
                        np.mean(actions == 1),
                    ],
                    dtype=np.float64,
                )
                values[f"action_decrease_frequency_{agent_id}"] = float(
                    action_probabilities[0]
                )
                values[f"action_maintain_frequency_{agent_id}"] = float(
                    action_probabilities[1]
                )
                values[f"action_increase_frequency_{agent_id}"] = float(
                    action_probabilities[2]
                )
                nonzero_probabilities = action_probabilities[
                    action_probabilities > 0.0
                ]
                values[f"action_entropy_{agent_id}"] = float(
                    -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))
                )
                exploratory = np.asarray(
                    getattr(self.interval, f"is_exploratory_{agent_id}")[
                        interval_slice
                    ],
                    dtype=np.float64,
                )
                values[f"exploratory_action_frequency_{agent_id}"] = float(
                    np.mean(exploratory)
                )
                td_error = np.asarray(
                    getattr(self.interval, f"td_error_{agent_id}")[interval_slice],
                    dtype=np.float64,
                )
                values[f"td_error_mean_abs_{agent_id}"] = float(
                    np.mean(np.abs(td_error))
                )
                values[f"td_error_rms_{agent_id}"] = _effective_scale(td_error)
                values[f"td_error_max_abs_{agent_id}"] = float(
                    np.max(np.abs(td_error))
                )
                learning = learning_by_agent[agent_id]
                values[f"q_delta_rms_{agent_id}"] = learning["q_delta_rms"]
                values[f"q_value_mean_{agent_id}"] = learning["q_value_mean"]
                values[f"q_value_std_{agent_id}"] = learning["q_value_std"]
                values[f"q_value_min_{agent_id}"] = learning["q_value_min"]
                values[f"q_value_max_{agent_id}"] = learning["q_value_max"]
                values[f"policy_stability_{agent_id}"] = learning["policy_stability"]
                values[f"visited_state_count_{agent_id}"] = learning["visited_state_count"]
                values[f"visited_state_frequency_{agent_id}"] = learning[
                    "visited_state_frequency"
                ]
                q_delta_values.append(learning["q_delta_rms"])
                policy_stability_values.append(learning["policy_stability"])
            else:
                values[f"ordinary_learning_return_{agent_id}"] = None
                values[f"total_learning_return_{agent_id}"] = None
                values[f"terminal_bonus_abs_return_fraction_{agent_id}"] = None
                values[
                    f"terminal_bonus_separated_abs_return_fraction_{agent_id}"
                ] = None
                values[
                    f"terminal_bonus_discounted_abs_return_fraction_{agent_id}"
                ] = None
                values[f"action_change_count_{agent_id}"] = None
                values[f"action_change_frequency_{agent_id}"] = None
                values[f"action_decrease_frequency_{agent_id}"] = None
                values[f"action_maintain_frequency_{agent_id}"] = None
                values[f"action_increase_frequency_{agent_id}"] = None
                values[f"action_entropy_{agent_id}"] = None
                values[f"exploratory_action_frequency_{agent_id}"] = None
                values[f"td_error_mean_abs_{agent_id}"] = None
                values[f"td_error_rms_{agent_id}"] = None
                values[f"td_error_max_abs_{agent_id}"] = None
                values[f"q_delta_rms_{agent_id}"] = None
                values[f"q_value_mean_{agent_id}"] = None
                values[f"q_value_std_{agent_id}"] = None
                values[f"q_value_min_{agent_id}"] = None
                values[f"q_value_max_{agent_id}"] = None
                values[f"policy_stability_{agent_id}"] = None
                values[f"visited_state_count_{agent_id}"] = None
                values[f"visited_state_frequency_{agent_id}"] = None

        if training_enabled:
            values["q_delta_rms_agent_mean"] = float(np.mean(q_delta_values))
            values["policy_stability_agent_mean"] = float(np.mean(policy_stability_values))
        else:
            values["q_delta_rms_agent_mean"] = None
            values["policy_stability_agent_mean"] = None
