from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from metrics.metric_report import build_metric_report_signal_names


@dataclass(frozen=True)
class OutputRef:
    name: str
    array: np.ndarray


BASE_STEP_SIGNALS = [
    "t_sec",
    "cart_position_raw",
    "cart_velocity_raw",
    "pendulum_angle_raw",
    "pendulum_velocity_raw",
    "cart_position_norm",
    "cart_velocity_norm",
    "pendulum_angle_norm",
    "pendulum_velocity_norm",
    "cart_force",
    "u_total_raw",
    "u_total",
    "delta_u_total",
    "is_saturated_global",
    "saturation_excess",
    "saturation_proportion_global",
]

PID_RAW_NORM_STEMS = [
    "error",
    "derivative_error",
    "integral_error",
    "p_term",
    "i_term",
    "d_term",
    "u_demand",
    "delta_u_demand",
    "u_alloc",
    "delta_u_alloc",
    "u_unallocated",
]

PID_INCREMENTAL_RAW_NORM_STEMS = [
    "p_delta_term",
    "i_delta_term",
    "d_delta_term",
    "delta_u_incremental",
]

BASE_INTERVAL_SIGNALS = [
    "interval_id",
    "interval_start_step_idx",
    "interval_end_step_idx",
    "interval_t_start_sec",
    "interval_t_end_sec",
    "interval_duration_sec",
    "interval_n_steps",
    "terminated",
    "termination_reason_code",
    "interval_mean_abs_error_pendulum_angle_raw",
    "interval_mean_abs_error_cart_position_raw",
    "interval_rms_pendulum_velocity_raw",
    "interval_rms_cart_velocity_raw",
    "interval_control_mean_square",
    "interval_saturation_frequency",
    "mean_health_01",
    "health_gap_signed",
    "delta_health_gap_signed",
    "absolute_common_utility_signed",
    "balance_utility_signed",
    "synchrony_utility_signed",
    "absolute_common_utility_normalized_signed",
    "balance_utility_normalized_signed",
    "synchrony_utility_normalized_signed",
    "common_progress_signed",
    "factorial_absolute_contribution_signed",
    "factorial_balance_contribution_signed",
    "factorial_synchrony_contribution_signed",
    "factorial_differential_contribution_signed",
    "factorial_common_progress_contribution_signed",
    "generalized_welfare_01",
    "collective_noncompensability_loss_01",
    "collective_noncompensability_weighted_loss_01",
    "monotonic_weighted_mean_health_01",
    "monotonic_minimum_health_01",
    "monotonic_health_gap_01",
    "monotonic_utility_01",
    "monotonic_utility_post_penalty_01",
    "monotonic_potential_shaping_signed",
    "monotonic_base_reward_signed",
    "monotonic_post_penalty_reward_signed",
    "monotonic_post_compose_penalty_activation_01",
    "monotonic_post_compose_penalty_cost_01",
    "monotonic_post_compose_penalty_01",
    "cooperative_cost_01",
    "cooperative_reward_signed",
    "cooperative_reward_differential_signed",
    "cooperative_positive_progress_01",
    "cooperative_deterioration_01",
    "cooperative_progress_signed",
    "cooperative_potential_base_01",
    "health_balance_01",
    "balance_factor_01",
    "synchrony_factor_01",
    "collective_potential_01",
    "cooperative_potential_01",
    "delta_cooperative_potential_signed",
    "cooperative_progress_residual_signed",
    "residual_weight_effective",
    "reward_global_signed",
    "global_interval_reward",
    "terminal_bonus",
]

BASE_END_EPISODE_SIGNALS = [
    "run_id",
    "config_id",
    "seed",
    "end_terminated",
    "end_termination_reason_code",
    "end_termination_reason",
    "n_step",
    "n_interval",
    "final_t_sec",
    "final_cart_position_raw",
    "final_cart_velocity_raw",
    "final_pendulum_angle_raw",
    "final_pendulum_velocity_raw",
    "final_cart_position_norm",
    "final_cart_velocity_norm",
    "final_pendulum_angle_norm",
    "final_pendulum_velocity_norm",
    "total_reward",
    "total_learning_return",
    "learning_return_rate",
    "total_agent_decisions",
    "epsilon",
    "learning_rate",
]


def _append_unique(signal_names, signal_name):
    if signal_name not in signal_names:
        signal_names.append(signal_name)


def build_step_signal_names(config):
    signal_names = list(BASE_STEP_SIGNALS)
    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        for stem in PID_RAW_NORM_STEMS:
            _append_unique(signal_names, f"{stem}_{controller_id}_raw")
            _append_unique(signal_names, f"{stem}_{controller_id}_norm")
        if controller_config["calculation_mode"] == "incremental":
            for stem in PID_INCREMENTAL_RAW_NORM_STEMS:
                _append_unique(signal_names, f"{stem}_{controller_id}_raw")
                _append_unique(signal_names, f"{stem}_{controller_id}_norm")
        _append_unique(signal_names, f"kp_{controller_id}")
        _append_unique(signal_names, f"ki_{controller_id}")
        _append_unique(signal_names, f"kd_{controller_id}")

    for metric_config in config["metrics"]["interval_metrics"]:
        sample_transform_config = metric_config["sample_transform"]
        if "audit_target" in sample_transform_config:
            _append_unique(signal_names, sample_transform_config["audit_target"])
    return signal_names


def build_interval_signal_names(config):
    signal_names = list(BASE_INTERVAL_SIGNALS)

    for metric_config in config["metrics"]["interval_metrics"]:
        _append_unique(signal_names, metric_config["raw_target"])
        _append_unique(signal_names, metric_config["target"])

        interval_reduce_config = metric_config["interval_reduce"]
        if "audit_target" in interval_reduce_config:
            _append_unique(signal_names, interval_reduce_config["audit_target"])
        if "audit_stats" in interval_reduce_config:
            for stat_name in interval_reduce_config["audit_stats"]:
                _append_unique(signal_names, interval_reduce_config["audit_stats"][stat_name])

        post_transform_config = metric_config["post_transform"]
        if "audit_target" in post_transform_config:
            _append_unique(signal_names, post_transform_config["audit_target"])

    reward_config = config["reward"]
    for local_config in reward_config["local"]["controllers"]:
        controller_id = local_config["controller"]
        _append_unique(signal_names, f"local_cost_01_{controller_id}")
        _append_unique(signal_names, f"local_health_01_{controller_id}")
        _append_unique(signal_names, f"delta_health_{controller_id}_signed")
        _append_unique(signal_names, f"health_delta_signed_{controller_id}")
        _append_unique(signal_names, f"local_need_01_{controller_id}")
        _append_unique(signal_names, f"local_reward_signed_{controller_id}")
        _append_unique(signal_names, f"local_contribution_signed_{controller_id}")
        _append_unique(signal_names, f"cooperative_weight_effective_{controller_id}")
        _append_unique(signal_names, f"differential_utility_signed_{controller_id}")
        _append_unique(
            signal_names,
            f"differential_utility_normalized_signed_{controller_id}",
        )
        _append_unique(signal_names, f"cooperative_reward_signed_{controller_id}")
        _append_unique(signal_names, f"cooperative_contribution_signed_{controller_id}")
        _append_unique(signal_names, f"reward_controller_{controller_id}")
        _append_unique(signal_names, f"reward_pre_clip_{controller_id}")
        _append_unique(signal_names, f"reward_clip_excess_{controller_id}")
        _append_unique(signal_names, f"reward_clipped_{controller_id}")
        _append_unique(signal_names, f"terminal_bonus_{controller_id}")
        _append_unique(signal_names, f"terminal_inside_bands_{controller_id}")
        _append_unique(signal_names, f"terminal_band_duration_sec_{controller_id}")
        _append_unique(signal_names, f"terminal_band_qualified_{controller_id}")
        _append_unique(signal_names, f"terminal_bonus_eligible_{controller_id}")

    cooperative_config = reward_config["cooperative"]
    potential_config = cooperative_config["potential_cooperative"]
    if (
        cooperative_config["type"] == "potential_cooperative"
        and potential_config["reward_mode"] == "regime_residual"
    ):
        _append_unique(signal_names, potential_config["regime_residual_params"]["signal"])

    monotonic_config = cooperative_config["monotonic_cooperative"]
    for variable_config in monotonic_config["post_compose_penalty"]["variables"]:
        _append_unique(
            signal_names,
            "monotonic_post_compose_penalty_contribution_01_"
            f"{variable_config['id']}",
        )

    for gain_agent_config in config["agent"]["gain_agents"]:
        agent_id = gain_agent_config["id"]
        _append_unique(signal_names, f"reward_agent_{agent_id}")
        _append_unique(signal_names, f"reward_for_learning_{agent_id}")
        _append_unique(signal_names, f"action_{agent_id}")
        _append_unique(signal_names, f"is_exploratory_{agent_id}")
        _append_unique(signal_names, f"delta_gain_requested_{agent_id}")
        _append_unique(signal_names, f"delta_gain_applied_{agent_id}")
        _append_unique(signal_names, f"gain_value_{agent_id}")
        _append_unique(signal_names, f"td_error_{agent_id}")
        _append_unique(signal_names, f"q_value_{agent_id}")

    if "additional_state" in config["agent"]:
        for additional_state_config in config["agent"]["additional_state"]:
            _append_unique(signal_names, additional_state_config["source"])

    return signal_names


def build_end_episode_signal_names(config):
    signal_names = list(BASE_END_EPISODE_SIGNALS)
    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        _append_unique(signal_names, f"total_reward_{controller_id}")
        _append_unique(signal_names, f"total_learning_return_{controller_id}")
        _append_unique(signal_names, f"final_kp_{controller_id}")
        _append_unique(signal_names, f"final_ki_{controller_id}")
        _append_unique(signal_names, f"final_kd_{controller_id}")
    for signal_name in build_metric_report_signal_names(config):
        _append_unique(signal_names, signal_name)
    return signal_names


def build_runtime_arrays(step_capacity, interval_capacity, config, dtype=np.float64):
    if step_capacity <= 0:
        raise ValueError("step_capacity must be positive")
    if interval_capacity <= 0:
        raise ValueError("interval_capacity must be positive")

    step_signals = build_step_signal_names(config)
    interval_signals = build_interval_signal_names(config)
    end_episode_signals = build_end_episode_signal_names(config)
    metric_report_signals = build_metric_report_signal_names(config)

    step = SimpleNamespace()
    interval = SimpleNamespace()
    episode = SimpleNamespace()

    for signal_name in step_signals:
        setattr(step, signal_name, np.zeros(step_capacity, dtype=dtype))

    for signal_name in interval_signals:
        setattr(interval, signal_name, np.zeros(interval_capacity, dtype=dtype))

    for signal_name in end_episode_signals:
        setattr(episode, signal_name, 0.0)

    return SimpleNamespace(
        step=step,
        interval=interval,
        episode=episode,
        step_capacity=step_capacity,
        interval_capacity=interval_capacity,
        step_signal_names=list(step_signals),
        interval_signal_names=list(interval_signals),
        end_episode_signal_names=list(end_episode_signals),
        metric_report_signal_names=list(metric_report_signals),
    )


def reset_runtime_arrays(runtime):
    for signal_name in runtime.step_signal_names:
        getattr(runtime.step, signal_name).fill(0.0)

    for signal_name in runtime.interval_signal_names:
        getattr(runtime.interval, signal_name).fill(0.0)

    for signal_name in runtime.end_episode_signal_names:
        setattr(runtime.episode, signal_name, 0.0)


def build_output_refs(scope, signal_names, scope_name):
    output_refs = []
    for signal_name in signal_names:
        if not hasattr(scope, signal_name):
            raise KeyError(f"Unknown {scope_name} output signal: {signal_name}")
        output_refs.append(OutputRef(signal_name, getattr(scope, signal_name)))
    return output_refs
