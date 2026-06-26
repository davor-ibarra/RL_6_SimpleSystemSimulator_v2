from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


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
    "u_demand",
    "delta_u_demand",
    "u_alloc",
    "delta_u_alloc",
    "u_unallocated",
]

BASE_INTERVAL_SIGNALS = [
    "interval_id",
    "interval_start_step_idx",
    "interval_end_step_idx",
    "interval_t_start_sec",
    "interval_n_steps",
    "terminated",
    "termination_reason_code",
    "cooperative_cost_01",
    "cooperative_reward_signed",
    "reward_global_signed",
    "global_interval_reward",
]

BASE_END_EPISODE_SIGNALS = [
    "end_terminated",
    "end_termination_reason_code",
    "end_termination_reason",
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
        _append_unique(signal_names, f"kp_{controller_id}")
        _append_unique(signal_names, f"ki_{controller_id}")
        _append_unique(signal_names, f"kd_{controller_id}")
    return signal_names


def build_interval_signal_names(config):
    signal_names = list(BASE_INTERVAL_SIGNALS)

    for metric_config in config["metrics"]["interval_metrics"]:
        diagnostic_configs = []
        if "diagnostics" in metric_config:
            diagnostic_configs = metric_config["diagnostics"]
        for diagnostic_config in diagnostic_configs:
            _append_unique(signal_names, diagnostic_config["target"])
        _append_unique(signal_names, metric_config["target"])

    for local_config in config["reward"]["local"]["controllers"]:
        controller_id = local_config["controller"]
        _append_unique(signal_names, f"local_cost_01_{controller_id}")
        _append_unique(signal_names, f"local_reward_signed_{controller_id}")
        _append_unique(signal_names, f"reward_controller_{controller_id}")

    for gain_agent_config in config["agent"]["gain_agents"]:
        agent_id = gain_agent_config["id"]
        _append_unique(signal_names, f"reward_agent_{agent_id}")
        _append_unique(signal_names, f"reward_for_learning_{agent_id}")
        _append_unique(signal_names, f"action_{agent_id}")
        _append_unique(signal_names, f"delta_gain_requested_{agent_id}")
        _append_unique(signal_names, f"delta_gain_applied_{agent_id}")
        _append_unique(signal_names, f"gain_value_{agent_id}")
        _append_unique(signal_names, f"td_error_{agent_id}")
        _append_unique(signal_names, f"q_value_{agent_id}")

    return signal_names


def build_end_episode_signal_names(config):
    signal_names = list(BASE_END_EPISODE_SIGNALS)
    for controller_config in config["control"]["controllers"]:
        controller_id = controller_config["id"]
        _append_unique(signal_names, f"total_reward_{controller_id}")
        _append_unique(signal_names, f"final_kp_{controller_id}")
        _append_unique(signal_names, f"final_ki_{controller_id}")
        _append_unique(signal_names, f"final_kd_{controller_id}")
    return signal_names


def build_runtime_arrays(step_capacity, interval_capacity, config, dtype=np.float64):
    if step_capacity <= 0:
        raise ValueError("step_capacity must be positive")
    if interval_capacity <= 0:
        raise ValueError("interval_capacity must be positive")

    step_signals = build_step_signal_names(config)
    interval_signals = build_interval_signal_names(config)
    end_episode_signals = build_end_episode_signal_names(config)

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
