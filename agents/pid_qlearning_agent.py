from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdditionalStateSpec:
    signal_id: str
    source_name: str
    source_array: np.ndarray
    min_value: float
    max_value: float
    uses_thresholds: bool
    thresholds: tuple
    step: float
    state_count: int


@dataclass
class GainAgent:
    agent_id: str
    controller_id: str
    gain_name: str
    min_gain: float
    max_gain: float
    state_step: float
    gain_state_count: int
    initial_gain: float
    additional_state_specs: tuple
    additional_state_counts: tuple
    action_array: np.ndarray
    is_exploratory_array: np.ndarray
    reward_array: np.ndarray
    gain_value_array: np.ndarray
    td_error_array: np.ndarray
    q_value_array: np.ndarray
    q_table: np.ndarray
    visits: np.ndarray
    episode_initial_q_table: np.ndarray
    episode_visited_state_indices: set
    current_state_idx: int
    last_state_idx: int
    last_action_idx: int


def _state_count(min_gain, max_gain, state_step):
    if max_gain <= min_gain:
        raise ValueError("gain max must be greater than gain min")
    if state_step <= 0.0:
        raise ValueError("state_step must be positive")

    n_steps_float = (max_gain - min_gain) / state_step
    n_steps = int(round(n_steps_float))
    if abs(n_steps_float - n_steps) > 1e-9:
        raise ValueError("gain range must be exactly divisible by state_step")

    return n_steps + 1


def _product(values):
    product = 1
    for value in values:
        product *= int(value)
    return product


def _gain_to_state_idx(gain_value, min_gain, max_gain, state_step):
    if not np.isfinite(gain_value):
        raise ValueError("gain value must be finite")
    if not np.isfinite(min_gain):
        raise ValueError("min_gain must be finite")
    if not np.isfinite(max_gain):
        raise ValueError("max_gain must be finite")
    if not np.isfinite(state_step):
        raise ValueError("state_step must be finite")

    if gain_value < min_gain - 1e-9 or gain_value > max_gain + 1e-9:
        raise ValueError("gain value must be inside gain state range")

    state_idx_float = (gain_value - min_gain) / state_step
    state_idx = int(round(state_idx_float))
    if abs(state_idx_float - state_idx) > 1e-9:
        raise ValueError("gain value must lie exactly on the gain state grid")

    return state_idx


def _validate_additional_state_range(min_value, max_value):
    if not np.isfinite(min_value):
        raise ValueError("additional_state min must be finite")
    if not np.isfinite(max_value):
        raise ValueError("additional_state max must be finite")
    if max_value <= min_value:
        raise ValueError("additional_state max must be greater than min")


def _build_additional_state_spec(interval, additional_state_config):
    signal_id = additional_state_config["id"]
    source_name = additional_state_config["source"]
    min_value = float(additional_state_config["min"])
    max_value = float(additional_state_config["max"])
    _validate_additional_state_range(min_value, max_value)

    has_thresholds = "thresholds" in additional_state_config
    has_step = "step" in additional_state_config
    if has_thresholds == has_step:
        raise ValueError("additional_state must define exactly one discretization rule: thresholds or step")

    if has_thresholds:
        thresholds = tuple(float(value) for value in additional_state_config["thresholds"])
        if len(thresholds) == 0:
            raise ValueError("additional_state thresholds must not be empty")
        previous_threshold = min_value
        for threshold in thresholds:
            if not np.isfinite(threshold):
                raise ValueError("additional_state thresholds must be finite")
            if threshold <= previous_threshold:
                raise ValueError("additional_state thresholds must be strictly increasing")
            if threshold >= max_value:
                raise ValueError("additional_state thresholds must be lower than max")
            previous_threshold = threshold
        return AdditionalStateSpec(
            signal_id=signal_id,
            source_name=source_name,
            source_array=getattr(interval, source_name),
            min_value=min_value,
            max_value=max_value,
            uses_thresholds=True,
            thresholds=thresholds,
            step=0.0,
            state_count=len(thresholds) + 1,
        )

    step = float(additional_state_config["step"])
    state_count = _state_count(min_value, max_value, step)
    return AdditionalStateSpec(
        signal_id=signal_id,
        source_name=source_name,
        source_array=getattr(interval, source_name),
        min_value=min_value,
        max_value=max_value,
        uses_thresholds=False,
        thresholds=(),
        step=step,
        state_count=state_count,
    )


def _additional_state_to_idx(spec, value):
    if not np.isfinite(value):
        raise ValueError(f"additional_state source {spec.source_name} must be finite")
    if value < spec.min_value - 1e-9 or value > spec.max_value + 1e-9:
        raise ValueError(f"additional_state source {spec.source_name} must be inside its configured range")

    if spec.uses_thresholds:
        for threshold_idx, threshold in enumerate(spec.thresholds):
            if value < threshold:
                return threshold_idx
        return len(spec.thresholds)

    return _gain_to_state_idx(value, spec.min_value, spec.max_value, spec.step)


def _additional_state_indices(additional_state_specs, interval_idx):
    indices = []
    for spec in additional_state_specs:
        indices.append(_additional_state_to_idx(spec, float(spec.source_array[interval_idx])))
    return tuple(indices)


def _initial_additional_state_indices(additional_state_specs):
    indices = []
    for spec in additional_state_specs:
        indices.append(_additional_state_to_idx(spec, spec.min_value))
    return tuple(indices)


def _flatten_state_idx(gain_state_idx, additional_state_indices, gain_state_count, additional_state_counts):
    state_idx = int(gain_state_idx)
    multiplier = int(gain_state_count)
    for additional_state_idx, state_count in zip(additional_state_indices, additional_state_counts):
        state_idx += int(additional_state_idx) * multiplier
        multiplier *= int(state_count)
    return state_idx


def _unflatten_state_idx(state_idx, gain_state_count, additional_state_counts):
    gain_state_idx = int(state_idx) % int(gain_state_count)
    remainder = int(state_idx) // int(gain_state_count)
    additional_state_indices = []
    for state_count in additional_state_counts:
        additional_state_indices.append(remainder % int(state_count))
        remainder //= int(state_count)
    return gain_state_idx, tuple(additional_state_indices)


class PIDQLearningAgent:
    def __init__(self, runtime, agent_config, control_config, seed):
        self.interval = runtime.interval
        self.params = agent_config["params"]
        self.algorithm = self.params["algorithm"]
        if self.algorithm != "q_learning":
            raise ValueError("Only q_learning algorithm is supported")

        self.discount_factor = float(self.params["discount_factor"])
        self.epsilon = float(self.params["epsilon"])
        self.epsilon_decay_enabled = bool(self.params["epsilon_decay"]["enabled"])
        self.epsilon_decay_factor = float(self.params["epsilon_decay"]["decay_factor"])
        self.epsilon_min = float(self.params["epsilon_decay"]["epsilon_min"])
        self.learning_rate = float(self.params["learning_rate"])
        self.learning_rate_decay_enabled = bool(self.params["learning_rate_decay"]["enabled"])
        self.learning_rate_decay_factor = float(self.params["learning_rate_decay"]["decay_factor"])
        self.learning_rate_min = float(self.params["learning_rate_decay"]["learning_rate_min"])

        self.action_moves = [int(action_config["move"]) for action_config in agent_config["actions"]]
        if self.action_moves != [-1, 0, 1]:
            raise ValueError("PIDQLearningAgent expects actions with moves [-1, 0, 1]")
        self.maintain_action_idx = self.action_moves.index(0)

        self.additional_state_specs = []
        if "additional_state" in agent_config:
            for additional_state_config in agent_config["additional_state"]:
                self.additional_state_specs.append(
                    _build_additional_state_spec(self.interval, additional_state_config)
                )
        self.additional_state_specs = tuple(self.additional_state_specs)
        
        control_by_id = {}
        for controller_config in control_config["controllers"]:
            control_by_id[controller_config["id"]] = controller_config

        self.tracks = []
        for gain_agent_config in agent_config["gain_agents"]:
            agent_id = gain_agent_config["id"]
            controller_id = gain_agent_config["controller"]
            gain_name = gain_agent_config["gain"]
            controller_config = control_by_id[controller_id]
            gain_limits = controller_config["gain_update"][gain_name]
            initial_gain = float(controller_config["initial_gains"][gain_name])
            min_gain = float(gain_limits["min"])
            max_gain = float(gain_limits["max"])
            state_step = float(gain_limits["delta_gain"])
            gain_state_count = _state_count(min_gain, max_gain, state_step)
            initial_gain_state_idx = _gain_to_state_idx(initial_gain, min_gain, max_gain, state_step)
            additional_state_specs = ()
            if "additional_state" in gain_agent_config:
                if bool(gain_agent_config["additional_state"]):
                    if len(self.additional_state_specs) == 0:
                        raise ValueError("gain agent requested additional_state but none are declared")
                    additional_state_specs = self.additional_state_specs
            additional_state_counts = tuple(spec.state_count for spec in additional_state_specs)
            n_states = gain_state_count * _product(additional_state_counts)
            initial_state_idx = _flatten_state_idx(
                initial_gain_state_idx,
                _initial_additional_state_indices(additional_state_specs),
                gain_state_count,
                additional_state_counts,
            )
            q_table = np.zeros((n_states, len(self.action_moves)), dtype=np.float64)
            self.tracks.append(
                GainAgent(
                    agent_id=agent_id,
                    controller_id=controller_id,
                    gain_name=gain_name,
                    min_gain=min_gain,
                    max_gain=max_gain,
                    state_step=state_step,
                    gain_state_count=gain_state_count,
                    initial_gain=initial_gain,
                    additional_state_specs=additional_state_specs,
                    additional_state_counts=additional_state_counts,
                    action_array=getattr(self.interval, f"action_{agent_id}"),
                    is_exploratory_array=getattr(self.interval, f"is_exploratory_{agent_id}"),
                    reward_array=getattr(self.interval, f"reward_for_learning_{agent_id}"),
                    gain_value_array=getattr(self.interval, f"gain_value_{agent_id}"),
                    td_error_array=getattr(self.interval, f"td_error_{agent_id}"),
                    q_value_array=getattr(self.interval, f"q_value_{agent_id}"),
                    q_table=q_table,
                    visits=np.zeros((n_states, len(self.action_moves)), dtype=np.int64),
                    episode_initial_q_table=q_table.copy(),
                    episode_visited_state_indices={initial_state_idx},
                    current_state_idx=initial_state_idx,
                    last_state_idx=initial_state_idx,
                    last_action_idx=self.maintain_action_idx
                )
            )

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset_episode(self):
        for track in self.tracks:
            initial_gain_state_idx = _gain_to_state_idx(
                track.initial_gain,
                track.min_gain,
                track.max_gain,
                track.state_step,
            )
            track.current_state_idx = _flatten_state_idx(
                initial_gain_state_idx,
                _initial_additional_state_indices(track.additional_state_specs),
                track.gain_state_count,
                track.additional_state_counts,
            )
            track.last_state_idx = track.current_state_idx
            track.last_action_idx = self.maintain_action_idx
            track.episode_initial_q_table = track.q_table.copy()
            track.episode_visited_state_indices = {track.current_state_idx}

    def random_action_idx(self):
        return int(self.rng.integers(0, len(self.action_moves)))

    def select_greedy_action_idx(self, q_values):
        best_actions = self.greedy_action_indices(q_values)
        return int(self.rng.choice(best_actions))

    def greedy_action_indices(self, q_values):
        max_q = np.max(q_values)
        return np.flatnonzero(np.isclose(q_values, max_q))

    def select_actions(self, interval_idx):
        for track in self.tracks:
            track.episode_visited_state_indices.add(track.current_state_idx)
            if self.rng.random() < self.epsilon:
                action_idx = self.random_action_idx()
                is_exploratory = 1.0
            else:
                q_values = track.q_table[track.current_state_idx]
                action_idx = self.select_greedy_action_idx(q_values)
                is_exploratory = 0.0
            track.last_state_idx = track.current_state_idx
            track.last_action_idx = action_idx
            track.action_array[interval_idx] = self.action_moves[action_idx]
            track.is_exploratory_array[interval_idx] = is_exploratory

    def learn(self, interval_idx, terminal_for_learning):
        for track in self.tracks:
            state_idx = track.last_state_idx
            action_idx = track.last_action_idx

            reward = float(track.reward_array[interval_idx])
            if not np.isfinite(reward):
                raise ValueError(f"Non-finite reward_for_learning for agent {track.agent_id}")

            applied_gain = float(track.gain_value_array[interval_idx])
            if not np.isfinite(applied_gain):
                raise ValueError(f"Non-finite gain_value for agent {track.agent_id}")

            next_gain_state_idx = _gain_to_state_idx(applied_gain, track.min_gain, track.max_gain, track.state_step)
            next_state_idx = _flatten_state_idx(
                next_gain_state_idx,
                _additional_state_indices(track.additional_state_specs, interval_idx),
                track.gain_state_count,
                track.additional_state_counts,
            )

            current_q = track.q_table[state_idx, action_idx]

            if terminal_for_learning:
                td_target = reward
            else:
                next_q = float(np.max(track.q_table[next_state_idx]))
                td_target = reward + self.discount_factor * next_q

            td_error = td_target - current_q
            updated_q = current_q + self.learning_rate * td_error

            track.q_table[state_idx, action_idx] = updated_q
            track.visits[state_idx, action_idx] += 1
            track.td_error_array[interval_idx] = td_error
            track.q_value_array[interval_idx] = updated_q
            track.current_state_idx = next_state_idx
            track.episode_visited_state_indices.add(next_state_idx)

    def decay_learning_parameters(self):
        if self.epsilon_decay_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_factor)
        if self.learning_rate_decay_enabled:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay_factor)

    def episode_learning_metrics(self):
        metrics = []
        for track in self.tracks:
            q_delta = track.q_table - track.episode_initial_q_table
            q_delta_rms = float(np.sqrt(np.mean(q_delta**2)))
            stable_state_count = 0
            visited_state_indices = sorted(track.episode_visited_state_indices)
            for state_idx in visited_state_indices:
                initial_greedy_actions = tuple(
                    self.greedy_action_indices(track.episode_initial_q_table[state_idx]).tolist()
                )
                final_greedy_actions = tuple(self.greedy_action_indices(track.q_table[state_idx]).tolist())
                if initial_greedy_actions == final_greedy_actions:
                    stable_state_count += 1
            metrics.append(
                {
                    "agent_id": track.agent_id,
                    "q_delta_rms": q_delta_rms,
                    "q_value_mean": float(np.mean(track.q_table)),
                    "q_value_std": float(np.std(track.q_table)),
                    "q_value_min": float(np.min(track.q_table)),
                    "q_value_max": float(np.max(track.q_table)),
                    "policy_stability": float(stable_state_count / len(visited_state_indices)),
                    "visited_state_count": len(visited_state_indices),
                    "visited_state_frequency": float(
                        len(visited_state_indices) / track.q_table.shape[0]
                    ),
                }
            )
        return metrics

    def export_state(self):
        agents = []
        for track in self.tracks:
            gain_values = [
                track.min_gain + state_idx * track.state_step for state_idx in range(track.gain_state_count)
            ]
            additional_states = []
            for spec in track.additional_state_specs:
                additional_state = {
                    "id": spec.signal_id,
                    "source": spec.source_name,
                    "min": spec.min_value,
                    "max": spec.max_value,
                    "state_count": spec.state_count,
                }
                if spec.uses_thresholds:
                    additional_state["thresholds"] = list(spec.thresholds)
                else:
                    additional_state["step"] = spec.step
                additional_states.append(additional_state)

            states = []
            for state_idx in range(track.q_table.shape[0]):
                gain_state_idx, additional_state_indices = _unflatten_state_idx(
                    state_idx,
                    track.gain_state_count,
                    track.additional_state_counts,
                )
                state_row = {
                    "state_index": int(state_idx),
                    "gain_state_index": int(gain_state_idx),
                    "gain_value": track.min_gain + gain_state_idx * track.state_step,
                }
                for spec, additional_state_idx in zip(track.additional_state_specs, additional_state_indices):
                    state_row[f"{spec.signal_id}_state_index"] = int(additional_state_idx)
                states.append(state_row)

            agents.append(
                {
                    "agent_id": track.agent_id,
                    "controller_id": track.controller_id,
                    "gain_name": track.gain_name,
                    "min_gain": track.min_gain,
                    "max_gain": track.max_gain,
                    "state_step": track.state_step,
                    "gain_state_count": track.gain_state_count,
                    "current_state_idx": int(track.current_state_idx),
                    "gain_values": gain_values,
                    "additional_states": additional_states,
                    "states": states,
                    "q_table": track.q_table.tolist(),
                    "visits": track.visits.tolist(),
                }
            )
        return {
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "action_moves": list(self.action_moves),
            "agents": agents,
        }
