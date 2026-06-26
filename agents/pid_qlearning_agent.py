from dataclasses import dataclass

import numpy as np


@dataclass
class GainAgent:
    agent_id: str
    controller_id: str
    gain_name: str
    min_gain: float
    max_gain: float
    state_step: float
    initial_gain: float
    action_array: np.ndarray
    reward_array: np.ndarray
    gain_value_array: np.ndarray
    td_error_array: np.ndarray
    q_value_array: np.ndarray
    q_table: np.ndarray
    visits: np.ndarray
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


class PIDQLearningAgent:
    def __init__(self, runtime, agent_config, control_config, seed=12345):
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
            n_states = _state_count(min_gain, max_gain, state_step)
            initial_state_idx = _gain_to_state_idx(initial_gain, min_gain, max_gain, state_step)
            self.tracks.append(
                GainAgent(
                    agent_id=agent_id,
                    controller_id=controller_id,
                    gain_name=gain_name,
                    min_gain=min_gain,
                    max_gain=max_gain,
                    state_step=state_step,
                    initial_gain=initial_gain,
                    action_array=getattr(self.interval, f"action_{agent_id}"),
                    reward_array=getattr(self.interval, f"reward_for_learning_{agent_id}"),
                    gain_value_array=getattr(self.interval, f"gain_value_{agent_id}"),
                    td_error_array=getattr(self.interval, f"td_error_{agent_id}"),
                    q_value_array=getattr(self.interval, f"q_value_{agent_id}"),
                    q_table=np.zeros((n_states, len(self.action_moves)), dtype=np.float64),
                    visits=np.zeros((n_states, len(self.action_moves)), dtype=np.int64),
                    current_state_idx=initial_state_idx,
                    last_state_idx=initial_state_idx,
                    last_action_idx=self.maintain_action_idx
                )
            )

        self.rng = np.random.default_rng(seed)

    def reset_episode(self):
        for track in self.tracks:
            track.current_state_idx = _gain_to_state_idx(
                track.initial_gain,
                track.min_gain,
                track.max_gain,
                track.state_step,
            )
            track.last_state_idx = track.current_state_idx
            track.last_action_idx = self.maintain_action_idx

    def random_action_idx(self):
        return int(self.rng.integers(0, len(self.action_moves)))

    def select_greedy_action_idx(self, q_values):
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.rng.choice(best_actions))

    def select_actions(self, interval_idx):
        for track in self.tracks:
            if self.rng.random() < self.epsilon:
                action_idx = self.random_action_idx()
            else:
                q_values = track.q_table[track.current_state_idx]
                action_idx = self.select_greedy_action_idx(q_values)
            track.last_state_idx = track.current_state_idx
            track.last_action_idx = action_idx
            track.action_array[interval_idx] = self.action_moves[action_idx]

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

            next_state_idx = _gain_to_state_idx(applied_gain, track.min_gain, track.max_gain, track.state_step)

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

    def decay_learning_parameters(self):
        if self.epsilon_decay_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_factor)
        if self.learning_rate_decay_enabled:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay_factor)

    def export_state(self):
        agents = []
        for track in self.tracks:
            gain_values = [
                track.min_gain + state_idx * track.state_step for state_idx in range(track.q_table.shape[0])
            ]
            agents.append(
                {
                    "agent_id": track.agent_id,
                    "controller_id": track.controller_id,
                    "gain_name": track.gain_name,
                    "min_gain": track.min_gain,
                    "max_gain": track.max_gain,
                    "state_step": track.state_step,
                    "current_state_idx": int(track.current_state_idx),
                    "gain_values": gain_values,
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
