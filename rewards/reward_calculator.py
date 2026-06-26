from dataclasses import dataclass

import numpy as np


SUPPORTED_COOPERATIVE_MODES = ("norm-p",)


@dataclass(frozen=True)
class LocalRewardComponent:
    cost_array: object
    weight: float


@dataclass(frozen=True)
class LocalRewardTrack:
    controller_id: str
    components: tuple[LocalRewardComponent, ...]
    local_cost_array: object
    local_reward_array: object
    reward_controller_array: object


@dataclass(frozen=True)
class AgentRewardTrack:
    agent_id: str
    controller_id: str
    reward_agent_array: object
    reward_for_learning_array: object


def _clip_01(value):
    if not np.isfinite(value):
        raise ValueError("value must be finite before clipping to [0, 1]")
    return min(1.0, max(0.0, value))


def _clip_signed(value):
    if not np.isfinite(value):
        raise ValueError("value must be finite before clipping to [-1, 1]")
    return min(1.0, max(-1.0, value))


def _reward_from_cost(cost_01):
    return _clip_signed(1.0 - 2.0 * _clip_01(cost_01))


class RewardCalculator:
    def __init__(self, runtime, reward_config, agent_config, termination_config):
        self.interval = runtime.interval
        self.assignment = reward_config["assignment"]
        if reward_config["domain"] != "signed_unit":
            raise ValueError("Only signed_unit reward domain is supported")

        weights = reward_config["composition"]["weights"]
        self.local_weight = float(weights["local"])
        self.cooperative_weight = float(weights["cooperative"])
        self.strategic_weight = float(weights["strategic"])
        if (
            not np.isfinite(self.local_weight)
            or not np.isfinite(self.cooperative_weight)
            or not np.isfinite(self.strategic_weight)
        ):
            raise ValueError("reward composition weights must be finite")
        if self.local_weight < 0.0 or self.cooperative_weight < 0.0 or self.strategic_weight < 0.0:
            raise ValueError("reward composition weights must be non-negative")
        weight_sum = self.local_weight + self.cooperative_weight + self.strategic_weight
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("reward composition weights must sum to 1.0")

        self.strategic_enabled = bool(reward_config["strategic"]["enabled"])
        if self.strategic_enabled and self.strategic_weight <= 0.0:
            raise ValueError("strategic reward enabled with zero strategic weight")

        self.local_tracks = []
        for local_config in reward_config["local"]["controllers"]:
            controller_id = local_config["controller"]
            components = []
            component_weight_sum = 0.0
            for component_config in local_config["components"]:
                component_weight = float(component_config["weight"])
                if not np.isfinite(component_weight):
                    raise ValueError("local reward component weights must be finite")
                if component_weight < 0.0:
                    raise ValueError("local reward component weights must be non-negative")
                component_weight_sum += component_weight
                components.append(
                    LocalRewardComponent(
                        cost_array=getattr(self.interval, component_config["cost"]),
                        weight=component_weight,
                    )
                )
            if abs(component_weight_sum - 1.0) > 1e-9:
                raise ValueError("local reward component weights must sum to 1.0")
            self.local_tracks.append(
                LocalRewardTrack(
                    controller_id=controller_id,
                    components=tuple(components),
                    local_cost_array=getattr(self.interval, f"local_cost_01_{controller_id}"),
                    local_reward_array=getattr(self.interval, f"local_reward_signed_{controller_id}"),
                    reward_controller_array=getattr(self.interval, f"reward_controller_{controller_id}"),
                )
            )

        cooperative_config = reward_config["cooperative"]
        self.cooperative_enabled = bool(cooperative_config["enabled"])
        self.cooperative_mode = cooperative_config["mode"]
        if self.cooperative_mode not in SUPPORTED_COOPERATIVE_MODES:
            raise ValueError(f"Unsupported cooperative reward mode: {self.cooperative_mode}")
        self.cooperative_p = float(cooperative_config["p"])
        if not np.isfinite(self.cooperative_p):
            raise ValueError("cooperative p must be finite")
        if self.cooperative_p < 1.0:
            raise ValueError("cooperative p must be greater than or equal to 1.0")
        self.cooperative_weights = cooperative_config["variable_weights"]
        cooperative_sum = 0.0
        for local_track in self.local_tracks:
            cooperative_weight = float(self.cooperative_weights[local_track.controller_id])
            if not np.isfinite(cooperative_weight):
                raise ValueError("cooperative variable_weights must be finite")
            if cooperative_weight < 0.0:
                raise ValueError("cooperative variable_weights must be non-negative")
            cooperative_sum += cooperative_weight
        if abs(cooperative_sum - 1.0) > 1e-9:
            raise ValueError("cooperative variable_weights must sum to 1.0")

        self.agent_tracks = []
        for gain_agent_config in agent_config["gain_agents"]:
            agent_id = gain_agent_config["id"]
            self.agent_tracks.append(
                AgentRewardTrack(
                    agent_id=agent_id,
                    controller_id=gain_agent_config["controller"],
                    reward_agent_array=getattr(self.interval, f"reward_agent_{agent_id}"),
                    reward_for_learning_array=getattr(self.interval, f"reward_for_learning_{agent_id}"),
                )
            )

        if self.assignment == "controller_reward":
            self.assign_rewards = self.assign_controller_rewards
        elif self.assignment == "global_reward":
            self.assign_rewards = self.assign_global_reward
        elif self.assignment == "agent_reward":
            self.assign_rewards = self.assign_agent_reward
        else:
            raise ValueError(f"Unsupported reward assignment: {self.assignment}")

        self.strategic_reward_signed = 0.0

        self.terminal_bonus_config = reward_config["terminal_learning_bonus"]
        self.terminal_bonus_enabled = bool(self.terminal_bonus_config["enabled"])
        self.terminal_bonus_by_reason = self.terminal_bonus_config["bonuses"]
        self.reason_codes = termination_config["reason_codes"]
        self.reason_name_by_code = {}
        for reason_name, reason_code in self.reason_codes.items():
            self.reason_name_by_code[int(reason_code)] = reason_name
        self.terminal_bonus = 0

    def reset_episode(self):
        self.strategic_reward_signed = 0.0
        self.terminal_bonus = 0.0

    def evaluate_interval(self, interval_idx):
        self.terminal_learning_bonus(interval_idx)
        self.compute_local_rewards(interval_idx)
        self.compute_cooperative_reward(interval_idx)
        self.compute_strategic_reward(interval_idx)
        self.compose_controller_rewards(interval_idx)
        self.compose_global_reward(interval_idx)
        self.assign_rewards(interval_idx)

    def compute_local_rewards(self, interval_idx):
        for local_track in self.local_tracks:
            local_cost = 0.0
            for component in local_track.components:
                local_cost += component.weight * component.cost_array[interval_idx]
            local_cost = _clip_01(float(local_cost))
            local_track.local_cost_array[interval_idx] = local_cost
            local_track.local_reward_array[interval_idx] = _reward_from_cost(local_cost)

    def compute_cooperative_reward(self, interval_idx):
        if self.cooperative_enabled:
            cooperative_cost = self.compute_cooperative_cost(interval_idx)
            cooperative_cost = _clip_01(cooperative_cost)
        else:
            cooperative_cost = 0.0
        cooperative_reward = _reward_from_cost(cooperative_cost)
        self.interval.cooperative_cost_01[interval_idx] = cooperative_cost
        self.interval.cooperative_reward_signed[interval_idx] = cooperative_reward

    def compute_cooperative_cost(self, interval_idx):
        if self.cooperative_mode == "norm-p":
            return self.compute_norm_p_cooperative_cost(interval_idx)
        raise ValueError(f"Unsupported cooperative reward mode: {self.cooperative_mode}")

    def compute_norm_p_cooperative_cost(self, interval_idx):
        cooperative_cost_power = 0.0
        for local_track in self.local_tracks:
            local_cost = float(local_track.local_cost_array[interval_idx])
            cooperative_cost_power += (
                float(self.cooperative_weights[local_track.controller_id])
                * (local_cost ** self.cooperative_p)
            )
        return float(cooperative_cost_power ** (1.0 / self.cooperative_p))

    def compute_strategic_reward(self, interval_idx):
        del interval_idx
        if self.strategic_enabled:
            raise NotImplementedError("strategic reward is not implemented in this refactor")
        self.strategic_reward_signed = 0.0

    def compose_controller_rewards(self, interval_idx):
        cooperative_reward = self.interval.cooperative_reward_signed[interval_idx]
        for local_track in self.local_tracks:
            controller_reward = (
                self.local_weight * local_track.local_reward_array[interval_idx]
                + self.cooperative_weight * cooperative_reward
                + self.strategic_weight * self.strategic_reward_signed
            )
            local_track.reward_controller_array[interval_idx] = _clip_signed(float(controller_reward))

    def compose_global_reward(self, interval_idx):
        global_reward = 0.0
        for local_track in self.local_tracks:
            global_reward += local_track.reward_controller_array[interval_idx]
        global_reward /= len(self.local_tracks)
        global_reward = _clip_signed(float(global_reward))
        self.interval.reward_global_signed[interval_idx] = global_reward
        self.interval.global_interval_reward[interval_idx] = global_reward

    def assign_controller_rewards(self, interval_idx):
        for agent_track in self.agent_tracks:
            reward = self.find_controller_reward(agent_track.controller_id, interval_idx)
            agent_track.reward_agent_array[interval_idx] = reward + self.terminal_bonus
            agent_track.reward_for_learning_array[interval_idx] = reward + self.terminal_bonus

    def find_controller_reward(self, controller_id, interval_idx):
        for local_track in self.local_tracks:
            if local_track.controller_id == controller_id:
                return local_track.reward_controller_array[interval_idx]
        raise ValueError(f"Unknown controller for reward assignment: {controller_id}")

    def assign_global_reward(self, interval_idx):
        reward = self.interval.reward_global_signed[interval_idx]
        for agent_track in self.agent_tracks:
            agent_track.reward_agent_array[interval_idx] = reward + self.terminal_bonus
            agent_track.reward_for_learning_array[interval_idx] = reward + self.terminal_bonus

    def assign_agent_reward(self, interval_idx):
        del interval_idx
        raise NotImplementedError("agent_reward assignment requires explicit fine credit implementation")
    
    def terminal_learning_bonus(self, interval_idx):
        if not self.terminal_bonus_enabled:
            return 0.0

        reason_code = int(self.interval.termination_reason_code[interval_idx])
        reason_name = self.reason_name_by_code[reason_code]

        self.terminal_bonus = float(self.terminal_bonus_by_reason[reason_name])
