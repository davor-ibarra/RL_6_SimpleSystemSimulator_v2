from dataclasses import dataclass

import numpy as np


SUPPORTED_COOPERATIVE_MODES = ("norm-p", "geometric_consensus")
SUPPORTED_COOPERATIVE_REWARD_MODES = ("absolute", "progress", "progress_residual", "regime_residual")
SUPPORTED_REGIME_REWARD_MODES = ("absolute", "progress", "progress_residual")


@dataclass(frozen=True)
class LocalRewardComponent:
    cost_array: object
    weight: float


@dataclass(frozen=True)
class LocalRewardTrack:
    controller_id: str
    components: tuple[LocalRewardComponent, ...]
    local_cost_array: object
    local_health_array: object
    local_need_array: object
    local_reward_array: object
    cooperative_weight_effective_array: object
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
                    local_health_array=getattr(self.interval, f"local_health_01_{controller_id}"),
                    local_need_array=getattr(self.interval, f"local_need_01_{controller_id}"),
                    local_reward_array=getattr(self.interval, f"local_reward_signed_{controller_id}"),
                    cooperative_weight_effective_array=getattr(
                        self.interval,
                        f"cooperative_weight_effective_{controller_id}",
                    ),
                    reward_controller_array=getattr(self.interval, f"reward_controller_{controller_id}"),
                )
            )

        cooperative_config = reward_config["cooperative"]
        self.cooperative_enabled = bool(cooperative_config["enabled"])
        self.cooperative_mode = cooperative_config["mode"]
        if self.cooperative_mode not in SUPPORTED_COOPERATIVE_MODES:
            raise ValueError(f"Unsupported cooperative potential mode: {self.cooperative_mode}")
        self.cooperative_reward_mode = cooperative_config["reward_mode"]
        if self.cooperative_reward_mode not in SUPPORTED_COOPERATIVE_REWARD_MODES:
            raise ValueError(f"Unsupported cooperative reward mode: {self.cooperative_reward_mode}")
        if not self.cooperative_enabled and self.cooperative_weight != 0.0:
            raise ValueError("cooperative reward disabled with non-zero cooperative composition weight")

        self.cooperative_p = float(cooperative_config["p"])
        self.epsilon_geom_conc = float(cooperative_config["epsilon_geom_conc"])
        self.residual_weight = float(cooperative_config["residual_weight"])
        if not np.isfinite(self.cooperative_p):
            raise ValueError("cooperative p must be finite")
        if not np.isfinite(self.epsilon_geom_conc):
            raise ValueError("cooperative epsilon_geom_conc must be finite")
        if not np.isfinite(self.residual_weight):
            raise ValueError("cooperative residual_weight must be finite")
        if self.cooperative_mode == "norm-p" and self.cooperative_p < 1.0:
            raise ValueError("cooperative p must be greater than or equal to 1.0")
        if self.cooperative_mode == "geometric_consensus" and not 0.0 < self.epsilon_geom_conc < 1.0:
            raise ValueError("cooperative epsilon_geom_conc must be greater than 0 and less than 1")
        if self.residual_weight < 0.0:
            raise ValueError("cooperative residual_weight must be non-negative")

        adaptive_weights_config = cooperative_config["adaptive_weights"]
        self.adaptive_weights_enabled = bool(adaptive_weights_config["enabled"])
        self.weight_floor = float(adaptive_weights_config["weight_floor"])
        self.deterioration_weight = float(adaptive_weights_config["deterioration_weight"])
        self.smoothing_alpha = float(adaptive_weights_config["smoothing_alpha"])
        if not np.isfinite(self.weight_floor):
            raise ValueError("adaptive_weights weight_floor must be finite")
        if not np.isfinite(self.deterioration_weight):
            raise ValueError("adaptive_weights deterioration_weight must be finite")
        if not np.isfinite(self.smoothing_alpha):
            raise ValueError("adaptive_weights smoothing_alpha must be finite")
        if self.weight_floor < 0.0 or self.weight_floor > 1.0 / len(self.local_tracks):
            raise ValueError("adaptive_weights weight_floor must be between 0 and 1/N")
        if self.deterioration_weight < 0.0:
            raise ValueError("adaptive_weights deterioration_weight must be non-negative")
        if self.smoothing_alpha < 0.0 or self.smoothing_alpha > 1.0:
            raise ValueError("adaptive_weights smoothing_alpha must be between 0 and 1")

        synchrony_config = cooperative_config["synchrony"]
        self.synchrony_enabled = bool(synchrony_config["enabled"])
        self.synchrony_strength = float(synchrony_config["strength"])
        if not np.isfinite(self.synchrony_strength):
            raise ValueError("synchrony strength must be finite")
        if self.synchrony_strength < 0.0:
            raise ValueError("synchrony strength must be non-negative")

        adaptive_residual_config = cooperative_config["adaptive_residual"]
        self.adaptive_residual_enabled = bool(adaptive_residual_config["enabled"])
        self.residual_weight_min = float(adaptive_residual_config["residual_weight_min"])
        self.residual_weight_max = float(adaptive_residual_config["residual_weight_max"])
        self.residual_weight_gamma = float(adaptive_residual_config["gamma"])
        if not np.isfinite(self.residual_weight_min):
            raise ValueError("adaptive_residual residual_weight_min must be finite")
        if not np.isfinite(self.residual_weight_max):
            raise ValueError("adaptive_residual residual_weight_max must be finite")
        if not np.isfinite(self.residual_weight_gamma):
            raise ValueError("adaptive_residual gamma must be finite")
        if self.residual_weight_min < 0.0 or self.residual_weight_max < 0.0:
            raise ValueError("adaptive_residual residual weights must be non-negative")
        if self.residual_weight_max < self.residual_weight_min:
            raise ValueError("adaptive_residual residual_weight_max must be greater than or equal to min")
        if self.residual_weight_gamma < 1.0:
            raise ValueError("adaptive_residual gamma must be greater than or equal to 1")

        self.regime_signal_name = ""
        self.regime_thresholds = ()
        self.reward_mode_by_regime = ()
        self.residual_weight_by_regime = ()
        self.regime_array = None
        if self.cooperative_reward_mode == "regime_residual":
            regime_config = cooperative_config["regime_residual_params"]
            self.regime_signal_name = regime_config["signal"]
            self.regime_array = getattr(self.interval, self.regime_signal_name)
            self.regime_thresholds = tuple(float(value) for value in regime_config["thresholds"])
            previous_threshold = 0.0
            for threshold in self.regime_thresholds:
                if not np.isfinite(threshold):
                    raise ValueError("regime_residual thresholds must be finite")
                if threshold <= previous_threshold:
                    raise ValueError("regime_residual thresholds must be strictly increasing")
                if threshold >= 1.0:
                    raise ValueError("regime_residual thresholds must be lower than 1")
                previous_threshold = threshold
            n_regimes = len(self.regime_thresholds) + 1

            reward_mode_by_regime = regime_config["reward_mode_by_regime"]
            reward_modes = []
            for regime_idx in range(n_regimes):
                regime_reward_mode = reward_mode_by_regime[str(regime_idx)]
                if regime_reward_mode not in SUPPORTED_REGIME_REWARD_MODES:
                    raise ValueError(f"Unsupported regime reward mode: {regime_reward_mode}")
                reward_modes.append(regime_reward_mode)
            if len(reward_mode_by_regime) != n_regimes:
                raise ValueError("regime_residual reward_mode_by_regime must match the number of regimes")
            self.reward_mode_by_regime = tuple(reward_modes)

            residual_weight_by_regime = regime_config["residual_weight_by_regime"]
            weights = []
            for regime_idx in range(n_regimes):
                weight = float(residual_weight_by_regime[str(regime_idx)])
                if not np.isfinite(weight):
                    raise ValueError("regime_residual residual weights must be finite")
                if weight < 0.0:
                    raise ValueError("regime_residual residual weights must be non-negative")
                weights.append(weight)
            if len(residual_weight_by_regime) != n_regimes:
                raise ValueError("regime_residual residual_weight_by_regime must match the number of regimes")
            self.residual_weight_by_regime = tuple(weights)
        configured_cooperative_weights = cooperative_config["variable_weights"]
        if len(configured_cooperative_weights) != len(self.local_tracks):
            raise ValueError("cooperative variable_weights must match local controllers exactly")
        for cooperative_controller_id in configured_cooperative_weights:
            matched_local_controller = False
            for local_track in self.local_tracks:
                if cooperative_controller_id == local_track.controller_id:
                    matched_local_controller = True
            if not matched_local_controller:
                raise ValueError(f"Unknown cooperative variable_weight controller: {cooperative_controller_id}")

        self.cooperative_weights = {}
        cooperative_sum = 0.0
        for local_track in self.local_tracks:
            if local_track.controller_id not in configured_cooperative_weights:
                raise ValueError(f"Missing cooperative variable_weight controller: {local_track.controller_id}")
            cooperative_weight = float(configured_cooperative_weights[local_track.controller_id])
            if not np.isfinite(cooperative_weight):
                raise ValueError("cooperative variable_weights must be finite")
            if cooperative_weight < 0.0:
                raise ValueError("cooperative variable_weights must be non-negative")
            cooperative_sum += cooperative_weight
            self.cooperative_weights[local_track.controller_id] = cooperative_weight
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
        self.previous_cooperative_potential = 0.0
        self.has_previous_cooperative_potential = False
        self.previous_local_costs = {}
        self.previous_local_needs = {}
        for local_track in self.local_tracks:
            self.previous_local_costs[local_track.controller_id] = 0.0
            self.previous_local_needs[local_track.controller_id] = 0.0

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
        self.previous_cooperative_potential = 0.0
        self.has_previous_cooperative_potential = False
        for local_track in self.local_tracks:
            self.previous_local_costs[local_track.controller_id] = 0.0
            self.previous_local_needs[local_track.controller_id] = 0.0

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
            local_track.local_health_array[interval_idx] = 1.0 - local_cost
            local_track.local_reward_array[interval_idx] = _reward_from_cost(local_cost)

    def compute_cooperative_reward(self, interval_idx):
        cooperative_regime = 0
        effective_weights = self.compute_effective_cooperative_weights()
        self.write_effective_cooperative_weights(interval_idx, effective_weights)
        if self.cooperative_enabled:
            cooperative_potential_base = self.compute_cooperative_potential_base(interval_idx, effective_weights)
            (
                synchrony_health_balance,
                synchrony_factor,
                cooperative_potential,
            ) = self.compute_collective_potential(interval_idx, cooperative_potential_base, effective_weights)
            cooperative_cost = _clip_01(1.0 - cooperative_potential)
            delta_cooperative_potential = self.compute_delta_cooperative_potential(cooperative_potential)
            cooperative_regime = self.compute_cooperative_regime(cooperative_potential)
            residual_weight_effective = self.compute_residual_weight_effective(
                cooperative_potential,
                cooperative_regime,
            )
            cooperative_progress_residual = self.compute_cooperative_progress_residual(
                cooperative_potential,
                delta_cooperative_potential,
                residual_weight_effective,
            )
            cooperative_reward = self.transform_cooperative_reward(
                cooperative_potential,
                delta_cooperative_potential,
                cooperative_regime,
                residual_weight_effective,
            )
        else:
            cooperative_potential_base = 1.0
            synchrony_health_balance = 1.0
            synchrony_factor = 1.0
            cooperative_potential = 1.0
            cooperative_cost = 0.0
            delta_cooperative_potential = 0.0
            cooperative_regime = self.compute_cooperative_regime(cooperative_potential)
            residual_weight_effective = self.compute_residual_weight_effective(
                cooperative_potential,
                cooperative_regime,
            )
            cooperative_progress_residual = 0.0
            cooperative_reward = 0.0
        self.interval.cooperative_cost_01[interval_idx] = cooperative_cost
        self.interval.cooperative_potential_base_01[interval_idx] = cooperative_potential_base
        self.interval.synchrony_health_balance_01[interval_idx] = synchrony_health_balance
        self.interval.synchrony_factor_01[interval_idx] = synchrony_factor
        self.interval.collective_potential_01[interval_idx] = cooperative_potential
        self.interval.cooperative_potential_01[interval_idx] = cooperative_potential
        self.interval.delta_cooperative_potential_signed[interval_idx] = delta_cooperative_potential
        self.interval.cooperative_progress_residual_signed[interval_idx] = cooperative_progress_residual
        self.interval.residual_weight_effective[interval_idx] = residual_weight_effective
        self.interval.cooperative_reward_signed[interval_idx] = cooperative_reward
        if self.cooperative_reward_mode == "regime_residual":
            self.regime_array[interval_idx] = cooperative_regime
        self.update_local_needs(interval_idx)

    def compute_cooperative_cost(self, interval_idx):
        cooperative_potential = self.compute_cooperative_potential(interval_idx)
        return _clip_01(1.0 - cooperative_potential)

    def compute_cooperative_potential(self, interval_idx):
        effective_weights = self.compute_effective_cooperative_weights()
        cooperative_potential_base = self.compute_cooperative_potential_base(interval_idx, effective_weights)
        return self.compute_collective_potential(interval_idx, cooperative_potential_base, effective_weights)[2]

    def compute_cooperative_potential_base(self, interval_idx, effective_weights):
        if self.cooperative_mode == "norm-p":
            cooperative_cost = self.compute_norm_p_cooperative_cost(interval_idx, effective_weights)
            return _clip_01(1.0 - cooperative_cost)
        if self.cooperative_mode == "geometric_consensus":
            return self.compute_geometric_consensus_potential(interval_idx, effective_weights)
        raise ValueError(f"Unsupported cooperative potential mode: {self.cooperative_mode}")

    def compute_norm_p_cooperative_cost(self, interval_idx, effective_weights):
        cooperative_cost_power = 0.0
        for local_track in self.local_tracks:
            local_cost = _clip_01(float(local_track.local_cost_array[interval_idx]))
            cooperative_cost_power += (
                float(effective_weights[local_track.controller_id])
                * (local_cost ** self.cooperative_p)
            )
        return float(cooperative_cost_power ** (1.0 / self.cooperative_p))

    def compute_geometric_consensus_potential(self, interval_idx, effective_weights):
        log_potential = 0.0
        for local_track in self.local_tracks:
            local_cost = _clip_01(float(local_track.local_cost_array[interval_idx]))
            local_remaining = max(self.epsilon_geom_conc, 1.0 - local_cost)
            log_potential += float(effective_weights[local_track.controller_id]) * np.log(local_remaining)
        return _clip_01(float(np.exp(log_potential)))

    def compute_collective_potential(self, interval_idx, cooperative_potential_base, effective_weights):
        if not self.synchrony_enabled:
            return 1.0, 1.0, cooperative_potential_base

        arithmetic_health = 0.0
        for local_track in self.local_tracks:
            local_health = max(self.epsilon_geom_conc, float(local_track.local_health_array[interval_idx]))
            arithmetic_health += float(effective_weights[local_track.controller_id]) * local_health
        synchrony_health_balance = _clip_01(float(cooperative_potential_base / arithmetic_health))
        synchrony_factor = _clip_01(float(synchrony_health_balance ** self.synchrony_strength))
        collective_potential = _clip_01(float(cooperative_potential_base * synchrony_factor))
        return synchrony_health_balance, synchrony_factor, collective_potential

    def compute_effective_cooperative_weights(self):
        if not self.adaptive_weights_enabled:
            return dict(self.cooperative_weights)

        total_need = 0.0
        for local_track in self.local_tracks:
            total_need += float(self.previous_local_needs[local_track.controller_id])

        n_tracks = len(self.local_tracks)
        effective_weights = {}
        if total_need <= 0.0:
            uniform_weight = 1.0 / n_tracks
            for local_track in self.local_tracks:
                effective_weights[local_track.controller_id] = uniform_weight
            return effective_weights

        adaptive_span = 1.0 - n_tracks * self.weight_floor
        for local_track in self.local_tracks:
            local_need = float(self.previous_local_needs[local_track.controller_id])
            effective_weights[local_track.controller_id] = (
                self.weight_floor
                + adaptive_span * local_need / total_need
            )
        return effective_weights

    def write_effective_cooperative_weights(self, interval_idx, effective_weights):
        for local_track in self.local_tracks:
            local_track.cooperative_weight_effective_array[interval_idx] = effective_weights[local_track.controller_id]

    def update_local_needs(self, interval_idx):
        deterioration_weight = 0.0
        smoothing_alpha = 0.0
        if self.adaptive_weights_enabled:
            deterioration_weight = self.deterioration_weight
            smoothing_alpha = self.smoothing_alpha

        for local_track in self.local_tracks:
            controller_id = local_track.controller_id
            local_cost = _clip_01(float(local_track.local_cost_array[interval_idx]))
            previous_cost = _clip_01(float(self.previous_local_costs[controller_id]))
            deterioration = max(0.0, local_cost - previous_cost)
            local_need = (local_cost + deterioration_weight * deterioration) / (1.0 + deterioration_weight)
            previous_need = _clip_01(float(self.previous_local_needs[controller_id]))
            if smoothing_alpha == 0.0:
                smoothed_need = local_need
            else:
                smoothed_need = (1.0 - smoothing_alpha) * previous_need + smoothing_alpha * local_need
            smoothed_need = _clip_01(float(smoothed_need))
            local_track.local_need_array[interval_idx] = smoothed_need
            self.previous_local_costs[controller_id] = local_cost
            self.previous_local_needs[controller_id] = smoothed_need

    def compute_delta_cooperative_potential(self, cooperative_potential):
        if not self.has_previous_cooperative_potential:
            self.previous_cooperative_potential = cooperative_potential
            self.has_previous_cooperative_potential = True
        delta_cooperative_potential = cooperative_potential - self.previous_cooperative_potential
        self.previous_cooperative_potential = cooperative_potential
        return _clip_signed(float(delta_cooperative_potential))

    def compute_cooperative_regime(self, cooperative_potential):
        if self.cooperative_reward_mode != "regime_residual":
            return 0
        for regime_idx, threshold in enumerate(self.regime_thresholds):
            if cooperative_potential < threshold:
                return regime_idx
        return len(self.regime_thresholds)

    def compute_regime_residual_weight(self, cooperative_regime):
        if self.cooperative_reward_mode == "regime_residual":
            return self.residual_weight_by_regime[cooperative_regime]
        return self.residual_weight

    def compute_residual_weight_effective(self, cooperative_potential, cooperative_regime):
        if self.adaptive_residual_enabled:
            return float(
                self.residual_weight_min
                + (self.residual_weight_max - self.residual_weight_min)
                * (cooperative_potential ** self.residual_weight_gamma)
            )
        return self.compute_regime_residual_weight(cooperative_regime)

    def compute_cooperative_progress_residual(
        self,
        cooperative_potential,
        delta_cooperative_potential,
        residual_weight,
    ):
        cooperative_residual = 1.0 - cooperative_potential
        return _clip_signed(delta_cooperative_potential - residual_weight * cooperative_residual)

    def transform_basic_cooperative_reward(
        self,
        reward_mode,
        cooperative_potential,
        delta_cooperative_potential,
        residual_weight,
    ):
        if reward_mode == "absolute":
            return _clip_signed(2.0 * cooperative_potential - 1.0)
        if reward_mode == "progress":
            return _clip_signed(delta_cooperative_potential)
        if reward_mode == "progress_residual":
            cooperative_residual = 1.0 - cooperative_potential
            return _clip_signed(delta_cooperative_potential - residual_weight * cooperative_residual)
        raise ValueError(f"Unsupported cooperative reward mode: {reward_mode}")

    def transform_cooperative_reward(
        self,
        cooperative_potential,
        delta_cooperative_potential,
        cooperative_regime,
        residual_weight_effective,
    ):
        if self.cooperative_reward_mode == "absolute":
            return self.transform_basic_cooperative_reward(
                self.cooperative_reward_mode,
                cooperative_potential,
                delta_cooperative_potential,
                self.residual_weight,
            )
        if self.cooperative_reward_mode == "progress":
            return self.transform_basic_cooperative_reward(
                self.cooperative_reward_mode,
                cooperative_potential,
                delta_cooperative_potential,
                self.residual_weight,
            )
        if self.cooperative_reward_mode == "progress_residual":
            return self.transform_basic_cooperative_reward(
                self.cooperative_reward_mode,
                cooperative_potential,
                delta_cooperative_potential,
                residual_weight_effective,
            )
        if self.cooperative_reward_mode == "regime_residual":
            regime_reward_mode = self.reward_mode_by_regime[cooperative_regime]
            return self.transform_basic_cooperative_reward(
                regime_reward_mode,
                cooperative_potential,
                delta_cooperative_potential,
                residual_weight_effective,
            )
        raise ValueError(f"Unsupported cooperative reward mode: {self.cooperative_reward_mode}")

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
