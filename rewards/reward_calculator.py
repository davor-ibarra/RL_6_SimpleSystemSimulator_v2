from dataclasses import dataclass

import numpy as np


SUPPORTED_COOPERATIVE_MODES = ("norm-p", "geometric_consensus", "concurrent_progress")
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
    cost_array: object
    health_array: object
    reward_array: object
    controller_reward_array: object


@dataclass(frozen=True)
class CooperativeVariableTrack:
    variable_id: str
    potential_source_array: object
    need_cost_array: object
    balance_health_array: object
    nominal_weight: float
    need_array: object
    effective_weight_array: object


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

        self.build_local_tracks(reward_config["local"])
        self.build_cooperative_tracks(reward_config["cooperative"])
        self.configure_composition(reward_config["composition"])
        self.build_agent_tracks(agent_config)
        self.configure_assignment()

        self.strategic_enabled = bool(reward_config["strategic"]["enabled"])
        if self.strategic_enabled and self.strategic_weight <= 0.0:
            raise ValueError("strategic reward enabled with zero strategic weight")
        self.strategic_reward_signed = 0.0

        self.previous_cooperative_potential = 0.0
        self.has_previous_cooperative_potential = False
        self.previous_variable_costs = {}
        self.previous_variable_needs = {}
        self.previous_variable_healths = {}
        self.has_previous_variable_healths = False
        for variable_track in self.cooperative_variables:
            self.previous_variable_costs[variable_track.variable_id] = 0.0
            self.previous_variable_needs[variable_track.variable_id] = 0.0
            self.previous_variable_healths[variable_track.variable_id] = 0.0

        terminal_bonus_config = reward_config["terminal_learning_bonus"]
        self.terminal_bonus_enabled = bool(terminal_bonus_config["enabled"])
        self.terminal_bonus_by_reason = terminal_bonus_config["bonuses"]
        self.reason_name_by_code = {}
        for reason_name, reason_code in termination_config["reason_codes"].items():
            self.reason_name_by_code[int(reason_code)] = reason_name
        self.terminal_bonus = 0.0

    def build_local_tracks(self, local_config):
        self.local_tracks = []
        seen_controller_ids = set()
        for controller_config in local_config["controllers"]:
            controller_id = controller_config["controller"]
            if controller_id in seen_controller_ids:
                raise ValueError(f"Duplicate local reward controller: {controller_id}")
            seen_controller_ids.add(controller_id)

            components = []
            component_weight_sum = 0.0
            for component_config in controller_config["components"]:
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
                    cost_array=getattr(self.interval, f"local_cost_01_{controller_id}"),
                    health_array=getattr(self.interval, f"local_health_01_{controller_id}"),
                    reward_array=getattr(self.interval, f"local_reward_signed_{controller_id}"),
                    controller_reward_array=getattr(self.interval, f"reward_controller_{controller_id}"),
                )
            )
        if len(self.local_tracks) == 0:
            raise ValueError("local reward must declare at least one controller")

    def index_source_configs(self, variable_configs, label):
        source_config_by_id = {}
        for variable_config in variable_configs:
            variable_id = variable_config["id"]
            if variable_id in source_config_by_id:
                raise ValueError(f"Duplicate {label} variable id: {variable_id}")
            source_config_by_id[variable_id] = variable_config
        return source_config_by_id

    def build_cooperative_tracks(self, cooperative_config):
        self.cooperative_enabled = bool(cooperative_config["enabled"])
        self.cooperative_reward_mode = cooperative_config["reward_mode"]
        if self.cooperative_reward_mode not in SUPPORTED_COOPERATIVE_REWARD_MODES:
            raise ValueError(f"Unsupported cooperative reward mode: {self.cooperative_reward_mode}")

        potential_config = cooperative_config["potential"]
        self.cooperative_mode = potential_config["method"]
        if self.cooperative_mode not in SUPPORTED_COOPERATIVE_MODES:
            raise ValueError(f"Unsupported cooperative potential mode: {self.cooperative_mode}")
        self.cooperative_p = float(potential_config["p"])
        self.epsilon_geom_conc = float(potential_config["epsilon_geom_conc"])
        self.deterioration_order = float(potential_config["deterioration_order"])
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
        balance_config = cooperative_config["balance"]
        synchrony_config = cooperative_config["synchrony"]
        need_config_by_id = self.index_source_configs(
            adaptive_weights_config["need_variables"],
            "adaptive need",
        )
        health_config_by_id = self.index_source_configs(
            balance_config["health_variables"],
            "balance health",
        )

        potential_variable_configs = potential_config["variables"]
        potential_ids = {variable_config["id"] for variable_config in potential_variable_configs}
        if len(potential_ids) != len(potential_variable_configs):
            raise ValueError("cooperative potential variable ids must be unique")
        if set(need_config_by_id) != potential_ids:
            raise ValueError("adaptive need variable ids must match cooperative potential variables exactly")
        if set(health_config_by_id) != potential_ids:
            raise ValueError("balance health variable ids must match cooperative potential variables exactly")

        self.cooperative_variables = []
        nominal_weight_sum = 0.0
        for variable_config in potential_variable_configs:
            variable_id = variable_config["id"]
            nominal_weight = float(variable_config["weight"])
            if not np.isfinite(nominal_weight):
                raise ValueError("cooperative variable weights must be finite")
            if nominal_weight < 0.0:
                raise ValueError("cooperative variable weights must be non-negative")
            nominal_weight_sum += nominal_weight
            self.cooperative_variables.append(
                CooperativeVariableTrack(
                    variable_id=variable_id,
                    potential_source_array=getattr(self.interval, variable_config["source"]),
                    need_cost_array=getattr(self.interval, need_config_by_id[variable_id]["source"]),
                    balance_health_array=getattr(self.interval, health_config_by_id[variable_id]["source"]),
                    nominal_weight=nominal_weight,
                    need_array=getattr(self.interval, f"local_need_01_{variable_id}"),
                    effective_weight_array=getattr(
                        self.interval,
                        f"cooperative_weight_effective_{variable_id}",
                    ),
                )
            )
        if len(self.cooperative_variables) == 0:
            raise ValueError("cooperative potential must declare at least one variable")
        if abs(nominal_weight_sum - 1.0) > 1e-9:
            raise ValueError("cooperative variable weights must sum to 1.0")

        self.configure_adaptive_weights(adaptive_weights_config)
        self.configure_balance(balance_config)
        self.configure_synchrony(synchrony_config)
        self.configure_adaptive_residual(cooperative_config["adaptive_residual"])
        self.configure_regime_residual(cooperative_config)

    def configure_adaptive_weights(self, adaptive_weights_config):
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
        if self.weight_floor < 0.0 or self.weight_floor > 1.0 / len(self.cooperative_variables):
            raise ValueError("adaptive_weights weight_floor must be between 0 and 1/N")
        if self.deterioration_weight < 0.0:
            raise ValueError("adaptive_weights deterioration_weight must be non-negative")
        if self.smoothing_alpha < 0.0 or self.smoothing_alpha > 1.0:
            raise ValueError("adaptive_weights smoothing_alpha must be between 0 and 1")

    def configure_balance(self, balance_config):
        self.balance_enabled = bool(balance_config["enabled"])
        self.balance_strength = float(balance_config["strength"])

    def configure_synchrony(self, synchrony_config):
        self.synchrony_enabled = bool(synchrony_config["enabled"])
        self.synchrony_source_array = getattr(self.interval, synchrony_config["source"])
        self.synchrony_strength = float(synchrony_config["strength"])

    def configure_adaptive_residual(self, adaptive_residual_config):
        self.adaptive_residual_enabled = bool(adaptive_residual_config["enabled"])
        self.adaptive_residual_source_array = getattr(self.interval, adaptive_residual_config["source"])
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

    def configure_regime_residual(self, cooperative_config):
        self.regime_thresholds = ()
        self.reward_mode_by_regime = ()
        self.residual_weight_by_regime = ()
        self.regime_array = None
        if self.cooperative_reward_mode != "regime_residual":
            return

        regime_config = cooperative_config["regime_residual_params"]
        self.regime_array = getattr(self.interval, regime_config["signal"])
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

        configured_reward_modes = regime_config["reward_mode_by_regime"]
        reward_modes = []
        for regime_idx in range(n_regimes):
            reward_mode = configured_reward_modes[str(regime_idx)]
            if reward_mode not in SUPPORTED_REGIME_REWARD_MODES:
                raise ValueError(f"Unsupported regime reward mode: {reward_mode}")
            reward_modes.append(reward_mode)
        if len(configured_reward_modes) != n_regimes:
            raise ValueError("regime_residual reward_mode_by_regime must match the number of regimes")
        self.reward_mode_by_regime = tuple(reward_modes)

        configured_residual_weights = regime_config["residual_weight_by_regime"]
        residual_weights = []
        for regime_idx in range(n_regimes):
            weight = float(configured_residual_weights[str(regime_idx)])
            if not np.isfinite(weight):
                raise ValueError("regime_residual residual weights must be finite")
            if weight < 0.0:
                raise ValueError("regime_residual residual weights must be non-negative")
            residual_weights.append(weight)
        if len(configured_residual_weights) != n_regimes:
            raise ValueError("regime_residual residual_weight_by_regime must match the number of regimes")
        self.residual_weight_by_regime = tuple(residual_weights)

    def configure_composition(self, composition_config):
        weights = composition_config["weights"]
        self.local_weight = float(weights["local"])
        self.cooperative_weight = float(weights["cooperative"])
        self.strategic_weight = float(weights["strategic"])
        composition_weights = (self.local_weight, self.cooperative_weight, self.strategic_weight)
        if not all(np.isfinite(weight) for weight in composition_weights):
            raise ValueError("reward composition weights must be finite")
        if any(weight < 0.0 for weight in composition_weights):
            raise ValueError("reward composition weights must be non-negative")
        if abs(sum(composition_weights) - 1.0) > 1e-9:
            raise ValueError("reward composition weights must sum to 1.0")
        if not self.cooperative_enabled and self.cooperative_weight != 0.0:
            raise ValueError("cooperative reward disabled with non-zero cooperative composition weight")

    def build_agent_tracks(self, agent_config):
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

    def configure_assignment(self):
        if self.assignment == "controller_reward":
            self.assign_rewards = self.assign_controller_rewards
        elif self.assignment == "global_reward":
            self.assign_rewards = self.assign_global_reward
        elif self.assignment == "agent_reward":
            self.assign_rewards = self.assign_agent_reward
        else:
            raise ValueError(f"Unsupported reward assignment: {self.assignment}")

    def reset_episode(self):
        self.strategic_reward_signed = 0.0
        self.terminal_bonus = 0.0
        self.previous_cooperative_potential = 0.0
        self.has_previous_cooperative_potential = False
        self.has_previous_variable_healths = False
        for variable_track in self.cooperative_variables:
            self.previous_variable_costs[variable_track.variable_id] = 0.0
            self.previous_variable_needs[variable_track.variable_id] = 0.0
            self.previous_variable_healths[variable_track.variable_id] = 0.0

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
            local_track.cost_array[interval_idx] = local_cost
            local_track.health_array[interval_idx] = 1.0 - local_cost
            local_track.reward_array[interval_idx] = _reward_from_cost(local_cost)

    def compute_cooperative_reward(self, interval_idx):
        effective_weights = self.compute_effective_cooperative_weights()
        self.write_effective_cooperative_weights(interval_idx, effective_weights)
        if self.cooperative_enabled:
            (
                cooperative_potential_base,
                cooperative_positive_progress,
                cooperative_deterioration,
                cooperative_progress_signed,
            ) = self.compute_cooperative_potential_base(interval_idx, effective_weights)
            (
                health_balance,
                balance_factor,
                synchrony_factor,
                cooperative_potential,
            ) = self.compute_collective_potential(
                interval_idx,
                cooperative_potential_base,
                effective_weights,
            )
            cooperative_cost = _clip_01(1.0 - cooperative_potential)
            delta_cooperative_potential = self.compute_delta_cooperative_potential(cooperative_potential)
            cooperative_regime = self.compute_cooperative_regime(cooperative_potential)
        else:
            cooperative_potential_base = 1.0
            cooperative_positive_progress = 0.0
            cooperative_deterioration = 0.0
            cooperative_progress_signed = 0.0
            health_balance = 1.0
            balance_factor = 1.0
            synchrony_factor = 1.0
            cooperative_potential = 1.0
            cooperative_cost = 0.0
            delta_cooperative_potential = 0.0
            cooperative_regime = self.compute_cooperative_regime(cooperative_potential)

        self.interval.collective_potential_01[interval_idx] = cooperative_potential
        self.interval.cooperative_potential_01[interval_idx] = cooperative_potential
        residual_weight_effective = self.compute_residual_weight_effective(
            interval_idx,
            cooperative_regime,
        )
        if self.cooperative_enabled:
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
            cooperative_progress_residual = 0.0
            cooperative_reward = 0.0

        self.interval.cooperative_cost_01[interval_idx] = cooperative_cost
        self.interval.cooperative_positive_progress_01[interval_idx] = cooperative_positive_progress
        self.interval.cooperative_deterioration_01[interval_idx] = cooperative_deterioration
        self.interval.cooperative_progress_signed[interval_idx] = cooperative_progress_signed
        self.interval.cooperative_potential_base_01[interval_idx] = cooperative_potential_base
        self.interval.health_balance_01[interval_idx] = health_balance
        self.interval.balance_factor_01[interval_idx] = balance_factor
        self.interval.synchrony_factor_01[interval_idx] = synchrony_factor
        self.interval.delta_cooperative_potential_signed[interval_idx] = delta_cooperative_potential
        self.interval.cooperative_progress_residual_signed[interval_idx] = cooperative_progress_residual
        self.interval.residual_weight_effective[interval_idx] = residual_weight_effective
        self.interval.cooperative_reward_signed[interval_idx] = cooperative_reward
        if self.cooperative_reward_mode == "regime_residual":
            self.regime_array[interval_idx] = cooperative_regime
        self.update_cooperative_variable_needs(interval_idx)

    def compute_cooperative_cost(self, interval_idx):
        cooperative_potential = self.compute_cooperative_potential(interval_idx)
        return _clip_01(1.0 - cooperative_potential)

    def compute_cooperative_potential(self, interval_idx):
        effective_weights = self.compute_effective_cooperative_weights()
        cooperative_potential_base = self.compute_cooperative_potential_base(interval_idx, effective_weights)[0]
        return self.compute_collective_potential(interval_idx, cooperative_potential_base, effective_weights)[3]

    def compute_cooperative_potential_base(self, interval_idx, effective_weights):
        if self.cooperative_mode == "norm-p":
            cooperative_cost = self.compute_norm_p_cooperative_cost(interval_idx, effective_weights)
            return _clip_01(1.0 - cooperative_cost), 0.0, 0.0, 0.0
        if self.cooperative_mode == "geometric_consensus":
            cooperative_potential = self.compute_geometric_consensus_potential(interval_idx, effective_weights)
            return cooperative_potential, 0.0, 0.0, 0.0
        if self.cooperative_mode == "concurrent_progress":
            return self.compute_concurrent_progress_potential(interval_idx, effective_weights)
        raise ValueError(f"Unsupported cooperative potential mode: {self.cooperative_mode}")

    def compute_norm_p_cooperative_cost(self, interval_idx, effective_weights):
        cooperative_cost_power = 0.0
        for variable_track in self.cooperative_variables:
            variable_cost = _clip_01(float(variable_track.potential_source_array[interval_idx]))
            cooperative_cost_power += effective_weights[variable_track.variable_id] * (
                variable_cost ** self.cooperative_p
            )
        return float(cooperative_cost_power ** (1.0 / self.cooperative_p))

    def compute_geometric_consensus_potential(self, interval_idx, effective_weights):
        log_potential = 0.0
        for variable_track in self.cooperative_variables:
            variable_cost = _clip_01(float(variable_track.potential_source_array[interval_idx]))
            variable_health = max(self.epsilon_geom_conc, 1.0 - variable_cost)
            log_potential += effective_weights[variable_track.variable_id] * np.log(variable_health)
        return _clip_01(float(np.exp(log_potential)))

    def compute_concurrent_progress_potential(self, interval_idx, effective_weights):
        current_healths = {}
        for variable_track in self.cooperative_variables:
            variable_id = variable_track.variable_id
            current_healths[variable_id] = _clip_01(
                float(variable_track.potential_source_array[interval_idx])
            )

        if not self.has_previous_variable_healths:
            for variable_track in self.cooperative_variables:
                variable_id = variable_track.variable_id
                getattr(self.interval, f"health_delta_signed_{variable_id}")[interval_idx] = 0.0
                self.previous_variable_healths[variable_id] = current_healths[variable_id]
            self.has_previous_variable_healths = True
            return 0.5, 0.0, 0.0, 0.0

        positive_progress = 1.0
        all_variables_improve = True
        deterioration_power = 0.0
        for variable_track in self.cooperative_variables:
            variable_id = variable_track.variable_id
            health_delta = current_healths[variable_id] - self.previous_variable_healths[variable_id]
            getattr(self.interval, f"health_delta_signed_{variable_id}")[interval_idx] = health_delta

            variable_progress = max(health_delta, 0.0)
            variable_deterioration = max(-health_delta, 0.0)
            if variable_progress == 0.0:
                all_variables_improve = False
            variable_weight = effective_weights[variable_id]
            positive_progress *= variable_progress ** variable_weight
            deterioration_power += variable_weight * (
                variable_deterioration ** self.deterioration_order
            )
            self.previous_variable_healths[variable_id] = current_healths[variable_id]

        if not all_variables_improve:
            positive_progress = 0.0
        cooperative_deterioration = deterioration_power ** (1.0 / self.deterioration_order)
        cooperative_progress = positive_progress - cooperative_deterioration
        cooperative_potential = 0.5 * (cooperative_progress + 1.0)
        return (
            _clip_01(float(cooperative_potential)),
            _clip_01(float(positive_progress)),
            _clip_01(float(cooperative_deterioration)),
            _clip_signed(float(cooperative_progress)),
        )

    def compute_collective_potential(self, interval_idx, cooperative_potential_base, effective_weights):
        health_balance, balance_factor = self.compute_health_balance(interval_idx, effective_weights)
        synchrony_factor = self.compute_synchrony_factor(interval_idx)
        collective_potential = _clip_01(
            float(cooperative_potential_base * balance_factor * synchrony_factor)
        )
        return health_balance, balance_factor, synchrony_factor, collective_potential

    def compute_health_balance(self, interval_idx, effective_weights):
        if not self.balance_enabled:
            return 1.0, 1.0

        geometric_health = 1.0
        arithmetic_health = 0.0
        for variable_track in self.cooperative_variables:
            variable_health = _clip_01(float(variable_track.balance_health_array[interval_idx]))
            variable_weight = effective_weights[variable_track.variable_id]
            geometric_health *= variable_health ** variable_weight
            arithmetic_health += variable_weight * variable_health

        if arithmetic_health == 0.0:
            health_balance = 1.0
        else:
            health_balance = _clip_01(float(geometric_health / arithmetic_health))
        if self.balance_strength == 0.0:
            balance_factor = 1.0
        else:
            balance_factor = _clip_01(float(health_balance ** self.balance_strength))
        return health_balance, balance_factor

    def compute_synchrony_factor(self, interval_idx):
        if not self.synchrony_enabled:
            return 1.0
        if self.synchrony_strength == 0.0:
            return 1.0
        synchrony = _clip_01(float(self.synchrony_source_array[interval_idx]))
        return _clip_01(float(synchrony ** self.synchrony_strength))

    def compute_effective_cooperative_weights(self):
        if not self.adaptive_weights_enabled:
            return {
                variable_track.variable_id: variable_track.nominal_weight
                for variable_track in self.cooperative_variables
            }

        total_need = 0.0
        for variable_track in self.cooperative_variables:
            total_need += float(self.previous_variable_needs[variable_track.variable_id])

        n_variables = len(self.cooperative_variables)
        effective_weights = {}
        if total_need <= 0.0:
            uniform_weight = 1.0 / n_variables
            for variable_track in self.cooperative_variables:
                effective_weights[variable_track.variable_id] = uniform_weight
            return effective_weights

        adaptive_span = 1.0 - n_variables * self.weight_floor
        for variable_track in self.cooperative_variables:
            variable_need = float(self.previous_variable_needs[variable_track.variable_id])
            effective_weights[variable_track.variable_id] = (
                self.weight_floor + adaptive_span * variable_need / total_need
            )
        return effective_weights

    def write_effective_cooperative_weights(self, interval_idx, effective_weights):
        for variable_track in self.cooperative_variables:
            variable_track.effective_weight_array[interval_idx] = effective_weights[variable_track.variable_id]

    def update_cooperative_variable_needs(self, interval_idx):
        deterioration_weight = 0.0
        smoothing_alpha = 0.0
        if self.adaptive_weights_enabled:
            deterioration_weight = self.deterioration_weight
            smoothing_alpha = self.smoothing_alpha

        for variable_track in self.cooperative_variables:
            variable_id = variable_track.variable_id
            variable_cost = _clip_01(float(variable_track.need_cost_array[interval_idx]))
            previous_cost = _clip_01(float(self.previous_variable_costs[variable_id]))
            deterioration = max(0.0, variable_cost - previous_cost)
            variable_need = (variable_cost + deterioration_weight * deterioration) / (1.0 + deterioration_weight)
            previous_need = _clip_01(float(self.previous_variable_needs[variable_id]))
            if smoothing_alpha == 0.0:
                smoothed_need = variable_need
            else:
                smoothed_need = (1.0 - smoothing_alpha) * previous_need + smoothing_alpha * variable_need
            smoothed_need = _clip_01(float(smoothed_need))
            variable_track.need_array[interval_idx] = smoothed_need
            self.previous_variable_costs[variable_id] = variable_cost
            self.previous_variable_needs[variable_id] = smoothed_need

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

    def compute_residual_weight_effective(self, interval_idx, cooperative_regime):
        if self.adaptive_residual_enabled:
            residual_source = _clip_01(float(self.adaptive_residual_source_array[interval_idx]))
            return float(
                self.residual_weight_min
                + (self.residual_weight_max - self.residual_weight_min)
                * (residual_source ** self.residual_weight_gamma)
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
                self.local_weight * local_track.reward_array[interval_idx]
                + self.cooperative_weight * cooperative_reward
                + self.strategic_weight * self.strategic_reward_signed
            )
            local_track.controller_reward_array[interval_idx] = _clip_signed(float(controller_reward))

    def compose_global_reward(self, interval_idx):
        global_reward = 0.0
        for local_track in self.local_tracks:
            global_reward += local_track.controller_reward_array[interval_idx]
        global_reward = _clip_signed(float(global_reward / len(self.local_tracks)))
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
                return local_track.controller_reward_array[interval_idx]
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
