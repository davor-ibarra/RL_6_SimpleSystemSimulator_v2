from controllers.pid_controller import PIDController


def _clip(value, lower, upper):
    return min(upper, max(lower, value))


class ControllerManager:
    def __init__(self, runtime, control_config, agent_config, dt_sec):
        self.step = runtime.step
        self.interval = runtime.interval
        self.dt_sec = dt_sec
        self.global_limits = control_config["global_actuator_limits"]
        self.mixing_policy = control_config["mixing_policy"]
        self.objectives_in = control_config["objectives_in"]
        if self.mixing_policy != "sum":
            raise ValueError("Only sum mixing_policy is supported")
        if self.objectives_in not in ("raw", "norm"):
            raise ValueError("control.objectives_in must be raw or norm")

        self.controllers = []
        self.controllers_by_id = {}
        for controller_config in control_config["controllers"]:
            controller_id = controller_config["id"]
            controller = PIDController(
                runtime,
                controller_config,
                self.objectives_in,
            )
            self.controllers.append(controller)
            self.controllers_by_id[controller_id] = controller

        self.gain_agents = []
        for gain_agent_config in agent_config["gain_agents"]:
            self.gain_agents.append(
                (
                    gain_agent_config["controller"],
                    gain_agent_config["gain"],
                    getattr(self.interval, f"action_{gain_agent_config['id']}"),
                    getattr(self.interval, f"delta_gain_requested_{gain_agent_config['id']}"),
                    getattr(self.interval, f"delta_gain_applied_{gain_agent_config['id']}"),
                    getattr(self.interval, f"gain_value_{gain_agent_config['id']}"),
                )
            )

        self.previous_u_total = 0.0

    def reset_episode(self, step_idx):
        self.previous_u_total = 0.0
        self.step.u_total_raw[step_idx] = 0.0
        self.step.u_total[step_idx] = 0.0
        self.step.delta_u_total[step_idx] = 0.0
        self.step.is_saturated_global[step_idx] = 0.0
        self.step.saturation_excess[step_idx] = 0.0
        self.step.saturation_proportion_global[step_idx] = 0.0
        for controller in self.controllers:
            controller.reset_episode(step_idx)

    def apply_gain_actions(self, interval_idx):
        for (controller_id, gain_name,action_array, requested_array, applied_array, gain_value_array) in self.gain_agents:
            controller = self.controllers_by_id[controller_id]
            action_move = int(action_array[interval_idx])
            gain_limits = controller.gain_update[gain_name]
            requested_delta = action_move * float(gain_limits["delta_gain"])
            applied_delta = controller.apply_gain_delta(gain_name, requested_delta)
            requested_array[interval_idx] = requested_delta
            applied_array[interval_idx] = applied_delta
            gain_value_array[interval_idx] = controller.gain_value(gain_name)

    def compute_control_step(self, step_idx):
        u_demands = self.compute_local_u_demands(step_idx)
        u_total_raw = self.sum_u_demands(u_demands)
        u_total = self.saturate_u_total(u_total_raw)
        saturation_proportion_global = self.write_global_control(step_idx, u_total_raw, u_total)
        self.apply_global_allocation(step_idx, u_demands, u_total_raw, u_total, saturation_proportion_global)
        self.previous_u_total = u_total
        return u_total

    def compute_local_u_demands(self, step_idx):
        u_demands = []
        for controller in self.controllers:
            u_demand = controller.compute_u_demand(step_idx, self.dt_sec)
            u_demands.append((controller, u_demand))
        return u_demands

    def sum_u_demands(self, u_demands):
        u_total_raw = 0.0
        for controller, u_demand in u_demands:
            u_total_raw += u_demand
        return u_total_raw

    def saturate_u_total(self, u_total_raw):
        lower = float(self.global_limits[0])
        upper = float(self.global_limits[1])
        return _clip(u_total_raw, lower, upper)

    def write_global_control(self, step_idx, u_total_raw, u_total):
        delta_u_total = u_total - self.previous_u_total
        saturation_excess = abs(u_total_raw - u_total)
        is_saturated_global = 1.0 if saturation_excess > 1e-12 else 0.0
        if is_saturated_global and abs(u_total_raw) > 1e-12:
            allocation_scale = u_total / u_total_raw
            saturation_proportion_global = abs(1.0 - allocation_scale)
        else:
            saturation_proportion_global = 0.0

        self.step.u_total_raw[step_idx] = u_total_raw
        self.step.u_total[step_idx] = u_total
        self.step.delta_u_total[step_idx] = delta_u_total
        self.step.is_saturated_global[step_idx] = is_saturated_global
        self.step.saturation_excess[step_idx] = saturation_excess
        self.step.saturation_proportion_global[step_idx] = saturation_proportion_global
        return saturation_proportion_global

    def apply_global_allocation(self, step_idx, u_demands, u_total_raw, u_total, saturation_proportion_global):
        if abs(u_total_raw) > 1e-12:
            allocation_scale = u_total / u_total_raw
        else:
            allocation_scale = 1.0

        for controller, u_demand in u_demands:
            u_alloc = u_demand * allocation_scale
            controller.apply_allocation(step_idx, u_alloc, saturation_proportion_global)
