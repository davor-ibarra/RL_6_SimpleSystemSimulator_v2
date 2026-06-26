def _clip(value, lower, upper):
    return min(upper, max(lower, value))


def normalize_physical_value(value, limits, output_limits):
    source_min = float(limits[0])
    source_max = float(limits[1])
    target_min = float(output_limits[0])
    target_max = float(output_limits[1])
    if source_max <= source_min:
        raise ValueError("physical normalization range must be increasing")
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def normalize_signal_clipped(value, limits):
    lower = float(limits[0])
    upper = float(limits[1])
    if upper <= lower:
        raise ValueError("signal normalization range must be increasing")
    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    return _clip((value - midpoint) / half_range, -1.0, 1.0)


class PIDController:
    def __init__(
        self,
        runtime,
        controller_config,
        signal_ranges,
        objectives_in,
        objective_range,
        state_output_limits,
    ):
        self.step = runtime.step
        self.controller_id = controller_config["id"]
        self.objective = controller_config["objective"]
        self.setpoint_raw = float(controller_config["setpoint_raw"])
        self.error_sign = controller_config["error_sign"]
        self.calculation_mode = controller_config["calculation_mode"]
        self.initial_gains = controller_config["initial_gains"]
        self.gain_update = controller_config["gain_update"]
        self.antiwindup_method = controller_config["antiwindup"]["method"]
        self.signal_ranges = signal_ranges
        self.objectives_in = objectives_in
        self.objective_range = objective_range
        self.state_output_limits = state_output_limits
        self.setpoint_norm = normalize_physical_value(
            self.setpoint_raw,
            self.objective_range,
            self.state_output_limits,
        )

        if self.objectives_in not in ("raw", "norm"):
            raise ValueError(f"Unsupported control objectives_in: {self.objectives_in}")
        if self.calculation_mode not in ("position", "incremental", "incremental_velocity"):
            raise ValueError(f"Unsupported PID calculation_mode: {self.calculation_mode}")
        if self.error_sign not in ("measurement_minus_setpoint", "setpoint_minus_measurement"):
            raise ValueError(f"Unsupported error_sign: {self.error_sign}")
        if self.antiwindup_method not in ("conditional", "none"):
            raise ValueError(f"Unsupported antiwindup method: {self.antiwindup_method}")

        self.measurement_raw = getattr(self.step, f"{self.objective}_raw")
        self.measurement_norm = getattr(self.step, f"{self.objective}_norm")
        self.error_raw_array = getattr(self.step, f"error_{self.controller_id}_raw")
        self.error_norm_array = getattr(self.step, f"error_{self.controller_id}_norm")
        self.derivative_error_raw_array = getattr(self.step, f"derivative_error_{self.controller_id}_raw")
        self.derivative_error_norm_array = getattr(self.step, f"derivative_error_{self.controller_id}_norm")
        self.integral_error_raw_array = getattr(self.step, f"integral_error_{self.controller_id}_raw")
        self.integral_error_norm_array = getattr(self.step, f"integral_error_{self.controller_id}_norm")
        self.u_demand_raw_array = getattr(self.step, f"u_demand_{self.controller_id}_raw")
        self.u_demand_norm_array = getattr(self.step, f"u_demand_{self.controller_id}_norm")
        self.delta_u_demand_raw_array = getattr(self.step, f"delta_u_demand_{self.controller_id}_raw")
        self.delta_u_demand_norm_array = getattr(self.step, f"delta_u_demand_{self.controller_id}_norm")
        self.u_alloc_raw_array = getattr(self.step, f"u_alloc_{self.controller_id}_raw")
        self.u_alloc_norm_array = getattr(self.step, f"u_alloc_{self.controller_id}_norm")
        self.delta_u_alloc_raw_array = getattr(self.step, f"delta_u_alloc_{self.controller_id}_raw")
        self.delta_u_alloc_norm_array = getattr(self.step, f"delta_u_alloc_{self.controller_id}_norm")
        self.u_unallocated_raw_array = getattr(self.step, f"u_unallocated_{self.controller_id}_raw")
        self.u_unallocated_norm_array = getattr(self.step, f"u_unallocated_{self.controller_id}_norm")
        self.kp_array = getattr(self.step, f"kp_{self.controller_id}")
        self.ki_array = getattr(self.step, f"ki_{self.controller_id}")
        self.kd_array = getattr(self.step, f"kd_{self.controller_id}")

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
        self.integral_error_raw = 0.0
        self.integral_error_norm = 0.0
        self.candidate_integral_error_raw = 0.0
        self.candidate_integral_error_norm = 0.0
        self.previous_error_raw = 0.0
        self.previous_previous_error_raw = 0.0
        self.previous_error_norm = 0.0
        self.previous_previous_error_norm = 0.0
        self.previous_u_demand = 0.0
        self.previous_u_alloc = 0.0
        self.integral_enabled = True

        self.current_error_raw = 0.0
        self.current_error_norm = 0.0
        self.current_derivative_error_raw = 0.0
        self.current_derivative_error_norm = 0.0
        self.current_u_demand = 0.0
        self.current_delta_u_demand = 0.0

    def reset_episode(self, step_idx):
        self.kp = float(self.initial_gains["kp"])
        self.ki = float(self.initial_gains["ki"])
        self.kd = float(self.initial_gains["kd"])
        self.integral_error_raw = 0.0
        self.integral_error_norm = 0.0
        self.candidate_integral_error_raw = 0.0
        self.candidate_integral_error_norm = 0.0
        self.previous_error_raw = self.compute_error_raw(step_idx)
        self.previous_previous_error_raw = self.previous_error_raw
        self.previous_error_norm = self.compute_error_norm(step_idx)
        self.previous_previous_error_norm = self.previous_error_norm
        self.previous_u_demand = self.compute_initial_position_u_demand()
        self.previous_u_alloc = 0.0
        self.integral_enabled = True

        self.current_error_raw = self.previous_error_raw
        self.current_error_norm = self.previous_error_norm
        self.current_derivative_error_raw = 0.0
        self.current_derivative_error_norm = 0.0
        self.current_u_demand = self.previous_u_demand
        self.current_delta_u_demand = 0.0
        self.write_u_demand_signals(step_idx)
        self.apply_allocation(step_idx, 0.0, 0.0)

    def compute_u_demand(self, step_idx, dt):
        error_raw = self.compute_error_raw(step_idx)
        error_norm = self.compute_error_norm(step_idx)
        derivative_error_raw = (error_raw - self.previous_error_raw) / dt
        derivative_error_norm = (error_norm - self.previous_error_norm) / dt

        if self.calculation_mode == "position":
            u_demand = self.compute_position_u_demand(error_raw, error_norm, derivative_error_raw, derivative_error_norm, dt)
        elif self.calculation_mode == "incremental":
            u_demand = self.compute_incremental_u_demand(error_raw, error_norm, dt)

        self.current_error_raw = error_raw
        self.current_error_norm = error_norm
        self.current_derivative_error_raw = derivative_error_raw
        self.current_derivative_error_norm = derivative_error_norm
        self.current_u_demand = u_demand
        self.current_delta_u_demand = u_demand - self.previous_u_demand
        self.write_u_demand_signals(step_idx)
        return u_demand

    def compute_position_u_demand(self, error_raw, error_norm, derivative_error_raw, derivative_error_norm, dt):
        self.candidate_integral_error_raw = self.integral_error_raw + error_raw * dt
        self.candidate_integral_error_norm = self.integral_error_norm + error_norm * dt
        if self.objectives_in == "norm":
            return (
                self.kp * error_norm
                + self.ki * self.candidate_integral_error_norm
                + self.kd * derivative_error_norm
            )
        return (
            self.kp * error_raw
            + self.ki * self.candidate_integral_error_raw
            + self.kd * derivative_error_raw
        )

    def compute_incremental_u_demand(self, error_raw, error_norm, dt):
        delta_error_norm = error_norm - self.previous_error_norm
        previous_delta_error_norm = self.previous_error_norm - self.previous_previous_error_norm
        second_difference_norm = delta_error_norm - previous_delta_error_norm
        integral_step_norm = error_norm * dt if self.integral_enabled else 0.0

        delta_error_raw = error_raw - self.previous_error_raw
        previous_delta_error_raw = self.previous_error_raw - self.previous_previous_error_raw
        second_difference_raw = delta_error_raw - previous_delta_error_raw
        integral_step_raw = error_raw * dt if self.integral_enabled else 0.0

        if self.objectives_in == "norm":
            delta_u = self.kp * delta_error_norm + self.ki * integral_step_norm + self.kd * second_difference_norm / dt
        else:
            delta_u = self.kp * delta_error_raw + self.ki * integral_step_raw + self.kd * second_difference_raw / dt

        self.candidate_integral_error_raw = self.integral_error_raw + integral_step_raw
        self.candidate_integral_error_norm = self.integral_error_norm + integral_step_norm
        return self.previous_u_demand + delta_u

    def apply_allocation(self, step_idx, u_alloc, saturation_proportion_global):
        u_unallocated = self.current_u_demand - u_alloc
        delta_u_alloc = u_alloc - self.previous_u_alloc
        self.write_allocation_signals(step_idx, u_alloc, delta_u_alloc, u_unallocated)

        integral_allowed = self.integral_should_advance(u_alloc, saturation_proportion_global)
        if self.calculation_mode == "position":
            if integral_allowed:
                self.integral_error_raw = self.candidate_integral_error_raw
                self.integral_error_norm = self.candidate_integral_error_norm
        else:
            self.integral_error_raw = self.candidate_integral_error_raw
            self.integral_error_norm = self.candidate_integral_error_norm
            self.integral_enabled = integral_allowed

        self.integral_error_raw_array[step_idx] = self.integral_error_raw
        self.integral_error_norm_array[step_idx] = self.normalized_integral_error_signal()
        self.commit_step_history(u_alloc)

    def integral_should_advance(self, u_alloc, saturation_proportion_global):
        if self.antiwindup_method == "none":
            return True
        if saturation_proportion_global <= 0.0:
            return True
        if self.ki <= 0.0:
            return True
        u_unallocated = self.current_u_demand - u_alloc
        control_error = self.current_error_norm if self.objectives_in == "norm" else self.current_error_raw
        return u_unallocated * control_error <= 0.0

    def commit_step_history(self, u_alloc):
        self.previous_previous_error_raw = self.previous_error_raw
        self.previous_previous_error_norm = self.previous_error_norm
        self.previous_error_norm = self.current_error_norm
        self.previous_error_raw = self.current_error_raw
        self.previous_u_demand = self.current_u_demand
        self.previous_u_alloc = u_alloc

    def apply_gain_delta(self, gain_name, requested_delta):
        if gain_name == "kp":
            previous_value = self.kp
        elif gain_name == "ki":
            previous_value = self.ki
        elif gain_name == "kd":
            previous_value = self.kd
        else:
            raise ValueError(f"Unknown gain: {gain_name}")

        gain_limits = self.gain_update[gain_name]
        next_value = _clip(previous_value + requested_delta, float(gain_limits["min"]), float(gain_limits["max"]))
        applied_delta = next_value - previous_value

        if gain_name == "kp":
            self.kp = next_value
        elif gain_name == "ki":
            self.ki = next_value
        else:
            self.kd = next_value

        return applied_delta

    def gain_value(self, gain_name):
        if gain_name == "kp":
            return self.kp
        if gain_name == "ki":
            return self.ki
        if gain_name == "kd":
            return self.kd
        raise ValueError(f"Unknown gain: {gain_name}")

    def compute_error_raw(self, source_idx):
        measurement_raw = self.measurement_raw[source_idx]
        if self.error_sign == "measurement_minus_setpoint":
            return measurement_raw - self.setpoint_raw
        return self.setpoint_raw - measurement_raw

    def compute_error_norm(self, source_idx):
        measurement_norm = self.measurement_norm[source_idx]
        if self.error_sign == "measurement_minus_setpoint":
            return measurement_norm - self.setpoint_norm
        return self.setpoint_norm - measurement_norm

    def compute_initial_position_u_demand(self):
        if self.objectives_in == "norm":
            return self.kp * self.previous_error_norm + self.ki * self.integral_error_norm
        return self.kp * self.previous_error_raw + self.ki * self.integral_error_raw

    def normalized_derivative_error_signal(self):
        return normalize_signal_clipped(
            self.current_derivative_error_norm,
            self.signal_ranges["derivative_error"],
        )

    def normalized_integral_error_signal(self):
        return normalize_signal_clipped(
            self.integral_error_norm,
            self.signal_ranges["integral_error"],
        )

    def write_u_demand_signals(self, step_idx):
        self.error_raw_array[step_idx] = self.current_error_raw
        self.error_norm_array[step_idx] = self.current_error_norm
        self.derivative_error_raw_array[step_idx] = self.current_derivative_error_raw
        self.derivative_error_norm_array[step_idx] = self.normalized_derivative_error_signal()
        self.integral_error_raw_array[step_idx] = self.integral_error_raw
        self.integral_error_norm_array[step_idx] = self.normalized_integral_error_signal()
        self.u_demand_raw_array[step_idx] = self.current_u_demand
        self.u_demand_norm_array[step_idx] = normalize_signal_clipped(self.current_u_demand, self.signal_ranges["u_demand"])
        self.delta_u_demand_raw_array[step_idx] = self.current_delta_u_demand
        self.delta_u_demand_norm_array[step_idx] = normalize_signal_clipped(
            self.current_delta_u_demand,
            self.signal_ranges["delta_u_demand"],
        )
        self.kp_array[step_idx] = self.kp
        self.ki_array[step_idx] = self.ki
        self.kd_array[step_idx] = self.kd

    def write_allocation_signals(self, step_idx, u_alloc, delta_u_alloc, u_unallocated):
        self.u_alloc_raw_array[step_idx] = u_alloc
        self.u_alloc_norm_array[step_idx] = normalize_signal_clipped(u_alloc, self.signal_ranges["u_alloc"])
        self.delta_u_alloc_raw_array[step_idx] = delta_u_alloc
        self.delta_u_alloc_norm_array[step_idx] = normalize_signal_clipped(
            delta_u_alloc,
            self.signal_ranges["delta_u_alloc"],
        )
        self.u_unallocated_raw_array[step_idx] = u_unallocated
        self.u_unallocated_norm_array[step_idx] = normalize_signal_clipped(
            u_unallocated,
            self.signal_ranges["u_unallocated"],
        )
