def _clip(value, lower, upper):
    return min(upper, max(lower, value))


class PIDController:
    def __init__(self, runtime, controller_config, objectives_in):
        self.step = runtime.step
        self.controller_id = controller_config["id"]
        self.objective = controller_config["objective"]
        self.setpoint_raw = float(controller_config["setpoint_raw"])
        self.setpoint_norm = float(controller_config["setpoint_norm"])
        self.error_sign = controller_config["error_sign"]
        self.calculation_mode = controller_config["calculation_mode"]
        self.initial_gains = controller_config["initial_gains"]
        self.gain_update = controller_config["gain_update"]
        self.antiwindup_method = controller_config["antiwindup"]["method"]
        self.objectives_in = objectives_in

        if self.objectives_in not in ("raw", "norm"):
            raise ValueError(f"Unsupported control objectives_in: {self.objectives_in}")
        if self.calculation_mode not in ("position", "incremental"):
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
        self.p_term_raw_array = getattr(self.step, f"p_term_{self.controller_id}_raw")
        self.p_term_norm_array = getattr(self.step, f"p_term_{self.controller_id}_norm")
        self.i_term_raw_array = getattr(self.step, f"i_term_{self.controller_id}_raw")
        self.i_term_norm_array = getattr(self.step, f"i_term_{self.controller_id}_norm")
        self.d_term_raw_array = getattr(self.step, f"d_term_{self.controller_id}_raw")
        self.d_term_norm_array = getattr(self.step, f"d_term_{self.controller_id}_norm")
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

        if self.calculation_mode == "incremental":
            self.p_delta_term_raw_array = getattr(self.step, f"p_delta_term_{self.controller_id}_raw")
            self.p_delta_term_norm_array = getattr(self.step, f"p_delta_term_{self.controller_id}_norm")
            self.i_delta_term_raw_array = getattr(self.step, f"i_delta_term_{self.controller_id}_raw")
            self.i_delta_term_norm_array = getattr(self.step, f"i_delta_term_{self.controller_id}_norm")
            self.d_delta_term_raw_array = getattr(self.step, f"d_delta_term_{self.controller_id}_raw")
            self.d_delta_term_norm_array = getattr(self.step, f"d_delta_term_{self.controller_id}_norm")
            self.delta_u_incremental_raw_array = getattr(self.step, f"delta_u_incremental_{self.controller_id}_raw")
            self.delta_u_incremental_norm_array = getattr(self.step, f"delta_u_incremental_{self.controller_id}_norm")

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
        self.previous_u_demand_raw = 0.0
        self.previous_u_demand_norm = 0.0
        self.previous_u_alloc = 0.0
        self.integral_enabled = True

        self.current_error_raw = 0.0
        self.current_error_norm = 0.0
        self.current_derivative_error_raw = 0.0
        self.current_derivative_error_norm = 0.0
        self.current_p_term_raw = 0.0
        self.current_p_term_norm = 0.0
        self.current_i_term_raw = 0.0
        self.current_i_term_norm = 0.0
        self.current_d_term_raw = 0.0
        self.current_d_term_norm = 0.0
        self.current_p_delta_term_raw = 0.0
        self.current_p_delta_term_norm = 0.0
        self.current_i_delta_term_raw = 0.0
        self.current_i_delta_term_norm = 0.0
        self.current_d_delta_term_raw = 0.0
        self.current_d_delta_term_norm = 0.0
        self.current_delta_u_incremental_raw = 0.0
        self.current_delta_u_incremental_norm = 0.0
        self.current_u_demand = 0.0
        self.current_u_demand_raw = 0.0
        self.current_u_demand_norm = 0.0
        self.current_delta_u_demand = 0.0
        self.current_delta_u_demand_raw = 0.0
        self.current_delta_u_demand_norm = 0.0

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
        self.previous_u_alloc = 0.0
        self.integral_enabled = True

        self.current_error_raw = self.previous_error_raw
        self.current_error_norm = self.previous_error_norm
        self.current_derivative_error_raw = 0.0
        self.current_derivative_error_norm = 0.0
        self.update_position_terms(
            self.current_error_raw,
            self.current_error_norm,
            self.current_derivative_error_raw,
            self.current_derivative_error_norm,
            self.integral_error_raw,
            self.integral_error_norm,
        )
        self.reset_incremental_terms()
        self.current_u_demand_raw = self.position_u_demand_raw()
        self.current_u_demand_norm = self.position_u_demand_norm()
        self.current_u_demand = self.active_value(self.current_u_demand_raw, self.current_u_demand_norm)
        self.current_delta_u_demand = 0.0
        self.current_delta_u_demand_raw = 0.0
        self.current_delta_u_demand_norm = 0.0
        self.previous_u_demand = self.current_u_demand
        self.previous_u_demand_raw = self.current_u_demand_raw
        self.previous_u_demand_norm = self.current_u_demand_norm

        self.write_u_demand_signals(step_idx)
        self.apply_allocation(step_idx, 0.0, 0.0)

    def compute_u_demand(self, step_idx, dt):
        error_raw = self.compute_error_raw(step_idx)
        error_norm = self.compute_error_norm(step_idx)
        derivative_error_raw = (error_raw - self.previous_error_raw) / dt
        derivative_error_norm = (error_norm - self.previous_error_norm) / dt

        if self.calculation_mode == "position":
            u_demand = self.compute_position_u_demand(
                error_raw,
                error_norm,
                derivative_error_raw,
                derivative_error_norm,
                dt,
            )
        else:
            u_demand = self.compute_incremental_u_demand(
                error_raw,
                error_norm,
                derivative_error_raw,
                derivative_error_norm,
                dt,
            )

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
        self.update_position_terms(
            error_raw,
            error_norm,
            derivative_error_raw,
            derivative_error_norm,
            self.candidate_integral_error_raw,
            self.candidate_integral_error_norm,
        )
        self.reset_incremental_terms()
        self.current_u_demand_raw = self.position_u_demand_raw()
        self.current_u_demand_norm = self.position_u_demand_norm()
        self.current_delta_u_demand_raw = self.current_u_demand_raw - self.previous_u_demand_raw
        self.current_delta_u_demand_norm = self.current_u_demand_norm - self.previous_u_demand_norm
        return self.active_value(self.current_u_demand_raw, self.current_u_demand_norm)

    def compute_incremental_u_demand(
        self,
        error_raw,
        error_norm,
        derivative_error_raw,
        derivative_error_norm,
        dt,
    ):
        delta_error_raw = error_raw - self.previous_error_raw
        previous_delta_error_raw = self.previous_error_raw - self.previous_previous_error_raw
        second_difference_raw = delta_error_raw - previous_delta_error_raw
        integral_step_raw = error_raw * dt if self.integral_enabled else 0.0

        delta_error_norm = error_norm - self.previous_error_norm
        previous_delta_error_norm = self.previous_error_norm - self.previous_previous_error_norm
        second_difference_norm = delta_error_norm - previous_delta_error_norm
        integral_step_norm = error_norm * dt if self.integral_enabled else 0.0

        self.candidate_integral_error_raw = self.integral_error_raw + integral_step_raw
        self.candidate_integral_error_norm = self.integral_error_norm + integral_step_norm
        self.update_position_terms(
            error_raw,
            error_norm,
            derivative_error_raw,
            derivative_error_norm,
            self.candidate_integral_error_raw,
            self.candidate_integral_error_norm,
        )
        self.current_p_delta_term_raw = self.kp * delta_error_raw
        self.current_p_delta_term_norm = self.kp * delta_error_norm
        self.current_i_delta_term_raw = self.ki * integral_step_raw
        self.current_i_delta_term_norm = self.ki * integral_step_norm
        self.current_d_delta_term_raw = self.kd * second_difference_raw / dt
        self.current_d_delta_term_norm = self.kd * second_difference_norm / dt
        self.current_delta_u_incremental_raw = (
            self.current_p_delta_term_raw
            + self.current_i_delta_term_raw
            + self.current_d_delta_term_raw
        )
        self.current_delta_u_incremental_norm = (
            self.current_p_delta_term_norm
            + self.current_i_delta_term_norm
            + self.current_d_delta_term_norm
        )
        self.current_u_demand_raw = self.previous_u_demand_raw + self.current_delta_u_incremental_raw
        self.current_u_demand_norm = self.previous_u_demand_norm + self.current_delta_u_incremental_norm
        self.current_delta_u_demand_raw = self.current_delta_u_incremental_raw
        self.current_delta_u_demand_norm = self.current_delta_u_incremental_norm
        delta_u = self.active_value(self.current_delta_u_incremental_raw, self.current_delta_u_incremental_norm)
        return self.previous_u_demand + delta_u

    def update_position_terms(
        self,
        error_raw,
        error_norm,
        derivative_error_raw,
        derivative_error_norm,
        integral_error_raw,
        integral_error_norm,
    ):
        self.current_p_term_raw = self.kp * error_raw
        self.current_p_term_norm = self.kp * error_norm
        self.current_i_term_raw = self.ki * integral_error_raw
        self.current_i_term_norm = self.ki * integral_error_norm
        self.current_d_term_raw = self.kd * derivative_error_raw
        self.current_d_term_norm = self.kd * derivative_error_norm

    def reset_incremental_terms(self):
        self.current_p_delta_term_raw = 0.0
        self.current_p_delta_term_norm = 0.0
        self.current_i_delta_term_raw = 0.0
        self.current_i_delta_term_norm = 0.0
        self.current_d_delta_term_raw = 0.0
        self.current_d_delta_term_norm = 0.0
        self.current_delta_u_incremental_raw = 0.0
        self.current_delta_u_incremental_norm = 0.0

    def position_u_demand_raw(self):
        return self.current_p_term_raw + self.current_i_term_raw + self.current_d_term_raw

    def position_u_demand_norm(self):
        return self.current_p_term_norm + self.current_i_term_norm + self.current_d_term_norm

    def active_value(self, raw_value, norm_value):
        if self.objectives_in == "norm":
            return norm_value
        return raw_value

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
        self.integral_error_norm_array[step_idx] = self.integral_error_norm
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
        integral_control_direction = self.ki * control_error
        return u_unallocated * integral_control_direction <= 0.0

    def commit_step_history(self, u_alloc):
        self.previous_previous_error_raw = self.previous_error_raw
        self.previous_previous_error_norm = self.previous_error_norm
        self.previous_error_norm = self.current_error_norm
        self.previous_error_raw = self.current_error_raw
        self.previous_u_demand = self.current_u_demand
        self.previous_u_demand_raw = self.current_u_demand_raw
        self.previous_u_demand_norm = self.current_u_demand_norm
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

    def write_u_demand_signals(self, step_idx):
        self.error_raw_array[step_idx] = self.current_error_raw
        self.error_norm_array[step_idx] = self.current_error_norm
        self.derivative_error_raw_array[step_idx] = self.current_derivative_error_raw
        self.derivative_error_norm_array[step_idx] = self.current_derivative_error_norm
        self.integral_error_raw_array[step_idx] = self.integral_error_raw
        self.integral_error_norm_array[step_idx] = self.integral_error_norm
        self.p_term_raw_array[step_idx] = self.current_p_term_raw
        self.p_term_norm_array[step_idx] = self.current_p_term_norm
        self.i_term_raw_array[step_idx] = self.current_i_term_raw
        self.i_term_norm_array[step_idx] = self.current_i_term_norm
        self.d_term_raw_array[step_idx] = self.current_d_term_raw
        self.d_term_norm_array[step_idx] = self.current_d_term_norm
        self.u_demand_raw_array[step_idx] = self.current_u_demand_raw
        self.u_demand_norm_array[step_idx] = self.current_u_demand_norm
        self.delta_u_demand_raw_array[step_idx] = self.current_delta_u_demand_raw
        self.delta_u_demand_norm_array[step_idx] = self.current_delta_u_demand_norm
        if self.calculation_mode == "incremental":
            self.write_incremental_signals(step_idx)
        self.kp_array[step_idx] = self.kp
        self.ki_array[step_idx] = self.ki
        self.kd_array[step_idx] = self.kd

    def write_incremental_signals(self, step_idx):
        self.p_delta_term_raw_array[step_idx] = self.current_p_delta_term_raw
        self.p_delta_term_norm_array[step_idx] = self.current_p_delta_term_norm
        self.i_delta_term_raw_array[step_idx] = self.current_i_delta_term_raw
        self.i_delta_term_norm_array[step_idx] = self.current_i_delta_term_norm
        self.d_delta_term_raw_array[step_idx] = self.current_d_delta_term_raw
        self.d_delta_term_norm_array[step_idx] = self.current_d_delta_term_norm
        self.delta_u_incremental_raw_array[step_idx] = self.current_delta_u_incremental_raw
        self.delta_u_incremental_norm_array[step_idx] = self.current_delta_u_incremental_norm

    def write_allocation_signals(self, step_idx, u_alloc, delta_u_alloc, u_unallocated):
        self.u_alloc_raw_array[step_idx] = u_alloc
        self.u_alloc_norm_array[step_idx] = u_alloc
        self.delta_u_alloc_raw_array[step_idx] = delta_u_alloc
        self.delta_u_alloc_norm_array[step_idx] = delta_u_alloc
        self.u_unallocated_raw_array[step_idx] = u_unallocated
        self.u_unallocated_norm_array[step_idx] = u_unallocated
