import numpy as np
from scipy.integrate import odeint


def _require_positive(name, value):
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _normalize_clipped(value, limits, output_limits):
    source_min = float(limits[0])
    source_max = float(limits[1])
    target_min = float(output_limits[0])
    target_max = float(output_limits[1])
    if source_max <= source_min:
        raise ValueError("normalization range must be increasing")
    ratio = (value - source_min) / (source_max - source_min)
    normalized = target_min + ratio * (target_max - target_min)
    return min(target_max, max(target_min, normalized))


class CartPoleDynamicSystem:
    def __init__(self, runtime, dynamic_config):
        self.step = runtime.step
        self.initial_state = dynamic_config["initial_state"]
        params = dynamic_config["params"]
        self.mass_cart = float(params["mass_cart_kg"])
        self.mass_pendulum = float(params["mass_pendulum_kg"])
        self.length = float(params["l"])
        self.gravity = float(params["g"])
        self.cart_friction = float(params["cart_friction_coef"])
        self.pivot_friction = float(params["pivot_friction_coef"])
        self.max_torque = float(params["max_torque_nm"])
        self.gear_ratio = float(params["gear_ratio"])
        self.pinion_radius = float(params["pinion_radius_m"])

        _require_positive("mass_cart_kg", self.mass_cart)
        _require_positive("mass_pendulum_kg", self.mass_pendulum)
        _require_positive("l", self.length)
        _require_positive("g", self.gravity)
        _require_positive("max_torque_nm", self.max_torque)
        _require_positive("gear_ratio", self.gear_ratio)
        _require_positive("pinion_radius_m", self.pinion_radius)

        normalization = dynamic_config["state_normalization"]
        self.output_limits = normalization["output_limits"]
        self.state_ranges = normalization["ranges"]
        self.force_scale = self.max_torque * self.gear_ratio / self.pinion_radius

    def reset_episode(self, step_idx):
        cart_position = float(self.initial_state["cart_position"])
        cart_velocity = float(self.initial_state["cart_velocity"])
        pendulum_angle = self.reference_system_correction(float(self.initial_state["pendulum_angle"]))
        pendulum_velocity = float(self.initial_state["pendulum_velocity"])
        self.write_state(step_idx, 0.0, cart_position, cart_velocity, pendulum_angle, pendulum_velocity, 0.0)

    def integrate_step(self, step_idx, u_total, dt):
        previous_idx = step_idx - 1
        t_sec = self.step.t_sec[previous_idx] + dt
        previous_state = np.asarray(
            [
                self.step.cart_position_raw[previous_idx],
                self.step.cart_velocity_raw[previous_idx],
                self.step.pendulum_angle_raw[previous_idx],
                self.step.pendulum_velocity_raw[previous_idx],
            ],
            dtype=np.float64,
        )
        cart_force = self.apply_u_total(u_total)
        next_state = odeint(self.dynamics, previous_state, [0.0, dt], args=(cart_force,))[-1]
        next_state[2] = self.reference_system_correction(next_state[2])

        self.write_state(
            step_idx,
            t_sec,
            float(next_state[0]),
            float(next_state[1]),
            float(next_state[2]),
            float(next_state[3]),
            cart_force,
        )

    def apply_u_total(self, u_total):
        return float(u_total) * self.force_scale

    def dynamics(self, state_vector, time_t, cart_force):
        del time_t
        cart_position, cart_velocity, pendulum_angle, pendulum_velocity = state_vector
        del cart_position

        sin_angle = np.sin(pendulum_angle)
        cos_angle = np.cos(pendulum_angle)
        net_force = cart_force - self.cart_friction * cart_velocity
        common_denom = self.mass_cart + self.mass_pendulum * sin_angle**2

        d_cart_position = cart_velocity
        d_cart_velocity = (
            net_force
            + self.mass_pendulum * self.length * pendulum_velocity**2 * sin_angle
            - self.mass_pendulum * self.gravity * sin_angle * cos_angle
        ) / common_denom
        d_pendulum_angle = pendulum_velocity
        angular_numerator = (
            -net_force * cos_angle
            + (self.mass_cart + self.mass_pendulum) * self.gravity * sin_angle
            - self.mass_pendulum * self.length * pendulum_velocity**2 * sin_angle * cos_angle
            - self.pivot_friction * pendulum_velocity
        )
        d_pendulum_velocity = angular_numerator / (self.length * common_denom)

        return [d_cart_position, d_cart_velocity, d_pendulum_angle, d_pendulum_velocity]

    def reference_system_correction(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def write_state(self, step_idx, t_sec, cart_position, cart_velocity, pendulum_angle, pendulum_velocity, cart_force):
        self.step.t_sec[step_idx] = t_sec
        self.step.cart_position_raw[step_idx] = cart_position
        self.step.cart_velocity_raw[step_idx] = cart_velocity
        self.step.pendulum_angle_raw[step_idx] = pendulum_angle
        self.step.pendulum_velocity_raw[step_idx] = pendulum_velocity
        self.step.cart_position_norm[step_idx] = _normalize_clipped(
            cart_position, self.state_ranges["cart_position"], self.output_limits
        )
        self.step.cart_velocity_norm[step_idx] = _normalize_clipped(
            cart_velocity, self.state_ranges["cart_velocity"], self.output_limits
        )
        self.step.pendulum_angle_norm[step_idx] = _normalize_clipped(
            pendulum_angle, self.state_ranges["pendulum_angle"], self.output_limits
        )
        self.step.pendulum_velocity_norm[step_idx] = _normalize_clipped(
            pendulum_velocity, self.state_ranges["pendulum_velocity"], self.output_limits
        )
        self.step.cart_force[step_idx] = cart_force
