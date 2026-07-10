import numpy as np


def _clip(value, lower, upper):
    return min(upper, max(lower, value))


def _clip_01(value):
    if not np.isfinite(value):
        raise ValueError("metric value must be finite before clipping to [0, 1]")
    return _clip(float(value), 0.0, 1.0)


def _resolve_source(step, interval, source_name):
    if hasattr(step, source_name):
        return getattr(step, source_name), "step"
    return getattr(interval, source_name), "interval"


def _source_values(source_array, source_scope, interval_idx, start_step_idx, end_step_idx):
    if source_scope == "step":
        return source_array[start_step_idx : end_step_idx + 1]
    return source_array[interval_idx : interval_idx + 1]


def _finite_array(values, label):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} values must be finite")


def _finite_scalar(value, label):
    if not np.isfinite(value):
        raise ValueError(f"{label} value must be finite")


def _epsilon(config):
    if "epsilon" in config:
        epsilon = float(config["epsilon"])
        if epsilon < 0.0:
            raise ValueError("metric epsilon must be non-negative")
        return epsilon
    return 1.0e-12


def _transform_values(values, transform_config):
    method = transform_config["method"]
    if method == "none":
        return values
    if method == "identity":
        return values
    if method == "abs":
        return np.abs(values)
    if method == "square":
        return values * values
    if method == "signed_square":
        return np.sign(values) * values * values
    if method == "sqrt_abs":
        return np.sqrt(np.abs(values))
    if method == "log1p_abs":
        return np.log1p(np.abs(values))
    if method == "asinh":
        scale = float(transform_config["scale"])
        return np.arcsinh(values / scale)
    if method == "tanh":
        scale = float(transform_config["scale"])
        return np.tanh(values / scale)
    if method == "clip":
        lower = float(transform_config["min"])
        upper = float(transform_config["max"])
        return np.clip(values, lower, upper)
    raise ValueError(f"Unsupported metric transform: {method}")


def _transform_scalar(value, transform_config):
    values = np.asarray([value], dtype=np.float64)
    transformed = _transform_values(values, transform_config)
    return float(transformed[0])


def _interval_duration_sec(start_step_idx, end_step_idx, dt_sec):
    n_steps = end_step_idx - start_step_idx + 1
    return max(float(dt_sec), float(n_steps) * float(dt_sec))


def _reduce_values(values, reduce_config, interval_duration_sec, dt_sec, denominator_values):
    method = reduce_config["method"]
    epsilon = _epsilon(reduce_config)

    first = float(values[0])
    last = float(values[-1])

    if method == "none":
        if len(values) != 1:
            raise ValueError("interval_reduce none requires exactly one source value")
        return first
    if method == "value":
        if len(values) != 1:
            raise ValueError("interval_reduce value requires exactly one source value")
        return first
    if method == "first":
        return first
    if method == "last":
        return last
    if method == "last_minus_first":
        return last - first
    if method == "abs_last_minus_first":
        return abs(last - first)
    if method == "last_abs_minus_first_abs":
        return abs(last) - abs(first)
    if method == "abs_last_abs_minus_first_abs":
        return abs(abs(last) - abs(first))
    if method == "growth_abs_last_first":
        return max(0.0, abs(last) - abs(first))
    if method == "ratio_last_first":
        return last / (first + epsilon)
    if method == "ratio_abs_last_first_abs":
        return abs(last) / (abs(first) + epsilon)
    if method == "relative_change_last_first":
        return (last - first) / (abs(first) + epsilon)
    if method == "relative_magnitude_change_last_first":
        return (abs(last) - abs(first)) / (abs(first) + epsilon)
    if method == "abs_relative_change_last_first":
        return abs(last - first) / (abs(first) + epsilon)
    if method == "rate_last_minus_first":
        return (last - first) / interval_duration_sec
    if method == "abs_rate_last_minus_first":
        return abs(last - first) / interval_duration_sec
    if method == "rate_abs_last_minus_first_abs":
        return (abs(last) - abs(first)) / interval_duration_sec
    if method == "growth_rate_abs_last_first":
        return max(0.0, abs(last) - abs(first)) / interval_duration_sec
    if method == "sum":
        return float(np.sum(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    if method == "min":
        return float(np.min(values))
    if method == "max":
        return float(np.max(values))
    if method == "p25":
        return float(np.percentile(values, 25))
    if method == "p75":
        return float(np.percentile(values, 75))
    if method == "iqr":
        return float(np.percentile(values, 75) - np.percentile(values, 25))
    if method == "std":
        return float(np.std(values))
    if method == "var":
        return float(np.var(values))
    if method == "rms":
        return float(np.sqrt(np.mean(values * values)))
    if method == "mean_abs":
        return float(np.mean(np.abs(values)))
    if method == "mean_square":
        return float(np.mean(values * values))
    if method == "max_abs":
        return float(np.max(np.abs(values)))
    if method == "mean_step_delta":
        return float(np.mean(np.diff(values)))
    if method == "mean_abs_step_delta":
        return float(np.mean(np.abs(np.diff(values))))
    if method == "rms_step_delta":
        deltas = np.diff(values)
        return float(np.sqrt(np.mean(deltas * deltas)))
    if method == "mean_step_rate":
        return float(np.mean(np.diff(values) / dt_sec))
    if method == "mean_abs_step_rate":
        return float(np.mean(np.abs(np.diff(values) / dt_sec)))
    if method == "rms_step_rate":
        rates = np.diff(values) / dt_sec
        return float(np.sqrt(np.mean(rates * rates)))
    if method == "ratio_mean_abs":
        if denominator_values is None:
            raise ValueError("ratio_mean_abs requires denominator_source")
        numerator = float(np.mean(np.abs(values)))
        denominator = float(np.mean(np.abs(denominator_values)))
        return numerator / (denominator + epsilon)

    raise ValueError(f"Unsupported interval_reduce method: {method}")


def _write_audit_stats(interval, interval_idx, values, audit_stats):
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    iqr = p75 - p25
    for stat_name in audit_stats:
        target_name = audit_stats[stat_name]
        if stat_name == "p25":
            getattr(interval, target_name)[interval_idx] = p25
        elif stat_name == "p75":
            getattr(interval, target_name)[interval_idx] = p75
        elif stat_name == "iqr":
            getattr(interval, target_name)[interval_idx] = iqr
        else:
            raise ValueError(f"Unsupported interval audit stat: {stat_name}")


def _scale_value(value, scaler_config):
    method = scaler_config["method"]

    if method == "none":
        return float(value)
    if method == "clip_01":
        return _clip_01(value)
    if method == "signed_to_01":
        return _clip_01(0.5 * (float(value) + 1.0))
    if method == "minmax":
        value_min = float(scaler_config["range"][0])
        value_max = float(scaler_config["range"][1])
        if value_max <= value_min:
            raise ValueError("minmax scaler range max must be greater than min")
        scaled = (float(value) - value_min) / (value_max - value_min)
        if bool(scaler_config["clip"]):
            scaled = _clip_01(scaled)
        return float(scaled)
    if method == "bound":
        bound = float(scaler_config["bound"])
        if bound <= 0.0:
            raise ValueError("bound scaler bound must be positive")
        mode = scaler_config["mode"]
        if mode == "positive":
            scaled = float(value) / bound
        elif mode == "magnitude" or mode == "abs":
            scaled = abs(float(value)) / bound
        elif mode == "signed_to_01":
            scaled = 0.5 * (float(value) / bound + 1.0)
        else:
            raise ValueError(f"Unsupported bound scaler mode: {mode}")
        if bool(scaler_config["clip"]):
            scaled = _clip_01(scaled)
        return float(scaled)
    if method == "zscore":
        mean = float(scaler_config["mean"])
        std = float(scaler_config["std"])
        z_limit = float(scaler_config["z_limit"])
        if std <= 0.0:
            raise ValueError("zscore scaler std must be positive")
        if z_limit <= 0.0:
            raise ValueError("zscore scaler z_limit must be positive")
        z_value = (float(value) - mean) / std
        mode = scaler_config["mode"]
        if mode == "magnitude":
            scaled = abs(z_value) / z_limit
        elif mode == "positive":
            scaled = z_value / z_limit
        else:
            raise ValueError(f"Unsupported zscore scaler mode: {mode}")
        if bool(scaler_config["clip"]):
            scaled = _clip_01(scaled)
        return float(scaled)
    if method == "robust":
        median = float(scaler_config["median"])
        iqr = float(scaler_config["iqr"])
        z_limit = float(scaler_config["z_limit"])
        epsilon = _epsilon(scaler_config)
        robust_scale = max(iqr / 1.349, epsilon)
        if z_limit <= 0.0:
            raise ValueError("robust scaler z_limit must be positive")
        robust_z = (float(value) - median) / robust_scale
        scaled = abs(robust_z) / z_limit
        if bool(scaler_config["clip"]):
            scaled = _clip_01(scaled)
        return float(scaled)
    if method == "piecewise":
        raise NotImplementedError("piecewise metric scaler is reserved for a future refactor")

    raise ValueError(f"Unsupported metric scaler: {method}")


class MetricProcessor:
    def __init__(self, runtime, metrics_config, dt_sec):
        self.step = runtime.step
        self.interval = runtime.interval
        self.interval_metrics = metrics_config["interval_metrics"]
        self.output_limits = metrics_config["output_limits"]
        self.dt_sec = float(dt_sec)

        if float(self.output_limits[0]) != 0.0 or float(self.output_limits[1]) != 1.0:
            raise ValueError("Metric output_limits must be [0.0, 1.0]")
        if self.dt_sec <= 0.0:
            raise ValueError("MetricProcessor dt_sec must be positive")

    def reset_episode(self):
        pass

    def aggregate_interval(self, interval_idx, start_step_idx, end_step_idx):
        if end_step_idx < start_step_idx:
            raise ValueError("end_step_idx must be greater than or equal to start_step_idx")

        interval_duration_sec = _interval_duration_sec(start_step_idx, end_step_idx, self.dt_sec)

        for metric_config in self.interval_metrics:
            source_array, source_scope = _resolve_source(self.step, self.interval, metric_config["source"])
            source_values = _source_values(source_array, source_scope, interval_idx, start_step_idx, end_step_idx)
            _finite_array(source_values, metric_config["source"])

            sample_transform_config = metric_config["sample_transform"]
            transformed_values = _transform_values(source_values, sample_transform_config)
            _finite_array(transformed_values, metric_config["source"])
            if "audit_target" in sample_transform_config:
                audit_array = getattr(self.step, sample_transform_config["audit_target"])
                audit_array[start_step_idx : end_step_idx + 1] = transformed_values

            denominator_values = None
            interval_reduce_config = metric_config["interval_reduce"]
            if "denominator_source" in interval_reduce_config:
                denominator_array, denominator_scope = _resolve_source(
                    self.step,
                    self.interval,
                    interval_reduce_config["denominator_source"],
                )
                if denominator_scope != source_scope:
                    raise ValueError("metric denominator_source must have the same scope as source")
                denominator_values = _source_values(
                    denominator_array,
                    denominator_scope,
                    interval_idx,
                    start_step_idx,
                    end_step_idx,
                )
                _finite_array(denominator_values, interval_reduce_config["denominator_source"])

            reduced_value = _reduce_values(
                transformed_values,
                interval_reduce_config,
                interval_duration_sec,
                self.dt_sec,
                denominator_values,
            )
            _finite_scalar(reduced_value, metric_config["source"])

            if "audit_target" in interval_reduce_config:
                getattr(self.interval, interval_reduce_config["audit_target"])[interval_idx] = reduced_value
            if "audit_stats" in interval_reduce_config:
                _write_audit_stats(
                    self.interval,
                    interval_idx,
                    transformed_values,
                    interval_reduce_config["audit_stats"],
                )

            post_transform_config = metric_config["post_transform"]
            raw_value = _transform_scalar(reduced_value, post_transform_config)
            _finite_scalar(raw_value, metric_config["raw_target"])
            if "audit_target" in post_transform_config:
                getattr(self.interval, post_transform_config["audit_target"])[interval_idx] = raw_value

            getattr(self.interval, metric_config["raw_target"])[interval_idx] = raw_value
            target_value = _scale_value(raw_value, metric_config["scaler"])
            _finite_scalar(target_value, metric_config["target"])
            getattr(self.interval, metric_config["target"])[interval_idx] = target_value
