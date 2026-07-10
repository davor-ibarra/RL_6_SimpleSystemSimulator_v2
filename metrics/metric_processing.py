from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorChangeSpec:
    source_array: np.ndarray
    ratio_array: np.ndarray
    rate_array: np.ndarray
    growth_array: np.ndarray


@dataclass(frozen=True)
class BoundSpec:
    is_range: bool
    lower: float
    upper: float


@dataclass(frozen=True)
class IntervalDiagnosticSpec:
    raw_target_array: np.ndarray
    target_array: np.ndarray
    method: str
    bound: BoundSpec


@dataclass(frozen=True)
class IntervalMetricSpec:
    source_array: np.ndarray
    source_scope: str
    denominator_array: object
    denominator_scope: str
    raw_target_array: np.ndarray
    target_array: np.ndarray
    step_transform: str
    aggregation: str
    bound: BoundSpec
    ratio_epsilon: float
    diagnostics: tuple[IntervalDiagnosticSpec, ...]


def _resolve_source(step, interval, source_name):
    if hasattr(step, source_name):
        return getattr(step, source_name), "step"
    return getattr(interval, source_name), "interval"


def _error_change_signal_parts(signal_name):
    prefix = "ratio_error_"
    if not signal_name.startswith(prefix):
        return None
    stem = signal_name[len(prefix) :]
    for domain in ("raw", "norm"):
        suffix = f"_{domain}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], domain
    return None


def _clip_01(value):
    if not np.isfinite(value):
        raise ValueError("metric value must be finite before clipping to [0, 1]")
    return min(1.0, max(0.0, value))


def _parse_bound(bound_config, label):
    if isinstance(bound_config, list):
        if len(bound_config) != 2:
            raise ValueError(f"{label} bound range must contain exactly two values")
        lower = float(bound_config[0])
        upper = float(bound_config[1])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"{label} bound range values must be finite")
        if upper <= lower:
            raise ValueError(f"{label} bound range max must be greater than min")
        return BoundSpec(is_range=True, lower=lower, upper=upper)

    upper = float(bound_config)
    if not np.isfinite(upper):
        raise ValueError(f"{label} bound must be finite")
    if upper <= 0.0:
        raise ValueError(f"{label} bound must be positive")
    return BoundSpec(is_range=False, lower=0.0, upper=upper)


def _normalize_with_bound(value, bound):
    if bound.is_range:
        return _clip_01((value - bound.lower) / (bound.upper - bound.lower))
    return _clip_01(value / bound.upper)


def _step_values(values, step_transform):
    if step_transform == "none":
        return values
    if step_transform == "square":
        return values * values
    if step_transform == "abs":
        return np.abs(values)
    if step_transform == "identity":
        return values
    raise ValueError(f"Unsupported metric step_transform: {step_transform}")


def _aggregate_values(values, aggregation):
    if aggregation == "none":
        if len(values) != 1:
            raise ValueError("metric aggregation none requires exactly one source value")
        return float(values[0])
    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "max":
        return float(np.max(values))
    if aggregation == "sum":
        return float(np.sum(values))
    if aggregation == "rms":
        return float(np.sqrt(np.mean(values * values)))
    raise ValueError(f"Unsupported metric aggregation: {aggregation}")


def _aggregate_mean_abs_ratio(numerator_values, denominator_values, epsilon):
    numerator = float(np.mean(np.abs(numerator_values)))
    denominator = float(np.mean(np.abs(denominator_values)))
    return numerator / (denominator + epsilon)


def _diagnostic_value(source_values, method):
    if method == "rms":
        return float(np.sqrt(np.mean(source_values * source_values)))
    if method == "mean_abs":
        return float(np.mean(np.abs(source_values)))
    if method == "mean_square":
        return float(np.mean(source_values * source_values))
    raise ValueError(f"Unsupported metric diagnostic method: {method}")


class MetricProcessor:
    def __init__(self, runtime, metrics_config, decision_interval_sec):
        self.step = runtime.step
        self.interval = runtime.interval
        self.error_change_specs = self.build_error_change_specs(runtime)
        self.output_limits = metrics_config["aggregation_output_limits"]
        self.decision_interval_sec = float(decision_interval_sec)
        if float(self.output_limits[0]) != 0.0 or float(self.output_limits[1]) != 1.0:
            raise ValueError("Metric aggregation_output_limits must be [0.0, 1.0]")

        self.metric_specs = []
        for metric_config in metrics_config["interval_metrics"]:
            bound = _parse_bound(metric_config["bound"], "metric")

            diagnostic_configs = []
            if "diagnostics" in metric_config:
                diagnostic_configs = metric_config["diagnostics"]

            source_array, source_scope = _resolve_source(self.step, self.interval, metric_config["source"])
            denominator_array = None
            denominator_scope = ""
            ratio_epsilon = 0.0
            if "denominator_source" in metric_config:
                denominator_array, denominator_scope = _resolve_source(
                    self.step,
                    self.interval,
                    metric_config["denominator_source"],
                )
                if denominator_scope != source_scope:
                    raise ValueError("metric denominator_source must have the same scope as source")
                ratio_epsilon = float(metric_config["epsilon"])
                if not np.isfinite(ratio_epsilon):
                    raise ValueError("metric ratio epsilon must be finite")
                if ratio_epsilon < 0.0:
                    raise ValueError("metric ratio epsilon must be non-negative")

            if metric_config["aggregation"] == "none":
                if metric_config["step_transform"] != "none":
                    raise ValueError("metric aggregation none requires step_transform none")
                if source_scope != "interval":
                    raise ValueError("metric step_transform none is only supported for interval sources")
            if metric_config["aggregation"] == "mean_abs_ratio" and denominator_array is None:
                raise ValueError("metric aggregation mean_abs_ratio requires denominator_source")
            if metric_config["aggregation"] != "mean_abs_ratio" and denominator_array is not None:
                raise ValueError("metric denominator_source is only supported for mean_abs_ratio")

            diagnostic_specs = []
            for diagnostic_config in diagnostic_configs:
                diagnostic_bound = _parse_bound(diagnostic_config["bound"], "metric diagnostic")

                diagnostic_specs.append(
                    IntervalDiagnosticSpec(
                        raw_target_array=getattr(self.interval, diagnostic_config["raw_target"]),
                        target_array=getattr(self.interval, diagnostic_config["target"]),
                        method=diagnostic_config["method"],
                        bound=diagnostic_bound,
                    )
                )

            self.metric_specs.append(
                IntervalMetricSpec(
                    source_array=source_array,
                    source_scope=source_scope,
                    denominator_array=denominator_array,
                    denominator_scope=denominator_scope,
                    raw_target_array=getattr(self.interval, metric_config["raw_target"]),
                    target_array=getattr(self.interval, metric_config["target"]),
                    step_transform=metric_config["step_transform"],
                    aggregation=metric_config["aggregation"],
                    bound=bound,
                    ratio_epsilon=ratio_epsilon,
                    diagnostics=tuple(diagnostic_specs),
                )
            )

    def build_error_change_specs(self, runtime):
        specs = []
        seen = set()
        for signal_name in runtime.interval_signal_names:
            parts = _error_change_signal_parts(signal_name)
            if parts is None:
                continue
            controller_id, domain = parts
            key = (controller_id, domain)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                ErrorChangeSpec(
                    source_array=getattr(self.step, f"error_{controller_id}_{domain}"),
                    ratio_array=getattr(self.interval, f"ratio_error_{controller_id}_{domain}"),
                    rate_array=getattr(self.interval, f"rate_error_{controller_id}_{domain}"),
                    growth_array=getattr(self.interval, f"growth_error_{controller_id}_{domain}"),
                )
            )
        return tuple(specs)

    def reset_episode(self):
        pass

    def calculate_metrics(self, interval_idx, start_step_idx, end_step_idx):
        for spec in self.error_change_specs:
            self.write_error_change_metric(
                interval_idx,
                start_step_idx,
                end_step_idx,
                spec.source_array,
                spec.ratio_array,
                spec.rate_array,
                spec.growth_array,
            )

    def write_error_change_metric(
        self,
        interval_idx,
        start_step_idx,
        end_step_idx,
        source_array,
        ratio_array,
        rate_array,
        growth_array,
    ):
        error_floor = 1.0e-6

        start_error = abs(float(source_array[start_step_idx]))
        end_error = abs(float(source_array[end_step_idx]))

        ratio_array[interval_idx] = end_error / max(start_error, error_floor)
        growth_array[interval_idx] = max(0.0, end_error - start_error)
        rate_array[interval_idx] = (end_error - start_error) / self.decision_interval_sec

    def aggregate_interval(self, interval_idx, start_step_idx, end_step_idx):
        if end_step_idx < start_step_idx:
            raise ValueError("end_step_idx must be greater than or equal to start_step_idx")

        for metric_spec in self.metric_specs:
            if metric_spec.source_scope == "step":
                source_values = metric_spec.source_array[start_step_idx : end_step_idx + 1]
            else:
                source_values = metric_spec.source_array[interval_idx : interval_idx + 1]
            if not np.all(np.isfinite(source_values)):
                raise ValueError("metric source values must be finite")

            if metric_spec.aggregation == "mean_abs_ratio":
                if metric_spec.denominator_scope == "step":
                    denominator_values = metric_spec.denominator_array[start_step_idx : end_step_idx + 1]
                else:
                    denominator_values = metric_spec.denominator_array[interval_idx : interval_idx + 1]
                if not np.all(np.isfinite(denominator_values)):
                    raise ValueError("metric denominator values must be finite")
                raw_cost = _aggregate_mean_abs_ratio(
                    source_values,
                    denominator_values,
                    metric_spec.ratio_epsilon,
                )
            else:
                transformed_values = _step_values(source_values, metric_spec.step_transform)
                raw_cost = _aggregate_values(transformed_values, metric_spec.aggregation)

            metric_spec.raw_target_array[interval_idx] = raw_cost
            metric_spec.target_array[interval_idx] = _normalize_with_bound(raw_cost, metric_spec.bound)

            for diagnostic_spec in metric_spec.diagnostics:
                diagnostic_raw_cost = _diagnostic_value(source_values, diagnostic_spec.method)
                diagnostic_spec.raw_target_array[interval_idx] = diagnostic_raw_cost
                diagnostic_spec.target_array[interval_idx] = _normalize_with_bound(
                    diagnostic_raw_cost,
                    diagnostic_spec.bound,
                )
