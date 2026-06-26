from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntervalDiagnosticSpec:
    target_array: np.ndarray
    method: str


@dataclass(frozen=True)
class IntervalMetricSpec:
    source_array: np.ndarray
    target_array: np.ndarray
    step_transform: str
    aggregation: str
    bound: float
    diagnostics: tuple[IntervalDiagnosticSpec, ...]


def _clip_01(value):
    if not np.isfinite(value):
        raise ValueError("metric value must be finite before clipping to [0, 1]")
    return min(1.0, max(0.0, value))


def _step_values(values, step_transform):
    if step_transform == "square":
        return values * values
    if step_transform == "abs":
        return np.abs(values)
    if step_transform == "identity":
        return values
    raise ValueError(f"Unsupported metric step_transform: {step_transform}")


def _aggregate_values(values, aggregation):
    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "max":
        return float(np.max(values))
    if aggregation == "rms":
        return float(np.sqrt(np.mean(values * values)))
    raise ValueError(f"Unsupported metric aggregation: {aggregation}")


def _diagnostic_value(source_values, method):
    if method == "rms":
        return float(np.sqrt(np.mean(source_values * source_values)))
    if method == "mean_abs":
        return float(np.mean(np.abs(source_values)))
    if method == "mean_square":
        return float(np.mean(source_values * source_values))
    raise ValueError(f"Unsupported metric diagnostic method: {method}")


class MetricProcessor:
    def __init__(self, runtime, metrics_config):
        self.step = runtime.step
        self.interval = runtime.interval
        self.output_limits = metrics_config["aggregation_output_limits"]
        if float(self.output_limits[0]) != 0.0 or float(self.output_limits[1]) != 1.0:
            raise ValueError("Metric aggregation_output_limits must be [0.0, 1.0]")

        self.metric_specs = []
        for metric_config in metrics_config["interval_metrics"]:
            bound = float(metric_config["bound"])
            if bound <= 0.0:
                raise ValueError("metric bound must be positive")

            diagnostic_configs = []
            if "diagnostics" in metric_config:
                diagnostic_configs = metric_config["diagnostics"]

            diagnostic_specs = []
            for diagnostic_config in diagnostic_configs:
                diagnostic_specs.append(
                    IntervalDiagnosticSpec(
                        target_array=getattr(self.interval, diagnostic_config["target"]),
                        method=diagnostic_config["method"],
                    )
                )

            self.metric_specs.append(
                IntervalMetricSpec(
                    source_array=getattr(self.step, metric_config["source"]),
                    target_array=getattr(self.interval, metric_config["target"]),
                    step_transform=metric_config["step_transform"],
                    aggregation=metric_config["aggregation"],
                    bound=bound,
                    diagnostics=tuple(diagnostic_specs),
                )
            )

    def reset_episode(self):
        pass

    def aggregate_interval(self, interval_idx, start_step_idx, end_step_idx):
        if end_step_idx < start_step_idx:
            raise ValueError("end_step_idx must be greater than or equal to start_step_idx")
        for metric_spec in self.metric_specs:
            source_values = metric_spec.source_array[start_step_idx : end_step_idx + 1]
            if not np.all(np.isfinite(source_values)):
                raise ValueError("metric source values must be finite")
            transformed_values = _step_values(source_values, metric_spec.step_transform)
            raw_cost = _aggregate_values(transformed_values, metric_spec.aggregation)
            metric_spec.target_array[interval_idx] = _clip_01(raw_cost / metric_spec.bound)

            for diagnostic_spec in metric_spec.diagnostics:
                diagnostic_spec.target_array[interval_idx] = _diagnostic_value(
                    source_values,
                    diagnostic_spec.method,
                )
