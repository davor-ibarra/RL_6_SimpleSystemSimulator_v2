from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator


HEATMAP_AGGREGATIONS = ("count", "sum", "mean", "min", "max")
HEATMAP_DIAGNOSTIC_NAMES = (
    "valid_observations",
    "nonfinite_observations",
    "below_range_observations",
    "above_range_observations",
)

RAW_STATE_NAMES = {
    "cart_position_raw": "cart_position",
    "cart_velocity_raw": "cart_velocity",
    "pendulum_angle_raw": "pendulum_angle",
    "pendulum_velocity_raw": "pendulum_velocity",
}

NORMALIZED_STATE_NAMES = {
    "cart_position_norm",
    "cart_velocity_norm",
    "pendulum_angle_norm",
    "pendulum_velocity_norm",
}


def _apply_optional_axis_limits(ax, plot_config):
    xmin = plot_config["xmin"]
    xmax = plot_config["xmax"]
    ymin = plot_config["ymin"]
    ymax = plot_config["ymax"]
    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)


def _apply_optional_tick_counts(ax, plot_config):
    num_xticks = plot_config["num_xticks"]
    num_yticks = plot_config["num_yticks"]
    if num_xticks is not None:
        ax.xaxis.set_major_locator(MaxNLocator(int(num_xticks)))
    if num_yticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(int(num_yticks)))


def _set_grid(ax, enabled, axis="both"):
    if bool(enabled):
        ax.grid(True, axis=axis, alpha=0.3)
    else:
        ax.grid(False, axis=axis)


def _validate_increasing_range(variable_name, value_range):
    lower = float(value_range[0])
    upper = float(value_range[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"Heatmap range for {variable_name} must be finite")
    if upper <= lower:
        raise ValueError(f"Heatmap range for {variable_name} must be increasing")
    return lower, upper


def _canonical_variable_range(variable_name, full_config):
    if variable_name == "t_sec":
        return _validate_increasing_range(
            variable_name,
            (0.0, full_config["simulation"]["episode_duration_sec"]),
        )

    if variable_name in RAW_STATE_NAMES:
        state_name = RAW_STATE_NAMES[variable_name]
        boundary_config = full_config["termination"]["boundary"]
        if state_name in boundary_config:
            return _validate_increasing_range(
                variable_name,
                boundary_config[state_name],
            )
        return _validate_increasing_range(
            variable_name,
            full_config["dynamic_system"]["state_normalization"]["ranges"][state_name],
        )

    if variable_name in NORMALIZED_STATE_NAMES:
        return _validate_increasing_range(
            variable_name,
            full_config["dynamic_system"]["state_normalization"]["output_limits"],
        )

    if variable_name == "u_total":
        return _validate_increasing_range(
            variable_name,
            full_config["control"]["global_actuator_limits"],
        )

    if variable_name == "cart_force":
        params = full_config["dynamic_system"]["params"]
        max_torque = float(params["max_torque_nm"])
        gear_ratio = float(params["gear_ratio"])
        pinion_radius = float(params["pinion_radius_m"])
        if not np.isfinite(max_torque) or max_torque <= 0.0:
            raise ValueError("max_torque_nm must be finite and positive to resolve cart_force range")
        if not np.isfinite(gear_ratio) or gear_ratio <= 0.0:
            raise ValueError("gear_ratio must be finite and positive to resolve cart_force range")
        if not np.isfinite(pinion_radius) or pinion_radius <= 0.0:
            raise ValueError("pinion_radius_m must be finite and positive to resolve cart_force range")
        force_scale = max_torque * gear_ratio / pinion_radius
        actuator_limits = full_config["control"]["global_actuator_limits"]
        force_limits = (
            float(actuator_limits[0]) * force_scale,
            float(actuator_limits[1]) * force_scale,
        )
        return _validate_increasing_range(variable_name, force_limits)

    raise ValueError(
        f"Heatmap range for {variable_name} cannot be resolved; declare both limits explicitly"
    )


def _resolve_axis_range(variable_name, lower, upper, full_config):
    if lower is None or upper is None:
        canonical_lower, canonical_upper = _canonical_variable_range(variable_name, full_config)
    else:
        canonical_lower = None
        canonical_upper = None

    resolved_lower = canonical_lower if lower is None else float(lower)
    resolved_upper = canonical_upper if upper is None else float(upper)
    return _validate_increasing_range(variable_name, (resolved_lower, resolved_upper))


def _validate_bins(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Heatmap bins must be an integer")
    bins = int(value)
    if bins <= 0:
        raise ValueError("Heatmap bins must be positive")
    return bins


def _new_heatmap_accumulator(aggregation, bins):
    accumulator = {
        "aggregation": aggregation,
        "counts": np.zeros((bins, bins), dtype=np.uint64),
        "included_episodes": 0,
        "diagnostics": {
            "valid_observations": 0,
            "nonfinite_observations": 0,
            "below_range_observations": 0,
            "above_range_observations": 0,
        },
    }
    if aggregation == "count":
        accumulator["values"] = None
    elif aggregation in ("sum", "mean"):
        accumulator["values"] = np.zeros((bins, bins), dtype=np.float64)
    elif aggregation == "min":
        accumulator["values"] = np.full((bins, bins), np.inf, dtype=np.float64)
    elif aggregation == "max":
        accumulator["values"] = np.full((bins, bins), -np.inf, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported heatmap aggregation: {aggregation}")
    return accumulator


class VisualizationManager:
    def __init__(self, output_dir, visualization_config, full_config, runtime):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_config = visualization_config
        self.full_config = full_config
        self.step = runtime.step
        self.enabled = bool(visualization_config["enabled"])
        self.summary_columns = self.build_summary_columns(
            full_config["data"]["summary"]
        )
        self.termination_reasons = frozenset(
            full_config["termination"]["reason_codes"]
        )

        self.heatmap_groups = {}
        self.heatmap_group_by_plot = {}
        self.heatmap_accumulator_by_plot = {}
        self.heatmap_plot_configs = {}
        self.heatmap_plot_order = []

        if self.enabled:
            self.initialize_plots()

    def initialize_plots(self):
        enabled_plot_names = set()
        enabled_output_filenames = set()
        for plot_config in self.visualization_config["plots"]:
            if not bool(plot_config["enabled"]):
                continue

            output_filename = plot_config["output_filename"]
            if output_filename in enabled_output_filenames:
                raise ValueError(
                    f"Duplicate enabled visualization output filename: {output_filename}"
                )
            enabled_output_filenames.add(output_filename)

            plot_type = plot_config["type"]
            if plot_type == "heatmap":
                plot_name = plot_config["name"]
                if plot_name in enabled_plot_names:
                    raise ValueError(
                        f"Duplicate enabled visualization plot name: {plot_name}"
                    )
                enabled_plot_names.add(plot_name)
                self.initialize_heatmap(plot_config)
            elif plot_type == "line":
                self.validate_line_plot(plot_config)
            elif plot_type == "multi_line":
                self.validate_multi_line_plot(plot_config)
            elif plot_type == "bar":
                self.validate_bar_plot(plot_config)
            else:
                raise ValueError(f"Unsupported visualization plot type: {plot_type}")

    @staticmethod
    def build_summary_columns(summary_config):
        columns = set(summary_config["episode_fields"])
        columns.update(summary_config["derived_fields"])
        columns.update(summary_config["end_episode_fields"])
        for scope_name in ("step", "interval"):
            for stat_spec in summary_config[f"{scope_name}_stats"]:
                source_name = stat_spec["source"]
                for stat_name in stat_spec["stats"]:
                    columns.add(f"{scope_name}_{source_name}_{stat_name}")
        return frozenset(columns)

    def validate_summary_column(self, column_name, plot_type):
        if column_name not in self.summary_columns:
            raise KeyError(
                f"Unknown {plot_type} summary column: {column_name}"
            )

    def validate_line_plot(self, plot_config):
        if plot_config["source"] != "summary":
            raise ValueError("Line plots only support source: summary")
        self.validate_summary_column(plot_config["x_variable"], "line plot")
        self.validate_summary_column(plot_config["y_variable"], "line plot")
        self.validate_termination_filter(
            plot_config["config"]["filter_termination_reason"]
        )

    def validate_multi_line_plot(self, plot_config):
        if plot_config["source"] != "summary":
            raise ValueError("Multi-line plots only support source: summary")
        self.validate_summary_column(
            plot_config["x_variable"],
            "multi-line plot",
        )
        y_variables = plot_config["y_variables"]
        if not isinstance(y_variables, list) or len(y_variables) == 0:
            raise ValueError("Multi-line plots require at least one y variable")
        for y_config in y_variables:
            self.validate_summary_column(
                y_config["variable"],
                "multi-line plot",
            )
        self.validate_termination_filter(
            plot_config["config"]["filter_termination_reason"]
        )

    def validate_bar_plot(self, plot_config):
        if plot_config["source"] != "summary":
            raise ValueError("Bar plots only support source: summary")
        self.validate_summary_column(plot_config["variable"], "bar plot")
        group_size = plot_config["config"]["group_size"]
        if group_size is not None:
            if isinstance(group_size, (bool, np.bool_)) or not isinstance(
                group_size,
                (int, np.integer),
            ):
                raise ValueError("Bar group_size must be an integer or null")
            if int(group_size) <= 0:
                raise ValueError("Bar group_size must be positive")

    def initialize_heatmap(self, plot_config):
        if plot_config["source"] != "detailed":
            raise ValueError("Heatmaps only support source: detailed")

        cfg = plot_config["config"]
        aggregation = cfg["aggregation"]
        if aggregation not in HEATMAP_AGGREGATIONS:
            raise ValueError(f"Unsupported heatmap aggregation: {aggregation}")

        x_variable = plot_config["x_variable"]
        y_variable = plot_config["y_variable"]
        for variable_name in (x_variable, y_variable):
            if not hasattr(self.step, variable_name):
                raise KeyError(f"Unknown heatmap step signal: {variable_name}")
        if aggregation == "count":
            value_variable = None
            value_array = None
        else:
            value_variable = plot_config["value_variable"]
            if not hasattr(self.step, value_variable):
                raise KeyError(f"Unknown heatmap step signal: {value_variable}")
            value_array = getattr(self.step, value_variable)

        bins = _validate_bins(cfg["bins"])
        x_range = _resolve_axis_range(
            x_variable,
            cfg["xmin"],
            cfg["xmax"],
            self.full_config,
        )
        y_range = _resolve_axis_range(
            y_variable,
            cfg["ymin"],
            cfg["ymax"],
            self.full_config,
        )
        filter_key = self.validate_termination_filter(cfg["filter_termination_reason"])
        self.validate_color_range(cfg)

        if aggregation == "count":
            definition_key = (
                x_variable,
                y_variable,
                bins,
                x_range,
                y_range,
                aggregation,
            )
        else:
            definition_key = (
                x_variable,
                y_variable,
                value_variable,
                bins,
                x_range,
                y_range,
                aggregation,
            )
        if definition_key not in self.heatmap_groups:
            self.heatmap_groups[definition_key] = {
                "x_variable": x_variable,
                "y_variable": y_variable,
                "value_variable": value_variable,
                "x_array": getattr(self.step, x_variable),
                "y_array": getattr(self.step, y_variable),
                "value_array": value_array,
                "bins": bins,
                "x_range": x_range,
                "y_range": y_range,
                "x_edges": np.linspace(x_range[0], x_range[1], bins + 1),
                "y_edges": np.linspace(y_range[0], y_range[1], bins + 1),
                "aggregation": aggregation,
                "accumulators": {},
            }
        group = self.heatmap_groups[definition_key]

        if filter_key not in group["accumulators"]:
            group["accumulators"][filter_key] = _new_heatmap_accumulator(
                aggregation,
                bins,
            )
        accumulator = group["accumulators"][filter_key]

        plot_name = plot_config["name"]
        self.heatmap_group_by_plot[plot_name] = group
        self.heatmap_accumulator_by_plot[plot_name] = accumulator
        self.heatmap_plot_configs[plot_name] = plot_config
        self.heatmap_plot_order.append(plot_name)

    def validate_termination_filter(self, termination_filter):
        if termination_filter is None:
            return None
        if not isinstance(termination_filter, list):
            raise ValueError("Heatmap filter_termination_reason must be null or a list")
        if len(termination_filter) == 0:
            raise ValueError("Heatmap filter_termination_reason must not be empty")
        if len(set(termination_filter)) != len(termination_filter):
            raise ValueError("Heatmap filter_termination_reason contains duplicates")
        for reason_name in termination_filter:
            if reason_name not in self.termination_reasons:
                raise ValueError(f"Unknown heatmap termination reason: {reason_name}")
        return tuple(sorted(termination_filter))

    @staticmethod
    def validate_color_range(cfg):
        cmin = cfg["cmin"]
        cmax = cfg["cmax"]
        if cmin is not None and not np.isfinite(float(cmin)):
            raise ValueError("Heatmap cmin must be finite when declared")
        if cmax is not None and not np.isfinite(float(cmax)):
            raise ValueError("Heatmap cmax must be finite when declared")
        if cmin is not None and cmax is not None:
            if float(cmax) <= float(cmin):
                raise ValueError("Heatmap color range must be increasing")
        if bool(cfg["log_scale"]) and cmin is not None:
            if float(cmin) <= 0.0:
                raise ValueError("Heatmap cmin must be positive with log_scale")
        if bool(cfg["log_scale"]) and cmax is not None:
            if float(cmax) <= 0.0:
                raise ValueError("Heatmap cmax must be positive with log_scale")

    def accumulate_episode(self, n_steps, termination_reason):
        if len(self.heatmap_groups) == 0:
            return
        if isinstance(n_steps, (bool, np.bool_)) or not isinstance(
            n_steps, (int, np.integer)
        ):
            raise ValueError("Heatmap n_steps must be an integer")
        n_steps = int(n_steps)
        if n_steps <= 0:
            raise ValueError("Heatmap accumulation requires at least one step")
        if termination_reason not in self.termination_reasons:
            raise ValueError(f"Unknown episode termination reason: {termination_reason}")

        for group in self.heatmap_groups.values():
            matching_accumulators = []
            for filter_key, accumulator in group["accumulators"].items():
                if filter_key is None or termination_reason in filter_key:
                    matching_accumulators.append(accumulator)
            if len(matching_accumulators) == 0:
                continue

            contribution = self.build_episode_contribution(group, n_steps)
            for accumulator in matching_accumulators:
                self.apply_episode_contribution(accumulator, contribution)

    @staticmethod
    def build_episode_contribution(group, n_steps):
        if n_steps > len(group["x_array"]):
            raise ValueError("Heatmap n_steps exceeds x signal capacity")
        if n_steps > len(group["y_array"]):
            raise ValueError("Heatmap n_steps exceeds y signal capacity")
        x_values = np.asarray(group["x_array"][:n_steps], dtype=np.float64)
        y_values = np.asarray(group["y_array"][:n_steps], dtype=np.float64)
        aggregation = group["aggregation"]
        if aggregation == "count":
            value_values = None
            finite = np.isfinite(x_values) & np.isfinite(y_values)
        else:
            if n_steps > len(group["value_array"]):
                raise ValueError("Heatmap n_steps exceeds value signal capacity")
            value_values = np.asarray(
                group["value_array"][:n_steps],
                dtype=np.float64,
            )
            finite = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & np.isfinite(value_values)
            )

        x_range = group["x_range"]
        y_range = group["y_range"]
        below_range = finite & (
            (x_values < x_range[0]) | (y_values < y_range[0])
        )
        above_range = finite & (
            (x_values > x_range[1]) | (y_values > y_range[1])
        )
        valid = finite & ~below_range & ~above_range
        diagnostics = {
            "valid_observations": int(np.count_nonzero(valid)),
            "nonfinite_observations": int(n_steps - np.count_nonzero(finite)),
            "below_range_observations": int(np.count_nonzero(below_range)),
            "above_range_observations": int(np.count_nonzero(above_range)),
        }

        bins = group["bins"]
        if diagnostics["valid_observations"] == 0:
            return {
                "counts": None,
                "values": None,
                "diagnostics": diagnostics,
            }

        valid_x = x_values[valid]
        valid_y = y_values[valid]
        x_bins = np.searchsorted(group["x_edges"], valid_x, side="right") - 1
        y_bins = np.searchsorted(group["y_edges"], valid_y, side="right") - 1
        x_bins = np.minimum(x_bins, bins - 1)
        y_bins = np.minimum(y_bins, bins - 1)
        flat_indices = x_bins * bins + y_bins
        cell_count = bins * bins
        episode_counts = np.bincount(
            flat_indices,
            minlength=cell_count,
        ).astype(np.uint64, copy=False)
        episode_counts = episode_counts.reshape((bins, bins))

        if aggregation == "count":
            episode_values = None
        elif aggregation in ("sum", "mean"):
            episode_values = np.bincount(
                flat_indices,
                weights=value_values[valid],
                minlength=cell_count,
            ).reshape((bins, bins))
        elif aggregation == "min":
            episode_values = np.full(cell_count, np.inf, dtype=np.float64)
            np.minimum.at(episode_values, flat_indices, value_values[valid])
            episode_values = episode_values.reshape((bins, bins))
        elif aggregation == "max":
            episode_values = np.full(cell_count, -np.inf, dtype=np.float64)
            np.maximum.at(episode_values, flat_indices, value_values[valid])
            episode_values = episode_values.reshape((bins, bins))
        else:
            raise ValueError(f"Unsupported heatmap aggregation: {aggregation}")

        return {
            "counts": episode_counts,
            "values": episode_values,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def apply_episode_contribution(accumulator, contribution):
        accumulator["included_episodes"] += 1
        for diagnostic_name in HEATMAP_DIAGNOSTIC_NAMES:
            accumulator["diagnostics"][diagnostic_name] += contribution[
                "diagnostics"
            ][diagnostic_name]

        if contribution["counts"] is None:
            return
        accumulator["counts"] += contribution["counts"]
        aggregation = accumulator["aggregation"]
        if aggregation == "count":
            return
        if aggregation in ("sum", "mean"):
            accumulator["values"] += contribution["values"]
            return

        observed = contribution["counts"] > 0
        if aggregation == "min":
            np.minimum(
                accumulator["values"],
                contribution["values"],
                out=accumulator["values"],
                where=observed,
            )
            return
        if aggregation == "max":
            np.maximum(
                accumulator["values"],
                contribution["values"],
                out=accumulator["values"],
                where=observed,
            )
            return
        raise ValueError(f"Unsupported heatmap aggregation: {aggregation}")

    @staticmethod
    def accumulated_heatmap(accumulator):
        aggregation = accumulator["aggregation"]
        counts = accumulator["counts"]
        if aggregation == "count":
            return counts.astype(np.float64)

        observed = counts > 0
        heatmap = np.full(counts.shape, np.nan, dtype=np.float64)
        if aggregation == "sum":
            heatmap[observed] = accumulator["values"][observed]
        elif aggregation == "mean":
            heatmap[observed] = (
                accumulator["values"][observed] / counts[observed]
            )
        elif aggregation in ("min", "max"):
            heatmap[observed] = accumulator["values"][observed]
        else:
            raise ValueError(f"Unsupported heatmap aggregation: {aggregation}")
        return heatmap

    def render_post_run(self, summary_rows, latest_agent_state):
        del latest_agent_state
        if not self.enabled:
            return
        summary = pd.DataFrame(summary_rows)

        for plot_config in self.visualization_config["plots"]:
            if not bool(plot_config["enabled"]):
                continue
            if plot_config["type"] == "heatmap":
                self.render_heatmap(plot_config)
            elif len(summary) > 0:
                self.render_plot(plot_config, summary)

    def render_plot(self, plot_config, summary):
        plot_type = plot_config["type"]
        if plot_type == "line":
            self.render_line_plot(plot_config, summary)
        elif plot_type == "multi_line":
            self.render_multi_line_plot(plot_config, summary)
        elif plot_type == "bar":
            self.render_bar_plot(plot_config, summary)
        else:
            raise ValueError(f"Unsupported visualization plot type: {plot_type}")

    def render_line_plot(self, plot_config, summary):
        if plot_config["source"] != "summary":
            raise ValueError("Line plots only support source: summary")
        cfg = plot_config["config"]
        data = self.filter_summary(summary, cfg["filter_termination_reason"])
        if len(data) == 0:
            return

        fig, ax = plt.subplots(figsize=(float(cfg["figsize_w"]), float(cfg["figsize_h"])))
        marker_style = cfg["marker_style"]
        ax.plot(
            data[plot_config["x_variable"]],
            data[plot_config["y_variable"]],
            color=cfg["line_color"],
            linewidth=float(cfg["line_width"]),
            marker=marker_style,
            markersize=float(cfg["marker_size"]),
            markerfacecolor=cfg["marker_color"],
            label=plot_config["y_variable"],
        )
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"])
        ax.set_xlabel(cfg["xlabel"], fontsize=cfg["xlabel_fontsize"])
        ax.set_ylabel(cfg["ylabel"], fontsize=cfg["ylabel_fontsize"])
        ax.tick_params(axis="both", labelsize=cfg["tick_fontsize"])
        ax.tick_params(axis="x", rotation=cfg["xtick_rotation"])
        _apply_optional_axis_limits(ax, cfg)
        _apply_optional_tick_counts(ax, cfg)
        _set_grid(ax, cfg["grid_on"])
        if bool(cfg["show_legend"]):
            ax.legend(loc=cfg["legend_pos"])
        fig.tight_layout()
        fig.savefig(self.figures_dir / plot_config["output_filename"], dpi=150)
        plt.close(fig)

    def render_multi_line_plot(self, plot_config, summary):
        if plot_config["source"] != "summary":
            raise ValueError("Multi-line plots only support source: summary")
        cfg = plot_config["config"]
        data = self.filter_summary(summary, cfg["filter_termination_reason"])
        if len(data) == 0:
            return

        fig, ax = plt.subplots(figsize=(float(cfg["figsize_w"]), float(cfg["figsize_h"])))
        for y_config in plot_config["y_variables"]:
            ax.plot(
                data[plot_config["x_variable"]],
                data[y_config["variable"]],
                color=y_config["color"],
                linewidth=float(cfg["line_width"]),
                marker=cfg["marker_style"],
                markersize=float(cfg["marker_size"]),
                label=y_config["label"],
            )
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"])
        ax.set_xlabel(cfg["xlabel"], fontsize=cfg["xlabel_fontsize"])
        ax.set_ylabel(cfg["ylabel"], fontsize=cfg["ylabel_fontsize"])
        ax.tick_params(axis="both", labelsize=cfg["tick_fontsize"])
        ax.tick_params(axis="x", rotation=cfg["xtick_rotation"])
        _apply_optional_axis_limits(ax, cfg)
        _apply_optional_tick_counts(ax, cfg)
        _set_grid(ax, cfg["grid_on"])
        if bool(cfg["show_legend"]):
            ax.legend(loc=cfg["legend_pos"])
        fig.tight_layout()
        fig.savefig(self.figures_dir / plot_config["output_filename"], dpi=150)
        plt.close(fig)

    def render_bar_plot(self, plot_config, summary):
        if plot_config["source"] != "summary":
            raise ValueError("Bar plots only support source: summary")
        cfg = plot_config["config"]
        variable = plot_config["variable"]
        legend_values = [entry["value"] for entry in cfg["legend_var"]]
        legend_labels = [entry["label"] for entry in cfg["legend_var"]]
        legend_colors = [entry["color"] for entry in cfg["legend_var"]]

        fig, ax = plt.subplots(figsize=(float(cfg["figsize_w"]), float(cfg["figsize_h"])))
        group_size = cfg["group_size"]
        if group_size is None:
            counts = summary[variable].value_counts().reindex(legend_values, fill_value=0)
            counts.plot(kind="bar", ax=ax, color=legend_colors, width=float(cfg["bar_width"]))
            ax.set_xticklabels(legend_labels, rotation=cfg["xtick_rotation"])
        else:
            grouped = summary.copy()
            grouped["episode_group"] = (grouped["episode_id"] // int(group_size)) * int(group_size)
            counts = pd.crosstab(grouped["episode_group"], grouped[variable])
            counts = counts.reindex(columns=legend_values, fill_value=0)
            counts.plot(kind="bar", stacked=True, ax=ax, color=legend_colors, width=float(cfg["bar_width"]))

        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"])
        ax.set_xlabel(cfg["xlabel"], fontsize=cfg["xlabel_fontsize"])
        ax.set_ylabel(cfg["ylabel"], fontsize=cfg["ylabel_fontsize"])
        ax.tick_params(axis="both", labelsize=cfg["tick_fontsize"])
        ax.tick_params(axis="x", rotation=cfg["xtick_rotation"])
        _apply_optional_axis_limits(ax, cfg)
        _apply_optional_tick_counts(ax, cfg)
        _set_grid(ax, cfg["grid_on"], axis="y")
        if bool(cfg["show_legend"]):
            ax.legend(legend_labels, loc=cfg["legend_pos"], fontsize=cfg["legend_fontsize"])
        fig.tight_layout()
        fig.savefig(self.figures_dir / plot_config["output_filename"], dpi=150)
        plt.close(fig)

    def render_heatmap(self, plot_config):
        plot_name = plot_config["name"]
        group = self.heatmap_group_by_plot[plot_name]
        accumulator = self.heatmap_accumulator_by_plot[plot_name]
        if accumulator["diagnostics"]["valid_observations"] == 0:
            return

        cfg = plot_config["config"]
        heatmap = self.accumulated_heatmap(accumulator)
        if np.all(np.isnan(heatmap)):
            return

        fig, ax = plt.subplots(figsize=(float(cfg["figsize_w"]), float(cfg["figsize_h"])))
        plot_values = heatmap.T
        norm = None
        if bool(cfg["log_scale"]):
            plot_values = np.where(plot_values > 0.0, plot_values, np.nan)
            positive_values = plot_values[np.isfinite(plot_values)]
            if len(positive_values) == 0:
                plt.close(fig)
                return
            vmin = cfg["cmin"] if cfg["cmin"] is not None else float(np.min(positive_values))
            vmax = cfg["cmax"] if cfg["cmax"] is not None else float(np.max(positive_values))
            if float(vmax) <= float(vmin):
                if cfg["cmin"] is None and cfg["cmax"] is None:
                    center = float(positive_values[0])
                    vmin = center / 2.0
                    vmax = center * 2.0
                elif cfg["cmin"] is None:
                    vmin = float(vmax) / 2.0
                elif cfg["cmax"] is None:
                    vmax = float(vmin) * 2.0
                else:
                    plt.close(fig)
                    raise ValueError(
                        "Heatmap logarithmic color range must be increasing"
                    )
            norm = LogNorm(vmin=float(vmin), vmax=float(vmax))

        image = ax.pcolormesh(
            group["x_edges"],
            group["y_edges"],
            plot_values,
            cmap=cfg["cmap"],
            shading="auto",
            vmin=cfg["cmin"] if norm is None else None,
            vmax=cfg["cmax"] if norm is None else None,
            norm=norm,
        )
        ax.set_title(cfg["title"], fontsize=cfg["title_fontsize"])
        ax.set_xlabel(cfg["xlabel"], fontsize=cfg["xlabel_fontsize"])
        ax.set_ylabel(cfg["ylabel"], fontsize=cfg["ylabel_fontsize"])
        ax.tick_params(axis="both", labelsize=cfg["tick_fontsize"])
        ax.tick_params(axis="x", rotation=cfg["xtick_rotation"])
        ax.set_xlim(group["x_range"][0], group["x_range"][1])
        ax.set_ylim(group["y_range"][0], group["y_range"][1])
        _apply_optional_tick_counts(ax, cfg)
        _set_grid(ax, cfg["grid_on"])
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(cfg["clabel"], fontsize=cfg["clabel_fontsize"])
        colorbar.ax.tick_params(labelsize=cfg["tick_fontsize"])
        fig.tight_layout()
        fig.savefig(self.figures_dir / plot_config["output_filename"], dpi=150)
        plt.close(fig)

    @staticmethod
    def filter_summary(summary, filter_termination_reason):
        if filter_termination_reason is None:
            return summary
        return summary[summary["end_termination_reason"].isin(filter_termination_reason)]

    def metadata(self):
        heatmaps = []
        for plot_name in self.heatmap_plot_order:
            group = self.heatmap_group_by_plot[plot_name]
            accumulator = self.heatmap_accumulator_by_plot[plot_name]
            plot_config = self.heatmap_plot_configs[plot_name]
            termination_filter = plot_config["config"]["filter_termination_reason"]
            if termination_filter is None:
                serialized_filter = None
            else:
                serialized_filter = list(termination_filter)
            heatmap_metadata = {
                "name": plot_name,
                "x_variable": group["x_variable"],
                "y_variable": group["y_variable"],
                "aggregation": group["aggregation"],
                "bins": int(group["bins"]),
                "x_range": [
                    float(group["x_range"][0]),
                    float(group["x_range"][1]),
                ],
                "y_range": [
                    float(group["y_range"][0]),
                    float(group["y_range"][1]),
                ],
                "filter_termination_reason": serialized_filter,
                "included_episodes": int(accumulator["included_episodes"]),
                "diagnostics": {
                    diagnostic_name: int(
                        accumulator["diagnostics"][diagnostic_name]
                    )
                    for diagnostic_name in HEATMAP_DIAGNOSTIC_NAMES
                },
            }
            if group["aggregation"] != "count":
                heatmap_metadata["value_variable"] = group["value_variable"]
            heatmaps.append(heatmap_metadata)
        return {
            "definition_count": int(len(self.heatmap_groups)),
            "heatmaps": heatmaps,
        }
