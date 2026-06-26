import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator


def _json_load(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


class VisualizationManager:
    def __init__(self, output_dir, visualization_config):
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "dataset_chunks"
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_config = visualization_config
        self.summary_by_episode = {}

    def render_post_run(self, summary_rows, latest_agent_state):
        if len(summary_rows) == 0:
            return
        summary = pd.DataFrame(summary_rows)
        self.summary_by_episode = {
            int(row["episode_id"]): row for row in summary_rows
        }

        for plot_config in self.visualization_config["plots"]:
            if bool(plot_config["enabled"]):
                self.render_plot(plot_config, summary, latest_agent_state)

    def render_plot(self, plot_config, summary, latest_agent_state):
        del latest_agent_state
        plot_type = plot_config["type"]
        if plot_type == "line":
            self.render_line_plot(plot_config, summary)
        elif plot_type == "multi_line":
            self.render_multi_line_plot(plot_config, summary)
        elif plot_type == "bar":
            self.render_bar_plot(plot_config, summary)
        elif plot_type == "heatmap":
            self.render_heatmap(plot_config)
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
        if plot_config["source"] != "detailed":
            raise ValueError("Heatmaps only support source: detailed")
        cfg = plot_config["config"]
        x_values, y_values, value_values = self.load_detailed_series(
            plot_config["x_variable"],
            plot_config["y_variable"],
            plot_config["value_variable"],
            cfg["filter_termination_reason"],
        )
        if len(x_values) == 0:
            return

        x_range = self.resolve_range(x_values, cfg["xmin"], cfg["xmax"])
        y_range = self.resolve_range(y_values, cfg["ymin"], cfg["ymax"])
        heatmap, x_edges, y_edges = self.aggregate_heatmap(
            x_values,
            y_values,
            value_values,
            int(cfg["bins"]),
            x_range,
            y_range,
            cfg["aggregation"],
        )
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
            vmin = cfg["cmin"] if cfg["cmin"] is not None else float(np.nanmin(positive_values))
            vmax = cfg["cmax"] if cfg["cmax"] is not None else float(np.nanmax(positive_values))
            norm = LogNorm(vmin=max(float(vmin), 1e-12), vmax=float(vmax))

        image = ax.pcolormesh(
            x_edges,
            y_edges,
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
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        _apply_optional_tick_counts(ax, cfg)
        _set_grid(ax, cfg["grid_on"])
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(cfg["clabel"], fontsize=cfg["clabel_fontsize"])
        colorbar.ax.tick_params(labelsize=cfg["tick_fontsize"])
        fig.tight_layout()
        fig.savefig(self.figures_dir / plot_config["output_filename"], dpi=150)
        plt.close(fig)

    def filter_summary(self, summary, filter_termination_reason):
        if filter_termination_reason is None:
            return summary
        return summary[summary["end_termination_reason"].isin(filter_termination_reason)]

    def load_detailed_series(self, x_variable, y_variable, value_variable, filter_termination_reason):
        x_values = []
        y_values = []
        value_values = []
        for chunk_path in sorted(self.dataset_dir.glob("dataset_chunk_*.json")):
            chunk = _json_load(chunk_path)
            for episode in chunk["episodes"]:
                episode_id = int(episode["episode_id"])
                summary_row = self.summary_by_episode[episode_id]
                if filter_termination_reason is not None:
                    if summary_row["end_termination_reason"] not in filter_termination_reason:
                        continue
                step_data = episode["step_data"]
                n_steps = int(step_data["n_step"])
                x_values.extend(step_data[x_variable][:n_steps])
                y_values.extend(step_data[y_variable][:n_steps])
                value_values.extend(step_data[value_variable][:n_steps])
        return np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.float64), np.asarray(
            value_values, dtype=np.float64
        )

    def resolve_range(self, values, lower, upper):
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            raise ValueError("Cannot plot heatmap with no finite values")
        resolved_lower = float(lower) if lower is not None else float(np.min(finite_values))
        resolved_upper = float(upper) if upper is not None else float(np.max(finite_values))
        if resolved_upper <= resolved_lower:
            raise ValueError("Heatmap axis range must be increasing")
        return [resolved_lower, resolved_upper]

    def aggregate_heatmap(self, x_values, y_values, value_values, bins, x_range, y_range, aggregation):
        x_edges = np.linspace(x_range[0], x_range[1], bins + 1)
        y_edges = np.linspace(y_range[0], y_range[1], bins + 1)
        frame = pd.DataFrame({"x": x_values, "y": y_values, "value": value_values})
        frame = frame[
            (frame["x"] >= x_range[0])
            & (frame["x"] <= x_range[1])
            & (frame["y"] >= y_range[0])
            & (frame["y"] <= y_range[1])
        ]
        if len(frame) == 0:
            return np.full((bins, bins), np.nan), x_edges, y_edges

        frame["x_bin"] = pd.cut(frame["x"], x_edges, labels=False, include_lowest=True)
        frame["y_bin"] = pd.cut(frame["y"], y_edges, labels=False, include_lowest=True)
        frame = frame.dropna(subset=["x_bin", "y_bin"])
        heatmap = np.full((bins, bins), np.nan)

        if aggregation == "count":
            grouped = frame.groupby(["x_bin", "y_bin"], observed=True).size()
        elif aggregation == "mean":
            grouped = frame.groupby(["x_bin", "y_bin"], observed=True)["value"].mean()
        elif aggregation == "sum":
            grouped = frame.groupby(["x_bin", "y_bin"], observed=True)["value"].sum()
        elif aggregation == "min":
            grouped = frame.groupby(["x_bin", "y_bin"], observed=True)["value"].min()
        elif aggregation == "max":
            grouped = frame.groupby(["x_bin", "y_bin"], observed=True)["value"].max()
        else:
            raise ValueError(f"Unsupported heatmap aggregation: {aggregation}")

        for bin_indices, value in grouped.items():
            x_bin = int(bin_indices[0])
            y_bin = int(bin_indices[1])
            if x_bin < bins and y_bin < bins:
                heatmap[x_bin, y_bin] = float(value)
        if aggregation == "count":
            heatmap = np.nan_to_num(heatmap, nan=0.0)
        return heatmap, x_edges, y_edges
