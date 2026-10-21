import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from parquet_dataset_writer import ParquetDatasetWriter


SUMMARY_STATS = frozenset(
    ("mean", "std", "min", "p25", "p50", "p75", "max", "sum", "final")
)
SUMMARY_SECTIONS = frozenset(
    (
        "episode_fields",
        "derived_fields",
        "end_episode_fields",
        "step_stats",
        "interval_stats",
    )
)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _flatten_metadata(prefix, value, rows):
    if isinstance(value, dict):
        for nested_key in sorted(value):
            next_prefix = f"{prefix}.{nested_key}" if prefix else str(nested_key)
            _flatten_metadata(next_prefix, value[nested_key], rows)
    elif isinstance(value, list):
        rows.append(
            {
                "path": prefix,
                "value": json.dumps(value, ensure_ascii=False, default=_json_default),
            }
        )
    else:
        rows.append({"path": prefix, "value": value})


def _agent_tables(agent_state):
    q_rows = []
    visit_rows = []
    for agent in agent_state["agents"]:
        states = agent["states"]
        q_table = agent["q_table"]
        visits = agent["visits"]
        for state_idx, state in enumerate(states):
            q_row = {
                "agent_id": agent["agent_id"],
                "controller_id": agent["controller_id"],
                "gain_name": agent["gain_name"],
            }
            visit_row = {
                "agent_id": agent["agent_id"],
                "controller_id": agent["controller_id"],
                "gain_name": agent["gain_name"],
            }
            for field_name in state:
                q_row[field_name] = state[field_name]
                visit_row[field_name] = state[field_name]
            q_row["q_decrease"] = q_table[state_idx][0]
            q_row["q_maintain"] = q_table[state_idx][1]
            q_row["q_increase"] = q_table[state_idx][2]
            visit_row["visits_decrease"] = visits[state_idx][0]
            visit_row["visits_maintain"] = visits[state_idx][1]
            visit_row["visits_increase"] = visits[state_idx][2]
            q_rows.append(q_row)
            visit_rows.append(visit_row)
    return q_rows, visit_rows


def _series_stat(values, stat_name):
    if len(values) == 0:
        raise ValueError(f"Cannot calculate {stat_name} from an empty series")
    numeric_values = np.asarray(values, dtype=np.float64)
    if stat_name == "mean":
        return float(np.mean(numeric_values))
    if stat_name == "std":
        mean_value = float(np.mean(numeric_values))
        variance = float(np.mean((numeric_values - mean_value) ** 2))
        return math.sqrt(variance)
    if stat_name == "min":
        return float(np.min(numeric_values))
    if stat_name == "p25":
        return float(np.percentile(numeric_values, 25))
    if stat_name == "p50":
        return float(np.percentile(numeric_values, 50))
    if stat_name == "p75":
        return float(np.percentile(numeric_values, 75))
    if stat_name == "max":
        return float(np.max(numeric_values))
    if stat_name == "sum":
        return float(np.sum(numeric_values))
    if stat_name == "final":
        return float(numeric_values[-1])
    raise ValueError(f"Unsupported summary stat: {stat_name}")


def _error_text(error):
    return f"{type(error).__name__}: {error}"


class ResultHandler:
    def __init__(
        self,
        output_dir,
        data_config,
        termination_config,
        metric_report,
        runtime,
        visualization_manager,
    ):
        self.output_dir = Path(output_dir)
        self.agent_state_dir = self.output_dir / "agent_state"
        self.figures_dir = self.output_dir / "figures"
        self.agent_state_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.runtime = runtime
        self.step = runtime.step
        self.interval = runtime.interval
        self.episode = runtime.episode
        self.data_config = data_config
        self.summary_config = data_config["summary"]
        self.metric_report = metric_report
        self.visualization_manager = visualization_manager
        self.validate_summary_config()

        self.reason_codes = {}
        self.reason_names_by_code = {}
        for reason_name, reason_code in termination_config["reason_codes"].items():
            self.reason_codes[reason_name] = int(reason_code)
            self.reason_names_by_code[int(reason_code)] = reason_name

        run_id = metric_report.run_config_row["run_id"]
        self.parquet_writer = ParquetDatasetWriter(
            self.output_dir,
            data_config,
            runtime,
            metric_report,
            run_id,
        )
        self.summary_rows = []
        self.metadata = {}
        self.latest_agent_state = None
        self.persistence_status = "running"
        self.summary_status = "pending"
        self.visualization_status = (
            "pending" if visualization_manager is not None else "disabled"
        )
        self.summary_error = None
        self.visualization_error = None

    def validate_summary_config(self):
        if set(self.summary_config) != SUMMARY_SECTIONS:
            raise ValueError(
                "data.summary must contain exactly episode_fields, derived_fields, "
                "end_episode_fields, step_stats and interval_stats"
            )

        output_columns = set()
        for field_name in self.summary_config["episode_fields"]:
            if field_name not in ("episode_id", "n_step", "n_interval"):
                raise ValueError(f"Unsupported episode summary field: {field_name}")
            if field_name in output_columns:
                raise ValueError(f"Duplicate summary column: {field_name}")
            output_columns.add(field_name)

        for field_name in self.summary_config["derived_fields"]:
            if field_name not in (
                "end_termination_reason",
                "final_t_sec",
                "performance",
            ):
                raise ValueError(f"Unsupported derived summary field: {field_name}")
            if field_name in output_columns:
                raise ValueError(f"Duplicate summary column: {field_name}")
            output_columns.add(field_name)

        for field_name in self.summary_config["end_episode_fields"]:
            if not hasattr(self.episode, field_name):
                raise KeyError(f"Unknown end episode summary field: {field_name}")
            if field_name in output_columns:
                raise ValueError(f"Duplicate summary column: {field_name}")
            output_columns.add(field_name)

        for scope_name, scope in (("step", self.step), ("interval", self.interval)):
            for stat_spec in self.summary_config[f"{scope_name}_stats"]:
                source_name = stat_spec["source"]
                if not hasattr(scope, source_name):
                    raise KeyError(
                        f"Unknown {scope_name} summary signal: {source_name}"
                    )
                stat_names = stat_spec["stats"]
                if not isinstance(stat_names, list) or len(stat_names) == 0:
                    raise ValueError(
                        f"{scope_name} summary stats must be a non-empty list"
                    )
                for stat_name in stat_names:
                    if stat_name not in SUMMARY_STATS:
                        raise ValueError(f"Unsupported summary stat: {stat_name}")
                    output_name = f"{scope_name}_{source_name}_{stat_name}"
                    if output_name in output_columns:
                        raise ValueError(f"Duplicate summary column: {output_name}")
                    output_columns.add(output_name)

    def save_metadata(self, metadata):
        self.metadata = metadata
        self.metadata["run_id"] = self.metric_report.run_config_row["run_id"]
        self.metadata["config_id"] = self.metric_report.run_config_row["config_id"]
        self.metadata["seed"] = int(self.metric_report.config["experiment"]["seed"])
        self.update_metadata("running")

    def write_metadata(self):
        metadata_path = self.output_dir / "metadata.json"
        temporary_path = self.output_dir / "metadata.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        temporary_path.replace(metadata_path)

    def update_metadata(self, run_status):
        self.metadata["status"] = run_status
        self.metadata["persistence"] = self.parquet_writer.manifest(
            self.persistence_status
        )
        self.metadata["termination_reason_codes"] = dict(self.reason_codes)
        self.metadata["metric_registry"] = {
            "version": self.parquet_writer.metric_registry_version,
            "fingerprint": self.parquet_writer.metric_registry_fingerprint,
            "resolved_metrics": list(
                self.data_config["output_schema"]["metric_data"]
            ),
        }
        self.metadata["artifacts"] = {
            "summary": {"status": self.summary_status},
            "visualization": {"status": self.visualization_status},
        }
        if self.summary_error is not None:
            self.metadata["artifacts"]["summary"]["error"] = self.summary_error
        if self.visualization_error is not None:
            self.metadata["artifacts"]["visualization"][
                "error"
            ] = self.visualization_error
        if self.visualization_manager is not None:
            self.metadata["visualization"] = self.visualization_manager.metadata()
        self.write_metadata()

    def save_episode(self, episode_idx, n_steps, n_intervals):
        if len(self.metadata) == 0:
            raise RuntimeError("metadata must be initialized before persisting episodes")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if n_intervals <= 0:
            raise ValueError("n_intervals must be positive")

        if self.visualization_manager is not None and self.visualization_error is None:
            try:
                self.visualization_manager.accumulate_episode(
                    n_steps,
                    self.episode.end_termination_reason,
                )
            except Exception as error:
                self.visualization_status = "failed"
                self.visualization_error = _error_text(error)

        self.summary_rows.append(
            self.build_summary_row(episode_idx, n_steps, n_intervals)
        )
        try:
            part_closed = self.parquet_writer.write_episode(
                episode_idx,
                n_steps,
                n_intervals,
            )
        except Exception as error:
            self.fail_run(error)
            raise

        if part_closed:
            self.update_metadata("running")

    def fail_run(self, error):
        if self.persistence_status == "completed":
            return
        self.parquet_writer.abort()
        self.persistence_status = "failed"
        self.metadata["failure"] = _error_text(error)
        self.update_metadata("failed")

    def build_summary_row(self, episode_idx, n_steps, n_intervals):
        row = {}
        for section_name, section_config in self.summary_config.items():
            if section_name == "episode_fields":
                self.add_episode_fields(
                    row,
                    section_config,
                    episode_idx,
                    n_steps,
                    n_intervals,
                )
            elif section_name == "derived_fields":
                self.add_derived_fields(row, section_config)
            elif section_name == "end_episode_fields":
                self.add_end_episode_fields(row, section_config)
            elif section_name == "step_stats":
                self.add_scope_stats(
                    row,
                    "step",
                    self.step,
                    n_steps,
                    section_config,
                )
            elif section_name == "interval_stats":
                self.add_scope_stats(
                    row,
                    "interval",
                    self.interval,
                    n_intervals,
                    section_config,
                )
            else:
                raise ValueError(
                    f"Unsupported summary config section: {section_name}"
                )
        return row

    @staticmethod
    def add_episode_fields(row, field_names, episode_idx, n_steps, n_intervals):
        for field_name in field_names:
            if field_name == "episode_id":
                row[field_name] = int(episode_idx)
            elif field_name == "n_step":
                row[field_name] = int(n_steps)
            elif field_name == "n_interval":
                row[field_name] = int(n_intervals)
            else:
                raise ValueError(
                    f"Unsupported episode summary field: {field_name}"
                )

    def add_derived_fields(self, row, field_names):
        for field_name in field_names:
            row[field_name] = self.derive_summary_field(field_name)

    def add_end_episode_fields(self, row, field_names):
        for field_name in field_names:
            if not hasattr(self.episode, field_name):
                raise KeyError(f"Unknown end episode summary field: {field_name}")
            row[field_name] = getattr(self.episode, field_name)

    def derive_summary_field(self, field_name):
        if field_name == "end_termination_reason":
            return self.episode.end_termination_reason
        if field_name == "final_t_sec":
            return float(self.episode.final_t_sec)
        if field_name == "performance":
            final_t_sec = float(self.episode.final_t_sec)
            if final_t_sec == 0.0:
                return None
            return float(self.episode.total_reward) / final_t_sec
        raise ValueError(f"Unsupported derived summary field: {field_name}")

    def add_scope_stats(self, row, scope_name, scope, count, stat_specs):
        for stat_spec in stat_specs:
            source_name = stat_spec["source"]
            if not hasattr(scope, source_name):
                raise KeyError(f"Unknown {scope_name} summary signal: {source_name}")
            if (
                scope_name == "interval"
                and not self.parquet_writer.interval_signal_is_applicable(
                    source_name
                )
            ):
                for stat_name in stat_spec["stats"]:
                    row[f"{scope_name}_{source_name}_{stat_name}"] = None
                continue
            values = getattr(scope, source_name)[:count]
            for stat_name in stat_spec["stats"]:
                row[f"{scope_name}_{source_name}_{stat_name}"] = _series_stat(
                    values,
                    stat_name,
                )

    def save_agent_state(self, episode_idx, agent_state):
        self.latest_agent_state = agent_state
        json_path = self.agent_state_dir / f"agent_state_episode_{episode_idx:06d}.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(
                agent_state,
                file,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        q_rows, visit_rows = _agent_tables(agent_state)
        xlsx_path = self.agent_state_dir / f"agent_state_episode_{episode_idx:06d}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(q_rows).to_excel(writer, sheet_name="q_tables", index=False)
            pd.DataFrame(visit_rows).to_excel(
                writer,
                sheet_name="q_visits",
                index=False,
            )
            pd.DataFrame(
                [
                    {"field": "epsilon", "value": agent_state["epsilon"]},
                    {
                        "field": "learning_rate",
                        "value": agent_state["learning_rate"],
                    },
                    {
                        "field": "discount_factor",
                        "value": agent_state["discount_factor"],
                    },
                    {
                        "field": "action_moves",
                        "value": json.dumps(agent_state["action_moves"]),
                    },
                ]
            ).to_excel(writer, sheet_name="agent_meta", index=False)

    def finalize_run(self):
        try:
            self.parquet_writer.finalize()
            self.persistence_status = "completed"
        except Exception as error:
            self.fail_run(error)
            raise

        if self.visualization_manager is not None and self.visualization_error is None:
            try:
                self.visualization_manager.render_post_run(
                    self.summary_rows,
                    self.latest_agent_state,
                )
                self.visualization_status = "completed"
            except Exception as error:
                self.visualization_status = "failed"
                self.visualization_error = _error_text(error)

        self.summary_status = "running"
        projected_run_status = (
            "completed_with_artifact_errors"
            if self.visualization_status == "failed"
            else "completed"
        )
        self.update_metadata("running")
        try:
            self.write_summary_workbook(projected_run_status)
            self.summary_status = "completed"
        except Exception as error:
            self.summary_status = "failed"
            self.summary_error = _error_text(error)

        artifact_failed = (
            self.summary_status == "failed"
            or self.visualization_status == "failed"
        )
        run_status = "completed_with_artifact_errors" if artifact_failed else "completed"
        self.update_metadata(run_status)

    def write_summary_workbook(self, projected_run_status):
        summary_path = self.output_dir / "summary.xlsx"
        temporary_path = self.output_dir / "summary.tmp.xlsx"
        workbook_metadata = copy.deepcopy(self.metadata)
        workbook_metadata["status"] = projected_run_status
        workbook_metadata["artifacts"]["summary"]["status"] = "completed"
        metadata_rows = []
        _flatten_metadata("", workbook_metadata, metadata_rows)
        try:
            with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
                pd.DataFrame(self.summary_rows).to_excel(
                    writer,
                    sheet_name="episode_summary",
                    index=False,
                )
                pd.DataFrame(metadata_rows).to_excel(
                    writer,
                    sheet_name="metadata",
                    index=False,
                )
                if self.latest_agent_state is not None:
                    q_rows, visit_rows = _agent_tables(self.latest_agent_state)
                    pd.DataFrame(q_rows).to_excel(
                        writer,
                        sheet_name="q_tables_final",
                        index=False,
                    )
                    pd.DataFrame(visit_rows).to_excel(
                        writer,
                        sheet_name="q_visits_final",
                        index=False,
                    )
            temporary_path.replace(summary_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
