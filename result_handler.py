import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


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
        rows.append({"path": prefix, "value": json.dumps(value, ensure_ascii=False, default=_json_default)})
    else:
        rows.append({"path": prefix, "value": value})


def _agent_tables(agent_state):
    q_rows = []
    visit_rows = []
    for agent in agent_state["agents"]:
        gain_values = agent["gain_values"]
        q_table = agent["q_table"]
        visits = agent["visits"]
        for state_idx, gain_value in enumerate(gain_values):
            q_rows.append(
                {
                    "agent_id": agent["agent_id"],
                    "controller_id": agent["controller_id"],
                    "gain_name": agent["gain_name"],
                    "state_index": state_idx,
                    "gain_value": gain_value,
                    "q_decrease": q_table[state_idx][0],
                    "q_maintain": q_table[state_idx][1],
                    "q_increase": q_table[state_idx][2],
                }
            )
            visit_rows.append(
                {
                    "agent_id": agent["agent_id"],
                    "controller_id": agent["controller_id"],
                    "gain_name": agent["gain_name"],
                    "state_index": state_idx,
                    "gain_value": gain_value,
                    "visits_decrease": visits[state_idx][0],
                    "visits_maintain": visits[state_idx][1],
                    "visits_increase": visits[state_idx][2],
                }
            )
    return q_rows, visit_rows


def _series_stat(values, stat_name):
    if len(values) == 0:
        return 0.0
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


class ResultHandler:
    def __init__(self, output_dir, data_config, termination_config):
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "dataset_chunks"
        self.agent_state_dir = self.output_dir / "agent_state"
        self.figures_dir = self.output_dir / "figures"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.agent_state_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_dataset_chunk = int(data_config["episodes_per_dataset_chunk"])
        if self.episodes_per_dataset_chunk <= 0:
            raise ValueError("episodes_per_dataset_chunk must be positive")
        self.summary_config = data_config["summary"]
        self.reason_names_by_code = {}
        for reason_name, reason_code in termination_config["reason_codes"].items():
            self.reason_names_by_code[int(reason_code)] = reason_name
        self.pending_episodes = []
        self.chunk_idx = 0
        self.summary_rows = []
        self.metadata = {}
        self.latest_agent_state = None

    def save_metadata(self, metadata):
        self.metadata = metadata
        metadata_path = self.output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2, default=_json_default)

    def save_episode(self, episode_output):
        self.pending_episodes.append(episode_output)
        self.summary_rows.append(self.build_summary_row(episode_output))
        if len(self.pending_episodes) >= self.episodes_per_dataset_chunk:
            self.flush_chunk()

    def build_summary_row(self, episode_output):
        step_data = episode_output["step_data"]
        interval_data = episode_output["interval_data"]
        end_data = episode_output["end_episode_data"]
        row = {}

        for section_name, section_config in self.summary_config.items():
            if section_name == "episode_fields":
                self.add_episode_fields(row, section_config, episode_output, step_data, interval_data)
            elif section_name == "derived_fields":
                self.add_derived_fields(row, section_config, step_data, end_data)
            elif section_name == "end_episode_fields":
                self.add_end_episode_fields(row, section_config, end_data)
            elif section_name == "step_stats":
                self.add_scope_stats(row, "step", step_data, section_config)
            elif section_name == "interval_stats":
                self.add_scope_stats(row, "interval", interval_data, section_config)
            else:
                raise ValueError(f"Unsupported summary config section: {section_name}")
        return row

    def add_episode_fields(self, row, field_names, episode_output, step_data, interval_data):
        for field_name in field_names:
            if field_name == "episode_id":
                row[field_name] = episode_output["episode_id"]
            elif field_name == "n_step":
                row[field_name] = step_data["n_step"]
            elif field_name == "n_interval":
                row[field_name] = interval_data["n_interval"]
            else:
                raise ValueError(f"Unsupported episode summary field: {field_name}")

    def add_derived_fields(self, row, field_names, step_data, end_data):
        for field_name in field_names:
            row[field_name] = self.derive_summary_field(field_name, step_data, end_data)

    def add_end_episode_fields(self, row, field_names, end_data):
        for field_name in field_names:
            row[field_name] = end_data[field_name]

    def derive_summary_field(self, field_name, step_data, end_data):
        if field_name == "end_termination_reason":
            if field_name in end_data:
                return end_data[field_name]
            reason_code = int(end_data["end_termination_reason_code"])
            return self.reason_names_by_code[reason_code]
        if field_name == "final_t_sec":
            if field_name in end_data:
                return end_data[field_name]
            return _series_stat(step_data["t_sec"], "final")
        if field_name == "performance":
            if "final_t_sec" in end_data:
                final_t_sec = float(end_data["final_t_sec"])
            else:
                final_t_sec = _series_stat(step_data["t_sec"], "final")
            if final_t_sec <= 0.0:
                return 0.0
            return float(end_data["total_reward"]) / final_t_sec
        raise ValueError(f"Unsupported derived summary field: {field_name}")

    def add_scope_stats(self, row, scope_name, scope_data, stat_specs):
        for stat_spec in stat_specs:
            source_name = stat_spec["source"]
            values = scope_data[source_name]
            for stat_name in stat_spec["stats"]:
                row[f"{scope_name}_{source_name}_{stat_name}"] = _series_stat(values, stat_name)

    def flush_chunk(self):
        if len(self.pending_episodes) == 0:
            return
        chunk_output = {
            "chunk_id": self.chunk_idx,
            "n_episodes": len(self.pending_episodes),
            "episodes": self.pending_episodes,
        }
        chunk_path = self.dataset_dir / f"dataset_chunk_{self.chunk_idx:06d}.json"
        with chunk_path.open("w", encoding="utf-8") as file:
            json.dump(chunk_output, file, ensure_ascii=False, default=_json_default)
        self.pending_episodes = []
        self.chunk_idx += 1

    def save_agent_state(self, episode_idx, agent_state):
        self.latest_agent_state = agent_state
        json_path = self.agent_state_dir / f"agent_state_episode_{episode_idx:06d}.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(agent_state, file, ensure_ascii=False, indent=2, default=_json_default)

        q_rows, visit_rows = _agent_tables(agent_state)
        xlsx_path = self.agent_state_dir / f"agent_state_episode_{episode_idx:06d}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(q_rows).to_excel(writer, sheet_name="q_tables", index=False)
            pd.DataFrame(visit_rows).to_excel(writer, sheet_name="q_visits", index=False)
            pd.DataFrame(
                [
                    {"field": "epsilon", "value": agent_state["epsilon"]},
                    {"field": "learning_rate", "value": agent_state["learning_rate"]},
                    {"field": "discount_factor", "value": agent_state["discount_factor"]},
                    {"field": "action_moves", "value": json.dumps(agent_state["action_moves"])},
                ]
            ).to_excel(writer, sheet_name="agent_meta", index=False)

    def finalize_run(self):
        self.flush_chunk()
        self.write_summary_workbook()

    def write_summary_workbook(self):
        summary_path = self.output_dir / "summary.xlsx"
        metadata_rows = []
        _flatten_metadata("", self.metadata, metadata_rows)
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            pd.DataFrame(self.summary_rows).to_excel(writer, sheet_name="episode_summary", index=False)
            pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="metadata", index=False)
            if self.latest_agent_state is not None:
                q_rows, visit_rows = _agent_tables(self.latest_agent_state)
                pd.DataFrame(q_rows).to_excel(writer, sheet_name="q_tables_final", index=False)
                pd.DataFrame(visit_rows).to_excel(writer, sheet_name="q_visits_final", index=False)
