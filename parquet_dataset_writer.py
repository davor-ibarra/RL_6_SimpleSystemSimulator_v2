import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATA_LEVELS = (
    "step_data",
    "interval_data",
    "end_episode_data",
    "metric_data",
)

TECHNICAL_FIELDS = {
    "step_data": ("episode_id", "step_id"),
    "interval_data": ("episode_id", "interval_id"),
    "end_episode_data": ("episode_id", "n_step", "n_interval"),
    "metric_data": ("episode_id",),
}

REQUIRED_INTERVAL_OUTPUTS = frozenset(
    (
        "interval_mean_abs_error_pendulum_angle_raw",
        "interval_mean_abs_error_cart_position_raw",
        "interval_rms_pendulum_velocity_raw",
        "interval_rms_cart_velocity_raw",
        "interval_control_mean_square",
        "interval_saturation_frequency",
    )
)

FACTORIAL_BASE_INTERVAL_SIGNALS = frozenset(
    (
        "absolute_common_utility_normalized_signed",
        "factorial_absolute_contribution_signed",
    )
)

FACTORIAL_BALANCE_INTERVAL_SIGNALS = frozenset(
    (
        "balance_utility_normalized_signed",
        "factorial_balance_contribution_signed",
    )
)

FACTORIAL_SYNCHRONY_INTERVAL_SIGNALS = frozenset(
    (
        "synchrony_utility_normalized_signed",
        "factorial_synchrony_contribution_signed",
    )
)

FACTORIAL_DIFFERENTIAL_INTERVAL_SIGNALS = frozenset(
    ("factorial_differential_contribution_signed",)
)

FACTORIAL_COMMON_PROGRESS_INTERVAL_SIGNALS = frozenset(
    ("factorial_common_progress_contribution_signed",)
)

INTENSITY_INTERVAL_SIGNALS = frozenset(
    (
        "generalized_welfare_01",
        "collective_noncompensability_loss_01",
        "collective_noncompensability_weighted_loss_01",
    )
)

MONOTONIC_BASE_INTERVAL_SIGNALS = frozenset(
    (
        "monotonic_weighted_mean_health_01",
        "monotonic_minimum_health_01",
        "monotonic_health_gap_01",
        "monotonic_utility_01",
        "monotonic_utility_post_penalty_01",
        "monotonic_potential_shaping_signed",
        "monotonic_base_reward_signed",
        "monotonic_post_penalty_reward_signed",
    )
)

MONOTONIC_POST_COMPOSE_PENALTY_INTERVAL_SIGNALS = frozenset(
    (
        "monotonic_post_compose_penalty_activation_01",
        "monotonic_post_compose_penalty_cost_01",
        "monotonic_post_compose_penalty_01",
    )
)

POTENTIAL_BASE_INTERVAL_SIGNALS = frozenset(
    (
        "cooperative_cost_01",
        "cooperative_potential_base_01",
        "collective_potential_01",
        "cooperative_potential_01",
        "delta_cooperative_potential_signed",
        "cooperative_progress_residual_signed",
        "residual_weight_effective",
    )
)

POTENTIAL_CONCURRENT_PROGRESS_INTERVAL_SIGNALS = frozenset(
    (
        "cooperative_positive_progress_01",
        "cooperative_deterioration_01",
        "cooperative_progress_signed",
    )
)

POTENTIAL_BALANCE_INTERVAL_SIGNALS = frozenset(
    ("health_balance_01", "balance_factor_01")
)

POTENTIAL_SYNCHRONY_INTERVAL_SIGNALS = frozenset(("synchrony_factor_01",))

RL_INTERVAL_PREFIXES = (
    "reward_agent_",
    "reward_for_learning_",
    "action_",
    "is_exploratory_",
    "delta_gain_requested_",
    "delta_gain_applied_",
    "td_error_",
    "q_value_",
)

STATIC_METADATA_SIGNALS = frozenset(("run_id", "config_id", "seed"))
END_OBSERVATION_SIGNALS = frozenset(("epsilon", "learning_rate"))

TRAINING_APPLICABILITY = "simulation.mode == 'training'"
FACTORIAL_APPLICABILITY = (
    "reward.cooperative.enabled == true and "
    "reward.cooperative.type == 'factorial_cooperative'"
)
FACTORIAL_BALANCE_APPLICABILITY = (
    f"{FACTORIAL_APPLICABILITY} and "
    "reward.cooperative.factorial_cooperative.balance.enabled == true"
)
FACTORIAL_SYNCHRONY_APPLICABILITY = (
    f"{FACTORIAL_APPLICABILITY} and "
    "reward.cooperative.factorial_cooperative.synchrony.enabled == true"
)
FACTORIAL_DIFFERENTIAL_APPLICABILITY = (
    f"{FACTORIAL_APPLICABILITY} and "
    "reward.cooperative.factorial_cooperative.differential.enabled == true"
)
FACTORIAL_COMMON_PROGRESS_APPLICABILITY = (
    f"{FACTORIAL_APPLICABILITY} and "
    "reward.cooperative.factorial_cooperative.composition_mode == 'common_progress'"
)
INTENSITY_APPLICABILITY = (
    "reward.cooperative.enabled == true and "
    "reward.cooperative.type == 'intensity_compensability_cooperative'"
)
MONOTONIC_APPLICABILITY = (
    "reward.cooperative.enabled == true and "
    "reward.cooperative.type == 'monotonic_cooperative'"
)
MONOTONIC_POST_COMPOSE_PENALTY_APPLICABILITY = (
    f"{MONOTONIC_APPLICABILITY} and "
    "reward.cooperative.monotonic_cooperative.post_compose_penalty.enabled == true"
)
POTENTIAL_APPLICABILITY = (
    "reward.cooperative.enabled == true and "
    "reward.cooperative.type == 'potential_cooperative'"
)
POTENTIAL_CONCURRENT_PROGRESS_APPLICABILITY = (
    f"{POTENTIAL_APPLICABILITY} and "
    "reward.cooperative.potential_cooperative.potential.method == 'concurrent_progress'"
)
POTENTIAL_BALANCE_APPLICABILITY = (
    f"{POTENTIAL_APPLICABILITY} and "
    "reward.cooperative.potential_cooperative.balance.enabled == true"
)
POTENTIAL_SYNCHRONY_APPLICABILITY = (
    f"{POTENTIAL_APPLICABILITY} and "
    "reward.cooperative.potential_cooperative.synchrony.enabled == true"
)


def _require_integer(value, field_name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer")
    integer_value = int(value)
    if float(value) != float(integer_value):
        raise ValueError(f"{field_name} must be an integer")
    return integer_value


def _require_boolean(value, field_name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
        raise ValueError(f"{field_name} must be boolean")
    return bool(numeric_value)


def _is_boolean_signal(signal_name):
    return (
        signal_name == "terminated"
        or signal_name == "end_terminated"
        or signal_name.startswith("is_")
    )


def _is_integer_signal(signal_name):
    return (
        signal_name in ("episode_id", "step_id", "interval_id", "n_step", "n_interval")
        or signal_name.endswith("_id")
        or signal_name.endswith("_idx")
        or signal_name.endswith("_code")
        or signal_name.endswith("_n_steps")
        or "_count" in signal_name
        or signal_name == "total_agent_decisions"
        or signal_name.startswith("action_")
    )


def _signal_type(signal_name):
    if _is_boolean_signal(signal_name):
        return pa.bool_()
    if _is_integer_signal(signal_name):
        return pa.int64()
    return pa.float64()


def _metric_type(metric_spec):
    if metric_spec.unit_or_domain == "binario":
        return pa.bool_()
    if metric_spec.number_format == "0":
        return pa.int64()
    return pa.float64()


def _interval_applicability(signal_name):
    if signal_name.startswith(RL_INTERVAL_PREFIXES):
        return TRAINING_APPLICABILITY
    if signal_name in FACTORIAL_BALANCE_INTERVAL_SIGNALS:
        return FACTORIAL_BALANCE_APPLICABILITY
    if signal_name in FACTORIAL_SYNCHRONY_INTERVAL_SIGNALS:
        return FACTORIAL_SYNCHRONY_APPLICABILITY
    if signal_name in FACTORIAL_DIFFERENTIAL_INTERVAL_SIGNALS or signal_name.startswith(
        "differential_utility_normalized_signed_"
    ):
        return FACTORIAL_DIFFERENTIAL_APPLICABILITY
    if signal_name in FACTORIAL_COMMON_PROGRESS_INTERVAL_SIGNALS:
        return FACTORIAL_COMMON_PROGRESS_APPLICABILITY
    if signal_name in FACTORIAL_BASE_INTERVAL_SIGNALS:
        return FACTORIAL_APPLICABILITY
    if signal_name in INTENSITY_INTERVAL_SIGNALS:
        return INTENSITY_APPLICABILITY
    if signal_name in MONOTONIC_POST_COMPOSE_PENALTY_INTERVAL_SIGNALS or signal_name.startswith(
        "monotonic_post_compose_penalty_contribution_01_"
    ):
        return MONOTONIC_POST_COMPOSE_PENALTY_APPLICABILITY
    if signal_name in MONOTONIC_BASE_INTERVAL_SIGNALS:
        return MONOTONIC_APPLICABILITY
    if signal_name in POTENTIAL_CONCURRENT_PROGRESS_INTERVAL_SIGNALS:
        return POTENTIAL_CONCURRENT_PROGRESS_APPLICABILITY
    if signal_name in POTENTIAL_BALANCE_INTERVAL_SIGNALS:
        return POTENTIAL_BALANCE_APPLICABILITY
    if signal_name in POTENTIAL_SYNCHRONY_INTERVAL_SIGNALS:
        return POTENTIAL_SYNCHRONY_APPLICABILITY
    if signal_name in POTENTIAL_BASE_INTERVAL_SIGNALS or signal_name.startswith(
        ("cooperative_weight_effective_", "local_need_")
    ):
        return POTENTIAL_APPLICABILITY
    return None


def _metric_registry_fingerprint(metric_specs):
    fingerprint = hashlib.sha256()
    for metric_spec in metric_specs:
        fields = [
            metric_spec.canonical_name,
            metric_spec.scientific_label,
            metric_spec.family,
            metric_spec.entity_type,
            metric_spec.entity_id,
            metric_spec.unit_or_domain,
            metric_spec.definition,
            metric_spec.applicability,
            metric_spec.denominator,
            metric_spec.recommended_summary,
            metric_spec.interpretation,
            metric_spec.number_format,
            metric_spec.sources,
        ]
        for field_value in fields:
            fingerprint.update(str(field_value).encode("utf-8"))
            fingerprint.update(b"\x00")
        fingerprint.update(b"\x01")
    return f"sha256:{fingerprint.hexdigest()}"


def _integer_array(values, field_name):
    numeric_values = np.asarray(values, dtype=np.float64)
    if numeric_values.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"{field_name} contains non-finite integer values")
    if not np.all(numeric_values == np.trunc(numeric_values)):
        raise ValueError(f"{field_name} contains non-integer values")
    return pa.array(numeric_values.astype(np.int64), type=pa.int64())


def _boolean_array(values, field_name):
    numeric_values = np.asarray(values, dtype=np.float64)
    if numeric_values.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"{field_name} contains non-finite boolean values")
    valid_values = (numeric_values == 0.0) | (numeric_values == 1.0)
    if not np.all(valid_values):
        raise ValueError(f"{field_name} contains values outside 0 and 1")
    return pa.array(numeric_values.astype(np.bool_), type=pa.bool_())


def _float_array(values, field_name):
    numeric_values = np.asarray(values, dtype=np.float64)
    if numeric_values.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"{field_name} contains non-finite values")
    return pa.array(numeric_values, type=pa.float64())


def _column_array(values, field_type, field_name):
    if pa.types.is_boolean(field_type):
        return _boolean_array(values, field_name)
    if pa.types.is_int64(field_type):
        return _integer_array(values, field_name)
    return _float_array(values, field_name)


def _scalar_array(value, field_type, field_name):
    if value is None:
        return pa.nulls(1, type=field_type)
    scalar_values = np.asarray((value,))
    return _column_array(scalar_values, field_type, field_name)


class ParquetDatasetWriter:
    SCHEMA_VERSION = "1.0.0"

    def __init__(self, output_dir, data_config, runtime, metric_report, run_id):
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "dataset_chunks"
        self.data_config = data_config
        self.runtime = runtime
        self.metric_report = metric_report
        self.config = metric_report.config
        self.run_id = str(run_id)
        self.schema_version = self.SCHEMA_VERSION
        if len(self.run_id) == 0:
            raise ValueError("run_id must not be empty")

        self.episodes_per_dataset_chunk = _require_integer(
            data_config["episodes_per_dataset_chunk"],
            "episodes_per_dataset_chunk",
        )
        if self.episodes_per_dataset_chunk <= 0:
            raise ValueError("episodes_per_dataset_chunk must be positive")

        self.n_episodes = _require_integer(
            self.config["simulation"]["n_episodes"],
            "simulation.n_episodes",
        )
        if self.n_episodes <= 0:
            raise ValueError("simulation.n_episodes must be positive")

        self.simulation_mode = self.config["simulation"]["mode"]
        if self.simulation_mode not in ("training", "fixed_pid"):
            raise ValueError("simulation.mode must be training or fixed_pid")

        cooperative_config = self.config["reward"]["cooperative"]
        self.cooperative_enabled = bool(cooperative_config["enabled"])
        self.cooperative_type = cooperative_config["type"]
        if self.cooperative_type not in (
            "factorial_cooperative",
            "intensity_compensability_cooperative",
            "potential_cooperative",
            "monotonic_cooperative",
        ):
            raise ValueError(f"Unsupported cooperative reward type: {self.cooperative_type}")

        factorial_config = cooperative_config["factorial_cooperative"]
        self.factorial_balance_enabled = bool(
            factorial_config["balance"]["enabled"]
        )
        self.factorial_synchrony_enabled = bool(
            factorial_config["synchrony"]["enabled"]
        )
        self.factorial_differential_enabled = bool(
            factorial_config["differential"]["enabled"]
        )
        self.factorial_composition_mode = factorial_config["composition_mode"]

        potential_config = cooperative_config["potential_cooperative"]
        self.potential_method = potential_config["potential"]["method"]
        self.potential_balance_enabled = bool(
            potential_config["balance"]["enabled"]
        )
        self.potential_synchrony_enabled = bool(
            potential_config["synchrony"]["enabled"]
        )
        self.monotonic_post_compose_penalty_enabled = bool(
            cooperative_config["monotonic_cooperative"]["post_compose_penalty"][
                "enabled"
            ]
        )

        self.output_schema = self._validate_output_schema(data_config["output_schema"])
        self.metric_specs_by_name = self._build_metric_specs_by_name()
        self.metric_registry_fingerprint = _metric_registry_fingerprint(
            self.metric_report.metric_specs
        )
        self.metric_registry_version = str(self.metric_report.registry_version)
        if len(self.metric_registry_version) == 0:
            raise ValueError("metric registry version must not be empty")
        self._validate_runtime_signals()
        self.schemas = self._build_schemas()

        self.level_dirs = {}
        for data_level in DATA_LEVELS:
            level_dir = self.dataset_dir / data_level
            self.level_dirs[data_level] = level_dir

        self._prepare_directories()

        self._active_writers = None
        self._active_schemas = None
        self._active_temp_paths = None
        self._active_final_paths = None
        self._active_part_index = None
        self._active_first_episode_id = None
        self._active_last_episode_id = None
        self._part_episode_counts = []
        self._parts = []
        self._total_rows = {data_level: 0 for data_level in DATA_LEVELS}
        self._episodes_written = 0
        self._finalized = False
        self._failed = False

    def _validate_output_schema(self, output_schema):
        if not isinstance(output_schema, dict):
            raise TypeError("data.output_schema must be a mapping")
        configured_levels = set(output_schema)
        required_levels = set(DATA_LEVELS)
        if configured_levels != required_levels:
            missing_levels = sorted(required_levels - configured_levels)
            unexpected_levels = sorted(configured_levels - required_levels)
            raise ValueError(
                "data.output_schema must contain exactly four levels; "
                f"missing={missing_levels}, unexpected={unexpected_levels}"
            )

        all_technical_fields = set()
        for technical_fields in TECHNICAL_FIELDS.values():
            all_technical_fields.update(technical_fields)

        validated_schema = {}
        owner_by_signal = {}
        for data_level in DATA_LEVELS:
            signal_names = output_schema[data_level]
            if not isinstance(signal_names, (list, tuple)):
                raise TypeError(f"data.output_schema.{data_level} must be a list")

            validated_names = []
            names_in_level = set()
            for signal_name in signal_names:
                if not isinstance(signal_name, str) or len(signal_name) == 0:
                    raise TypeError(f"Invalid signal name in {data_level}")
                if signal_name in names_in_level:
                    raise ValueError(f"Duplicate {data_level} signal: {signal_name}")
                if signal_name in all_technical_fields:
                    raise ValueError(
                        f"{signal_name} is an automatic technical field and must not "
                        f"be declared in {data_level}"
                    )
                if signal_name in owner_by_signal:
                    raise ValueError(
                        f"Signal {signal_name} is declared in both "
                        f"{owner_by_signal[signal_name]} and {data_level}"
                    )
                if signal_name in STATIC_METADATA_SIGNALS:
                    raise ValueError(
                        f"Static signal {signal_name} belongs in metadata, not {data_level}"
                    )
                if data_level == "step_data" and signal_name.startswith(
                    ("kp_", "ki_", "kd_")
                ):
                    raise ValueError(
                        f"Gain signal {signal_name} belongs in interval_data"
                    )
                names_in_level.add(signal_name)
                owner_by_signal[signal_name] = data_level
                validated_names.append(signal_name)
            validated_schema[data_level] = tuple(validated_names)

        missing_interval_outputs = sorted(
            REQUIRED_INTERVAL_OUTPUTS - set(validated_schema["interval_data"])
        )
        if missing_interval_outputs:
            raise ValueError(
                "data.output_schema.interval_data is missing required reductions: "
                f"{missing_interval_outputs}"
            )
        return validated_schema

    def _build_metric_specs_by_name(self):
        metric_specs_by_name = {}
        for metric_spec in self.metric_report.metric_specs:
            metric_name = metric_spec.canonical_name
            if metric_name in metric_specs_by_name:
                raise ValueError(f"Duplicate metric specification: {metric_name}")
            metric_specs_by_name[metric_name] = metric_spec

        registered_names = tuple(self.metric_report.metric_signal_names)
        if len(set(registered_names)) != len(registered_names):
            raise ValueError("metric_report.metric_signal_names contains duplicates")
        if set(registered_names) != set(metric_specs_by_name):
            raise ValueError(
                "metric_report.metric_signal_names and metric_specs are inconsistent"
            )
        return metric_specs_by_name

    def _validate_runtime_signals(self):
        required_runtime_attributes = (
            "step",
            "interval",
            "episode",
            "step_capacity",
            "interval_capacity",
            "step_signal_names",
            "interval_signal_names",
            "end_episode_signal_names",
        )
        for attribute_name in required_runtime_attributes:
            if not hasattr(self.runtime, attribute_name):
                raise AttributeError(f"runtime is missing {attribute_name}")

        if not hasattr(self.runtime.interval, "interval_id"):
            raise KeyError("runtime.interval is missing interval_id")
        if not hasattr(self.runtime.interval, "interval_n_steps"):
            raise KeyError("runtime.interval is missing interval_n_steps")
        if not hasattr(self.runtime.episode, "n_step"):
            raise KeyError("runtime.episode is missing n_step")
        if not hasattr(self.runtime.episode, "n_interval"):
            raise KeyError("runtime.episode is missing n_interval")

        step_signal_names = set(self.runtime.step_signal_names)
        for signal_name in self.output_schema["step_data"]:
            if signal_name not in step_signal_names or not hasattr(self.runtime.step, signal_name):
                raise KeyError(f"Unknown step_data output signal: {signal_name}")

        interval_signal_names = set(self.runtime.interval_signal_names)
        for signal_name in self.output_schema["interval_data"]:
            if signal_name not in interval_signal_names or not hasattr(self.runtime.interval, signal_name):
                raise KeyError(f"Unknown interval_data output signal: {signal_name}")

        end_signal_names = set(self.runtime.end_episode_signal_names)
        for signal_name in self.output_schema["end_episode_data"]:
            if signal_name not in end_signal_names or not hasattr(self.runtime.episode, signal_name):
                raise KeyError(f"Unknown end_episode_data output signal: {signal_name}")
            if signal_name == "end_termination_reason":
                raise ValueError(
                    "end_termination_reason must be derived from its code and metadata"
                )
            if signal_name.startswith(("final_kp_", "final_ki_", "final_kd_")):
                raise ValueError(
                    f"Final gain {signal_name} must be selected from interval_data"
                )
            if (
                signal_name in self.metric_specs_by_name
                and signal_name not in END_OBSERVATION_SIGNALS
            ):
                raise ValueError(f"Metric {signal_name} belongs in metric_data")

        for signal_name in self.output_schema["metric_data"]:
            if signal_name not in self.metric_specs_by_name:
                raise KeyError(f"Unknown metric_data metric: {signal_name}")
            if signal_name in END_OBSERVATION_SIGNALS:
                raise ValueError(
                    f"Direct end observation {signal_name} belongs in end_episode_data"
                )
            if not hasattr(self.runtime.episode, signal_name):
                raise KeyError(f"runtime.episode is missing metric {signal_name}")

    def _build_schemas(self):
        schemas = {}
        for data_level in DATA_LEVELS:
            fields = []
            for field_name in TECHNICAL_FIELDS[data_level]:
                fields.append(pa.field(field_name, pa.int64(), nullable=False))
            for signal_name in self.output_schema[data_level]:
                if data_level == "metric_data":
                    field_type = _metric_type(self.metric_specs_by_name[signal_name])
                else:
                    field_type = _signal_type(signal_name)
                field_metadata = None
                if data_level == "interval_data":
                    applicability = _interval_applicability(signal_name)
                    if applicability is not None:
                        field_metadata = {
                            b"applicability": applicability.encode("utf-8")
                        }
                elif (
                    data_level == "end_episode_data"
                    and signal_name in END_OBSERVATION_SIGNALS
                ):
                    field_metadata = {
                        b"applicability": TRAINING_APPLICABILITY.encode("utf-8")
                    }
                fields.append(
                    pa.field(
                        signal_name,
                        field_type,
                        nullable=True,
                        metadata=field_metadata,
                    )
                )
            schemas[data_level] = pa.schema(fields)
        return schemas

    def _prepare_directories(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for data_level in DATA_LEVELS:
            level_dir = self.level_dirs[data_level]
            level_dir.mkdir(parents=True, exist_ok=True)
            for existing_path in level_dir.iterdir():
                if existing_path.name.startswith("part-"):
                    raise FileExistsError(
                        f"Parquet dataset directory is not empty: {level_dir}"
                    )

    def _file_metadata(self, data_level, part_index, first_episode_id, last_episode_id):
        return {
            b"run_id": self.run_id.encode("utf-8"),
            b"data_level": data_level.encode("utf-8"),
            b"schema_version": self.schema_version.encode("utf-8"),
            b"part_index": str(part_index).encode("ascii"),
            b"first_episode_id": str(first_episode_id).encode("ascii"),
            b"last_episode_id": str(last_episode_id).encode("ascii"),
        }

    def _open_part(self, part_index, first_episode_id):
        if self._active_writers is not None:
            raise RuntimeError("A Parquet part is already open")

        last_episode_id = min(
            first_episode_id + self.episodes_per_dataset_chunk - 1,
            self.n_episodes - 1,
        )
        active_writers = {}
        active_schemas = {}
        temp_paths = {}
        final_paths = {}

        try:
            for data_level in DATA_LEVELS:
                file_name = f"part-{part_index:06d}.parquet"
                final_path = self.level_dirs[data_level] / file_name
                temp_path = self.level_dirs[data_level] / f"{file_name}.tmp"
                if final_path.exists() or temp_path.exists():
                    raise FileExistsError(f"Parquet part already exists: {final_path}")

                file_schema = self.schemas[data_level].with_metadata(
                    self._file_metadata(
                        data_level,
                        part_index,
                        first_episode_id,
                        last_episode_id,
                    )
                )
                active_writers[data_level] = pq.ParquetWriter(
                    temp_path,
                    file_schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
                active_schemas[data_level] = file_schema
                temp_paths[data_level] = temp_path
                final_paths[data_level] = final_path
        except Exception:
            for writer in active_writers.values():
                writer.close()
            raise

        self._active_writers = active_writers
        self._active_schemas = active_schemas
        self._active_temp_paths = temp_paths
        self._active_final_paths = final_paths
        self._active_part_index = part_index
        self._active_first_episode_id = first_episode_id
        self._active_last_episode_id = last_episode_id
        self._part_episode_counts = []

    def _validate_episode_inputs(self, episode_id, n_steps, n_intervals):
        episode_id = _require_integer(episode_id, "episode_id")
        n_steps = _require_integer(n_steps, "n_steps")
        n_intervals = _require_integer(n_intervals, "n_intervals")

        if episode_id != self._episodes_written:
            raise ValueError(
                f"Episodes must be written in order; expected {self._episodes_written}, "
                f"received {episode_id}"
            )
        if episode_id >= self.n_episodes:
            raise ValueError(f"episode_id exceeds configured n_episodes: {episode_id}")
        if n_steps <= 0 or n_steps > int(self.runtime.step_capacity):
            raise ValueError(f"Invalid n_steps for episode {episode_id}: {n_steps}")
        if n_intervals <= 0 or n_intervals > int(self.runtime.interval_capacity):
            raise ValueError(f"Invalid n_intervals for episode {episode_id}: {n_intervals}")

        interval_ids = np.asarray(
            self.runtime.interval.interval_id[:n_intervals],
            dtype=np.float64,
        )
        expected_interval_ids = np.arange(n_intervals, dtype=np.float64)
        if not np.array_equal(interval_ids, expected_interval_ids):
            raise ValueError(f"Non-contiguous interval_id values in episode {episode_id}")

        interval_n_steps = np.asarray(
            self.runtime.interval.interval_n_steps[:n_intervals],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(interval_n_steps)):
            raise ValueError(f"Non-finite interval_n_steps in episode {episode_id}")
        if not np.all(interval_n_steps == np.trunc(interval_n_steps)):
            raise ValueError(f"Non-integer interval_n_steps in episode {episode_id}")
        if not np.all(interval_n_steps > 0.0):
            raise ValueError(f"Non-positive interval_n_steps in episode {episode_id}")
        if int(np.sum(interval_n_steps, dtype=np.float64)) != n_steps:
            raise ValueError(
                f"Interval step counts do not sum to n_steps in episode {episode_id}"
            )

        interval_start_steps = np.asarray(
            self.runtime.interval.interval_start_step_idx[:n_intervals],
            dtype=np.float64,
        )
        interval_end_steps = np.asarray(
            self.runtime.interval.interval_end_step_idx[:n_intervals],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(interval_start_steps)) or not np.all(
            np.isfinite(interval_end_steps)
        ):
            raise ValueError(f"Non-finite interval bounds in episode {episode_id}")
        if not np.all(interval_start_steps == np.trunc(interval_start_steps)) or not np.all(
            interval_end_steps == np.trunc(interval_end_steps)
        ):
            raise ValueError(f"Non-integer interval bounds in episode {episode_id}")
        expected_start_steps = np.concatenate(
            (
                np.asarray((0.0,)),
                np.cumsum(interval_n_steps[:-1], dtype=np.float64),
            )
        )
        expected_end_steps = expected_start_steps + interval_n_steps - 1.0
        if not np.array_equal(interval_start_steps, expected_start_steps):
            raise ValueError(f"Non-contiguous interval starts in episode {episode_id}")
        if not np.array_equal(interval_end_steps, expected_end_steps):
            raise ValueError(f"Invalid interval ends in episode {episode_id}")

        last_interval_terminated = _require_boolean(
            self.runtime.interval.terminated[n_intervals - 1],
            "interval.terminated[-1]",
        )
        end_terminated = _require_boolean(
            self.runtime.episode.end_terminated,
            "episode.end_terminated",
        )
        if last_interval_terminated != end_terminated:
            raise ValueError(f"Terminal flag mismatch in episode {episode_id}")
        last_reason_code = _require_integer(
            self.runtime.interval.termination_reason_code[n_intervals - 1],
            "interval.termination_reason_code[-1]",
        )
        end_reason_code = _require_integer(
            self.runtime.episode.end_termination_reason_code,
            "episode.end_termination_reason_code",
        )
        if last_reason_code != end_reason_code:
            raise ValueError(f"Termination reason mismatch in episode {episode_id}")

        runtime_n_steps = _require_integer(
            self.runtime.episode.n_step,
            "episode.n_step",
        )
        if runtime_n_steps != n_steps:
            raise ValueError(f"episode.n_step mismatch in episode {episode_id}")
        runtime_n_intervals = _require_integer(
            self.runtime.episode.n_interval,
            "episode.n_interval",
        )
        if runtime_n_intervals != n_intervals:
            raise ValueError(f"episode.n_interval mismatch in episode {episode_id}")

        return episode_id, n_steps, n_intervals

    def _interval_signal_is_applicable(self, signal_name):
        applicability = _interval_applicability(signal_name)
        if applicability is None:
            return True
        if applicability == TRAINING_APPLICABILITY:
            return self.simulation_mode == "training"
        factorial_active = (
            self.cooperative_enabled
            and self.cooperative_type == "factorial_cooperative"
        )
        if applicability == FACTORIAL_APPLICABILITY:
            return factorial_active
        if applicability == FACTORIAL_BALANCE_APPLICABILITY:
            return factorial_active and self.factorial_balance_enabled
        if applicability == FACTORIAL_SYNCHRONY_APPLICABILITY:
            return factorial_active and self.factorial_synchrony_enabled
        if applicability == FACTORIAL_DIFFERENTIAL_APPLICABILITY:
            return factorial_active and self.factorial_differential_enabled
        if applicability == FACTORIAL_COMMON_PROGRESS_APPLICABILITY:
            return (
                factorial_active
                and self.factorial_composition_mode == "common_progress"
            )
        if applicability == INTENSITY_APPLICABILITY:
            return (
                self.cooperative_enabled
                and self.cooperative_type == "intensity_compensability_cooperative"
            )
        monotonic_active = (
            self.cooperative_enabled
            and self.cooperative_type == "monotonic_cooperative"
        )
        if applicability == MONOTONIC_APPLICABILITY:
            return monotonic_active
        if applicability == MONOTONIC_POST_COMPOSE_PENALTY_APPLICABILITY:
            return monotonic_active and self.monotonic_post_compose_penalty_enabled
        potential_active = (
            self.cooperative_enabled
            and self.cooperative_type == "potential_cooperative"
        )
        if applicability == POTENTIAL_APPLICABILITY:
            return potential_active
        if applicability == POTENTIAL_CONCURRENT_PROGRESS_APPLICABILITY:
            return potential_active and self.potential_method == "concurrent_progress"
        if applicability == POTENTIAL_BALANCE_APPLICABILITY:
            return potential_active and self.potential_balance_enabled
        if applicability == POTENTIAL_SYNCHRONY_APPLICABILITY:
            return potential_active and self.potential_synchrony_enabled
        raise RuntimeError(f"Unsupported interval applicability rule: {applicability}")

    def interval_signal_is_applicable(self, signal_name):
        return self._interval_signal_is_applicable(signal_name)

    def _build_step_table(self, episode_id, n_steps):
        schema = self._active_schemas["step_data"]
        arrays = [
            pa.array(np.full(n_steps, episode_id, dtype=np.int64), type=pa.int64()),
            pa.array(np.arange(n_steps, dtype=np.int64), type=pa.int64()),
        ]
        for signal_name in self.output_schema["step_data"]:
            source = getattr(self.runtime.step, signal_name)
            if len(source) < n_steps:
                raise ValueError(f"step_data.{signal_name} is shorter than n_steps")
            field_type = schema.field(signal_name).type
            arrays.append(
                _column_array(source[:n_steps], field_type, f"step_data.{signal_name}")
            )
        return pa.Table.from_arrays(arrays, schema=schema)

    def _build_interval_table(self, episode_id, n_intervals):
        schema = self._active_schemas["interval_data"]
        arrays = [
            pa.array(np.full(n_intervals, episode_id, dtype=np.int64), type=pa.int64()),
            pa.array(np.arange(n_intervals, dtype=np.int64), type=pa.int64()),
        ]
        for signal_name in self.output_schema["interval_data"]:
            field_type = schema.field(signal_name).type
            if not self._interval_signal_is_applicable(signal_name):
                arrays.append(pa.nulls(n_intervals, type=field_type))
                continue
            source = getattr(self.runtime.interval, signal_name)
            if len(source) < n_intervals:
                raise ValueError(f"interval_data.{signal_name} is shorter than n_intervals")
            arrays.append(
                _column_array(
                    source[:n_intervals],
                    field_type,
                    f"interval_data.{signal_name}",
                )
            )
        return pa.Table.from_arrays(arrays, schema=schema)

    def _build_end_episode_table(self, episode_id, n_steps, n_intervals):
        schema = self._active_schemas["end_episode_data"]
        arrays = [
            pa.array(np.asarray((episode_id,), dtype=np.int64), type=pa.int64()),
            pa.array(np.asarray((n_steps,), dtype=np.int64), type=pa.int64()),
            pa.array(np.asarray((n_intervals,), dtype=np.int64), type=pa.int64()),
        ]
        for signal_name in self.output_schema["end_episode_data"]:
            field_type = schema.field(signal_name).type
            if (
                signal_name in END_OBSERVATION_SIGNALS
                and self.simulation_mode != "training"
            ):
                arrays.append(pa.nulls(1, type=field_type))
                continue
            value = getattr(self.runtime.episode, signal_name)
            arrays.append(
                _scalar_array(value, field_type, f"end_episode_data.{signal_name}")
            )
        return pa.Table.from_arrays(arrays, schema=schema)

    def _build_metric_table(self, episode_id):
        schema = self._active_schemas["metric_data"]
        arrays = [
            pa.array(np.asarray((episode_id,), dtype=np.int64), type=pa.int64()),
        ]
        for signal_name in self.output_schema["metric_data"]:
            field_type = schema.field(signal_name).type
            value = getattr(self.runtime.episode, signal_name)
            arrays.append(_scalar_array(value, field_type, f"metric_data.{signal_name}"))
        return pa.Table.from_arrays(arrays, schema=schema)

    def write_episode(self, episode_id, n_steps, n_intervals):
        if self._finalized:
            raise RuntimeError("Cannot write an episode after finalize")
        if self._failed:
            raise RuntimeError("Cannot write an episode after a Parquet writer failure")

        episode_id, n_steps, n_intervals = self._validate_episode_inputs(
            episode_id,
            n_steps,
            n_intervals,
        )
        part_index = episode_id // self.episodes_per_dataset_chunk
        try:
            if self._active_writers is None:
                self._open_part(part_index, episode_id)
            if part_index != self._active_part_index:
                raise RuntimeError("Parquet part index is not synchronized")

            tables = {
                "step_data": self._build_step_table(episode_id, n_steps),
                "interval_data": self._build_interval_table(
                    episode_id,
                    n_intervals,
                ),
                "end_episode_data": self._build_end_episode_table(
                    episode_id,
                    n_steps,
                    n_intervals,
                ),
                "metric_data": self._build_metric_table(episode_id),
            }
            for data_level in DATA_LEVELS:
                table = tables[data_level]
                self._active_writers[data_level].write_table(
                    table,
                    row_group_size=table.num_rows,
                )
        except Exception:
            self._failed = True
            self._close_active_writers_after_failure()
            raise

        self._part_episode_counts.append((episode_id, n_steps, n_intervals))
        self._episodes_written += 1

        part_is_complete = (
            len(self._part_episode_counts) == self.episodes_per_dataset_chunk
            or self._episodes_written == self.n_episodes
        )
        if part_is_complete:
            self._close_part()
        return part_is_complete

    def abort(self):
        if self._finalized:
            return
        self._failed = True
        self._close_active_writers_after_failure()

    def _close_active_writers_after_failure(self):
        if self._active_writers is None:
            return
        for writer in self._active_writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._active_writers = None

    def _close_writers(self):
        close_errors = []
        for data_level in DATA_LEVELS:
            try:
                self._active_writers[data_level].close()
            except Exception as error:
                close_errors.append((data_level, error))
        self._active_writers = None
        if close_errors:
            self._failed = True
            details = ", ".join(
                f"{data_level}: {error}" for data_level, error in close_errors
            )
            raise RuntimeError(f"Failed to close Parquet writers: {details}")

    def _expected_rows_for_level(self, data_level):
        if data_level == "step_data":
            return [item[1] for item in self._part_episode_counts]
        if data_level == "interval_data":
            return [item[2] for item in self._part_episode_counts]
        return [1 for item in self._part_episode_counts]

    def _validate_part_file(self, data_level):
        temp_path = self._active_temp_paths[data_level]
        parquet_file = pq.ParquetFile(temp_path)
        file_schema = parquet_file.schema_arrow
        expected_schema = self._active_schemas[data_level]
        if not file_schema.equals(expected_schema, check_metadata=True):
            raise ValueError(f"Schema mismatch in {temp_path}")

        file_metadata = file_schema.metadata
        expected_metadata = self._file_metadata(
            data_level,
            self._active_part_index,
            self._active_first_episode_id,
            self._active_last_episode_id,
        )
        if file_metadata is None:
            raise ValueError(f"Missing file metadata in {temp_path}")
        for metadata_key, metadata_value in expected_metadata.items():
            if metadata_key not in file_metadata:
                raise ValueError(f"Missing metadata {metadata_key!r} in {temp_path}")
            if file_metadata[metadata_key] != metadata_value:
                raise ValueError(f"Invalid metadata {metadata_key!r} in {temp_path}")

        expected_rows = self._expected_rows_for_level(data_level)
        if parquet_file.num_row_groups != len(expected_rows):
            raise ValueError(f"Expected one row group per episode in {temp_path}")
        if parquet_file.metadata.num_rows != sum(expected_rows):
            raise ValueError(f"Row count mismatch in {temp_path}")

        episode_id_column_idx = file_schema.get_field_index("episode_id")
        for row_group_idx, expected_row_count in enumerate(expected_rows):
            row_group_metadata = parquet_file.metadata.row_group(row_group_idx)
            if row_group_metadata.num_rows != expected_row_count:
                raise ValueError(
                    f"Row-group count mismatch in {temp_path}, group {row_group_idx}"
                )
            for column_idx in range(row_group_metadata.num_columns):
                if row_group_metadata.column(column_idx).compression != "ZSTD":
                    raise ValueError(f"Non-ZSTD column found in {temp_path}")

            episode_id = self._part_episode_counts[row_group_idx][0]
            statistics = row_group_metadata.column(episode_id_column_idx).statistics
            if statistics is None or not statistics.has_min_max:
                raise ValueError(f"Missing episode_id statistics in {temp_path}")
            if int(statistics.min) != episode_id or int(statistics.max) != episode_id:
                raise ValueError(f"Invalid episode_id statistics in {temp_path}")

            key_columns = ["episode_id"]
            if data_level == "step_data":
                key_columns.append("step_id")
            elif data_level == "interval_data":
                key_columns.append("interval_id")
            key_table = parquet_file.read_row_group(row_group_idx, columns=key_columns)
            episode_ids = key_table.column("episode_id").to_numpy(zero_copy_only=False)
            if not np.all(episode_ids == episode_id):
                raise ValueError(f"Unordered episode_id values in {temp_path}")
            if data_level == "step_data":
                step_ids = key_table.column("step_id").to_numpy(zero_copy_only=False)
                if not np.array_equal(step_ids, np.arange(expected_row_count)):
                    raise ValueError(f"Non-contiguous step_id values in {temp_path}")
            elif data_level == "interval_data":
                interval_ids = key_table.column("interval_id").to_numpy(
                    zero_copy_only=False
                )
                if not np.array_equal(interval_ids, np.arange(expected_row_count)):
                    raise ValueError(f"Non-contiguous interval_id values in {temp_path}")

        return parquet_file.metadata.num_rows

    def _rename_part_files(self):
        for data_level in DATA_LEVELS:
            if self._active_final_paths[data_level].exists():
                raise FileExistsError(
                    f"Final Parquet part already exists: {self._active_final_paths[data_level]}"
                )

        renamed_levels = []
        try:
            for data_level in DATA_LEVELS:
                self._active_temp_paths[data_level].rename(
                    self._active_final_paths[data_level]
                )
                renamed_levels.append(data_level)
        except Exception as rename_error:
            rollback_errors = []
            for data_level in reversed(renamed_levels):
                final_path = self._active_final_paths[data_level]
                temp_path = self._active_temp_paths[data_level]
                try:
                    final_path.rename(temp_path)
                except Exception as rollback_error:
                    rollback_errors.append((data_level, rollback_error))
            if rollback_errors:
                details = ", ".join(
                    f"{data_level}: {error}"
                    for data_level, error in rollback_errors
                )
                raise RuntimeError(
                    f"Parquet rename failed and rollback was incomplete: {details}"
                ) from rename_error
            raise

    def _close_part(self):
        if self._active_writers is None:
            return
        self._close_writers()

        try:
            row_counts = {}
            for data_level in DATA_LEVELS:
                row_counts[data_level] = self._validate_part_file(data_level)
            self._rename_part_files()
        except Exception:
            self._failed = True
            raise

        files = {}
        for data_level in DATA_LEVELS:
            files[data_level] = self._active_final_paths[data_level].relative_to(
                self.output_dir
            ).as_posix()
        self._parts.append(
            {
                "part_index": self._active_part_index,
                "first_episode_id": self._part_episode_counts[0][0],
                "last_episode_id": self._part_episode_counts[-1][0],
                "n_episodes": len(self._part_episode_counts),
                "row_counts": row_counts,
                "files": files,
            }
        )
        for data_level in DATA_LEVELS:
            self._total_rows[data_level] += row_counts[data_level]

        self._active_schemas = None
        self._active_temp_paths = None
        self._active_final_paths = None
        self._active_part_index = None
        self._active_first_episode_id = None
        self._active_last_episode_id = None
        self._part_episode_counts = []

    def finalize(self):
        if self._finalized:
            return self.manifest("completed")
        if self._failed:
            raise RuntimeError("Cannot finalize after a Parquet writer failure")
        if self._episodes_written != self.n_episodes:
            raise RuntimeError(
                f"Cannot finalize: expected {self.n_episodes} episodes, "
                f"received {self._episodes_written}"
            )
        if self._active_writers is not None:
            self._close_part()

        expected_parts = (
            self.n_episodes + self.episodes_per_dataset_chunk - 1
        ) // self.episodes_per_dataset_chunk
        if len(self._parts) != expected_parts:
            raise RuntimeError(
                f"Completed part count mismatch: expected {expected_parts}, "
                f"found {len(self._parts)}"
            )

        for data_level in DATA_LEVELS:
            manifested_rows = sum(
                part["row_counts"][data_level] for part in self._parts
            )
            if manifested_rows != self._total_rows[data_level]:
                raise RuntimeError(f"Manifest row count mismatch for {data_level}")

        self._finalized = True
        return self.manifest("completed")

    def manifest(self, status):
        if status not in ("running", "completed", "failed"):
            raise ValueError(f"Unsupported persistence status: {status}")
        if status == "completed" and not self._finalized:
            raise RuntimeError("Persistence is not complete")

        committed_episode_count = sum(
            part["n_episodes"] for part in self._parts
        )
        incomplete_episode_count = self._episodes_written - committed_episode_count
        datasets = {}
        for data_level in DATA_LEVELS:
            dataset_parts = []
            for part in self._parts:
                dataset_parts.append(
                    {
                        "part_index": part["part_index"],
                        "path": part["files"][data_level],
                        "first_episode_id": part["first_episode_id"],
                        "last_episode_id": part["last_episode_id"],
                        "row_count": part["row_counts"][data_level],
                    }
                )
            datasets[data_level] = {
                "directory": self.level_dirs[data_level].relative_to(
                    self.output_dir
                ).as_posix(),
                "row_count": self._total_rows[data_level],
                "parts": dataset_parts,
            }

        return {
            "status": status,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "metric_registry_version": self.metric_registry_version,
            "metric_registry_fingerprint": self.metric_registry_fingerprint,
            "n_episodes_expected": self.n_episodes,
            "n_episodes_written": committed_episode_count,
            "n_episodes_in_incomplete_part": incomplete_episode_count,
            "parts_complete": len(self._parts),
            "datasets": datasets,
        }
