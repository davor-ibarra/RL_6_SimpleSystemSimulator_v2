# RL_6 Simple System Simulator

Python simulator for a cart and inverted pendulum controlled by two PID controllers with tabular Q-learning for gain adaptation. The controllers regulate pendulum angle and cart position through a shared actuator. The system records physical trajectories, learning decisions, rewards and episode metrics for analysis.

This guide documents the [`v4_reward_colective` branch](https://github.com/davor-ibarra/RL_6_SimpleSystemSimulator_v2/tree/v4_reward_colective). Select that branch to use the implementation and configuration described here. The clone command below selects it automatically.

The supplementary figures, workbooks, prepared datasets and usage guide are available in the separate [supplementary materials repository](https://github.com/davor-ibarra/2q-pid-supplementary-materials).

The complete simulation outputs underlying the article are available from the corresponding author upon request because of their large size. Contact details are provided in the article.

## Installation and first run

Install Git and Python. The simulator was checked on Windows with Python 3.11.7. Run the commands below from a terminal.

```bash
git clone --branch v4_reward_colective https://github.com/davor-ibarra/RL_6_SimpleSystemSimulator_v2.git
cd RL_6_SimpleSystemSimulator_v2
```

Create a virtual environment and install the packages listed in `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --episodes 1 --output-root results_smoke
```

### Linux and macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py --episodes 1 --output-root results_smoke
```

The required packages are NumPy, pandas, Matplotlib, openpyxl, PyYAML, SciPy and PyArrow. The dependency file does not pin their versions.

The short run creates a new `run_*` folder beneath `results_smoke` and prints its path when finished. An episode can end at a physical limit during initial exploration. This is a recorded simulation outcome. A one-episode run checks the setup and output generation without assessing learning performance.

Run subsequent commands from the repository root. The examples below use the Windows virtual environment interpreter. On Linux and macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Running the simulator

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe main.py --episodes 100 --output-root results_custom
.\.venv\Scripts\python.exe main.py --config config/custom.yaml --episodes 100 --output-root results_custom
```

Create `config/custom.yaml` before using the last command, as explained below. Running without overrides uses 5,000 training episodes from the supplied configuration.

| Option | Purpose | Default |
| --- | --- | --- |
| `--config` | Main YAML configuration file | `config/config.yaml` |
| `--episodes` | Override `simulation.n_episodes` | Value in the configuration |
| `--output-root` | Parent directory for the new run folder | `data.output_root` |

Each invocation initializes new agents and creates a separate run folder. Within a training run, learned Q values persist across episodes while the physical state, controller memory and initial gains reset at the start of each episode.

## Configuration

Copy `config/config.yaml` to `config/custom.yaml` and edit the copy. Keep it inside `config/` so that its references to the visualization and summary configuration files remain valid. These references are resolved relative to the main configuration file.

The files are parsed as YAML, including the comments and flow-style mappings in the supplied main file.

| Configuration area | Controls |
| --- | --- |
| `experiment` | Random seed |
| `simulation` | Execution mode, episode count, duration and time steps |
| `termination` | Physical limits and stabilization criteria |
| `dynamic_system` | Plant parameters, initial state and normalization |
| `control` | PID objectives, initial gains, gain bounds, actuator mixing and saturation |
| `agent` | Gain agents, available actions and Q-learning settings |
| `metrics` | Interval metrics and episode reductions |
| `reward` | Local and cooperative reward settings |
| `data` | Output fields, Parquet chunk size and agent snapshots |
| `config_files` | Visualization and summary configuration filenames |

### Execution modes

Set `simulation.mode` in the main configuration.

| Mode | Behavior |
| --- | --- |
| `training` | Adapt PID gains through Q-learning and apply the configured exploration and learning-rate schedules |
| `fixed_pid` | Keep each controller's configured `initial_gains` and skip learning updates and schedules |

For a fixed PID experiment, set the mode to `fixed_pid` and edit `initial_gains` for both controllers in the copied configuration. Retain the complete configuration because metrics, rewards and output generation are used in both modes.

### Supplied settings

| Setting | Value |
| --- | --- |
| Random seed | 12345 |
| Execution mode | `training` |
| Episodes | 5,000 |
| Maximum episode duration | 10 s |
| Physical integration step | 0.001 s |
| Gain decision interval | 0.02 s |
| PID controllers | Pendulum angle and cart position |
| Initial PID gains | `kp = ki = kd = 1.0` for each controller |
| Gain bounds and increment | `[0, 5]` and `0.25` |
| Gain agents | Six, one per PID gain |
| Available actions | Decrease, maintain and increase |
| Shared actuator limits | `[-1, 1]` |
| Default output root | `results_history_CartPole` |
| Episodes per Parquet part | 250 |

Choose a decision interval that is an integer multiple of the integration step and an episode duration compatible with that step. When changing metric definitions, keep the referenced signals, output fields and summary settings consistent.

Figure selection and appearance are configured in `config/sub_config_visualization_CartPole.yaml`. Set `visualization.enabled` to `false` there to skip figures while retaining the data and summary. Workbook contents are configured in `config/sub_config_data_summary_CartPole.yaml`.

## Generated files

With the default output root, each run has the following structure. Files under `agent_state` and `figures` depend on the enabled settings.

```text
results_history_CartPole/
  run_YYYYMMDD_HHMM_microseconds/
    metadata.json
    summary.xlsx
    dataset_chunks/
      step_data/part-000000.parquet
      interval_data/part-000000.parquet
      end_episode_data/part-000000.parquet
      metric_data/part-000000.parquet
    agent_state/
      agent_state_episode_XXXXXX.json
      agent_state_episode_XXXXXX.xlsx
    figures/
      configured PNG files
```

| Dataset | Contents |
| --- | --- |
| `step_data` | Physical states and applied control at integration steps |
| `interval_data` | Gain decisions, rewards, learning values and interval metrics |
| `end_episode_data` | Episode outcomes and terminal states |
| `metric_data` | Episode-level scientific metrics |

Parquet datasets use ZSTD compression and split into additional numbered parts as the run grows. `metadata.json` records the effective configuration, seed, metric registry, persistence information and artifact status. Check that its overall `status` is `completed`. A value of `completed_with_artifact_errors` indicates that data were saved but an artifact such as a figure or workbook needs attention.

`summary.xlsx` contains `episode_summary` and `metadata`. When agent snapshots are available, it also contains `q_tables_final` and `q_visits_final`. Snapshot workbooks contain `q_tables`, `q_visits` and `agent_meta`. Snapshot filenames use zero-based episode indices. With the supplied settings, snapshots are saved every 2,000 episodes and at the end of the run.

### Reading the outputs in Python

The following example reads the outputs of a short run. Replace the example run folder name with the path printed by your execution.

```python
from pathlib import Path
import json
import pandas as pd

run = Path("results_smoke/run_YYYYMMDD_HHMM_microseconds")
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
episodes = pd.read_parquet(run / "dataset_chunks" / "end_episode_data")
metrics = pd.read_parquet(run / "dataset_chunks" / "metric_data")
summary = pd.read_excel(run / "summary.xlsx", sheet_name="episode_summary")

print(metadata["status"])
print(episodes.head())
print(metrics.head())
print(summary.head())
```

Passing a dataset directory to `read_parquet` reads its numbered parts together. Reading the full `step_data` dataset can require substantial memory for long experiments. Use individual part files and select the columns needed for the analysis.

## Source layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Load configuration and assemble a run |
| `simulation_manager.py` | Execute episodes and coordinate the control and learning cycle |
| `dynamic_system/` | Nonlinear cart and pendulum dynamics |
| `controllers/` | PID calculation, gain updates and actuator mixing |
| `agents/` | Tabular Q-learning agents for PID gains |
| `rewards/` | Reward calculation |
| `metrics/` | Signal collection, interval processing and episode reports |
| `runtime_arrays.py` | Runtime array allocation |
| `result_handler.py` | Run metadata, snapshots, summaries and artifact coordination |
| `parquet_dataset_writer.py` | Chunked dataset persistence |
| `stabilization.py` | Stabilization criteria |
| `visualization_manager.py` | Configured Matplotlib figures |
| `config/` | Main configuration and visualization and summary settings |

## Article materials and reproducibility

This repository provides the simulator and its current configurable execution path. Each run stores its effective settings and seed in `metadata.json`. Record the code commit and Python package versions alongside those outputs when comparing experiments.

The main entry point starts a fresh run and does not load saved Q tables for resumed training or frozen-policy evaluation. The supplied configuration describes one experiment. Reproducing the article's full experimental design requires its corresponding configurations, seeds and evaluation procedures.

Use the [supplementary materials repository](https://github.com/davor-ibarra/2q-pid-supplementary-materials) for the approved article materials and the instructions for reading them. Generated run outputs, local analyses, tests and internal specifications are excluded from this source repository by `.gitignore`.
