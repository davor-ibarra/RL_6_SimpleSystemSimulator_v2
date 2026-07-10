import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from agents.pid_qlearning_agent import PIDQLearningAgent
from controllers.controller_manager import ControllerManager
from dynamic_system.cart_pole_dynamic_system import CartPoleDynamicSystem
from metrics.metric_collector import MetricCollector
from metrics.metric_processing import MetricProcessor
from result_handler import ResultHandler
from rewards.reward_calculator import RewardCalculator
from runtime_arrays import build_runtime_arrays
from simulation_manager import SimulationManager


def load_mapping_file(config_path):
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded_config = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded_config = json.loads(text)
    return loaded_config


def load_config(config_path):
    config_file = Path(config_path)
    loaded_config = load_mapping_file(config_file)
    config_files = loaded_config["config_files"]
    config_dir = config_file.parent

    visualization_config = load_mapping_file(config_dir / config_files["visualization"])
    data_summary_config = load_mapping_file(config_dir / config_files["data_summary"])

    loaded_config["visualization"] = visualization_config["visualization"]
    loaded_config["data"]["summary"] = data_summary_config["data_summary"]
    return loaded_config


def build_capacities(config):
    simulation = config["simulation"]
    episode_duration_sec = float(simulation["episode_duration_sec"])
    dt_sec = float(simulation["dt_sec"])
    decision_interval_sec = float(simulation["decision_interval_sec"])
    if dt_sec <= 0.0:
        raise ValueError("dt_sec must be positive")
    if decision_interval_sec <= 0.0:
        raise ValueError("decision_interval_sec must be positive")
    max_steps = int(round(episode_duration_sec / dt_sec))
    steps_per_interval = max(1, int(round(decision_interval_sec / dt_sec)))
    step_capacity = max_steps + 1
    interval_capacity = int(math.ceil(max_steps / steps_per_interval)) + 1
    return step_capacity, interval_capacity


def build_output_dir(config, run_id):
    output_root = Path(config["data"]["output_root"])
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def apply_cli_overrides(config, args):
    if args.episodes is not None:
        config["simulation"]["n_episodes"] = int(args.episodes)
    if args.output_root is not None:
        config["data"]["output_root"] = str(args.output_root)


def main():
    parser = argparse.ArgumentParser(description="CartPole RL+Control simple refactor runner")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_cli_overrides(config, args)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M")
    output_dir = build_output_dir(config, run_id)

    step_capacity, interval_capacity = build_capacities(config)
    runtime = build_runtime_arrays(
        step_capacity=step_capacity,
        interval_capacity=interval_capacity,
        config=config,
    )

    result_handler = ResultHandler(output_dir, config["data"], config["termination"])
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "step_capacity": step_capacity,
        "interval_capacity": interval_capacity,
        "config": config,
    }
    result_handler.save_metadata(metadata)

    dynamic_system = CartPoleDynamicSystem(runtime, config["dynamic_system"])
    controller_manager = ControllerManager(
        runtime,
        config["control"],
        config["agent"],
        config["simulation"]["dt_sec"],
    )
    agent = PIDQLearningAgent(runtime, config["agent"], config["control"])
    metric_processor = MetricProcessor(runtime, config["metrics"], config["simulation"]["dt_sec"])
    reward_calculator = RewardCalculator(runtime, config["reward"], config["agent"], config["termination"])
    metric_collector = MetricCollector(runtime, config["data"], result_handler)

    simulation_manager = SimulationManager(
        runtime=runtime,
        config=config,
        dynamic_system=dynamic_system,
        controller_manager=controller_manager,
        agent=agent,
        metric_processor=metric_processor,
        reward_calculator=reward_calculator,
        metric_collector=metric_collector,
        result_handler=result_handler,
    )
    simulation_manager.run()

    if bool(config["visualization"]["enabled"]):
        from visualization_manager import VisualizationManager

        visualization_manager = VisualizationManager(output_dir, config["visualization"])
        visualization_manager.render_post_run(result_handler.summary_rows, result_handler.latest_agent_state)

    print(f"Run complete: {output_dir}")


if __name__ == "__main__":
    main()
