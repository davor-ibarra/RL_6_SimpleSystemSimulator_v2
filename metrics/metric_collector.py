from runtime_arrays import build_output_refs


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


class MetricCollector:
    def __init__(self, runtime, data_config, result_handler):
        output_schema = data_config["output_schema"]
        self.runtime = runtime
        self.result_handler = result_handler
        self.step_outputs = build_output_refs(runtime.step, output_schema["step_data"], "step_data")
        self.interval_outputs = build_output_refs(runtime.interval, output_schema["interval_data"], "interval_data")
        self.end_episode_signal_names = list(output_schema["end_episode_data"])
        for signal_name in runtime.metric_report_signal_names:
            if signal_name not in self.end_episode_signal_names:
                self.end_episode_signal_names.append(signal_name)
        for signal_name in self.end_episode_signal_names:
            if not hasattr(runtime.episode, signal_name):
                raise KeyError(f"Unknown end_episode_data output signal: {signal_name}")
        self.current_episode_idx = 0

    def start_episode(self, episode_idx):
        self.current_episode_idx = episode_idx

    def persist_episode(self, episode_idx, n_steps, n_intervals):
        step_data = {"n_step": int(n_steps)}
        for output in self.step_outputs:
            step_data[output.name] = output.array[:n_steps].tolist()

        interval_data = {"n_interval": int(n_intervals)}
        for output in self.interval_outputs:
            interval_data[output.name] = output.array[:n_intervals].tolist()

        end_episode_data = {}
        for signal_name in self.end_episode_signal_names:
            end_episode_data[signal_name] = _json_scalar(getattr(self.runtime.episode, signal_name))

        episode_output = {
            "episode_id": int(episode_idx),
            "step_data": step_data,
            "interval_data": interval_data,
            "end_episode_data": end_episode_data,
        }
        self.result_handler.save_episode(episode_output)

    def finalize(self):
        self.result_handler.finalize_run()
