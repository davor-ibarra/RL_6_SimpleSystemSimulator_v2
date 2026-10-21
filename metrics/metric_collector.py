from runtime_arrays import build_output_refs

class MetricCollector:
    def __init__(self, runtime, data_config, result_handler):
        output_schema = data_config["output_schema"]
        self.runtime = runtime
        self.result_handler = result_handler
        self.step_outputs = build_output_refs(runtime.step, output_schema["step_data"], "step_data")
        self.interval_outputs = build_output_refs(runtime.interval, output_schema["interval_data"], "interval_data")
        self.end_episode_signal_names = list(output_schema["end_episode_data"])
        for signal_name in self.end_episode_signal_names:
            if not hasattr(runtime.episode, signal_name):
                raise KeyError(f"Unknown end_episode_data output signal: {signal_name}")
        self.metric_signal_names = list(output_schema["metric_data"])
        for signal_name in self.metric_signal_names:
            if not hasattr(runtime.episode, signal_name):
                raise KeyError(f"Unknown metric_data output signal: {signal_name}")
        self.current_episode_idx = 0

    def start_episode(self, episode_idx):
        self.current_episode_idx = episode_idx

    def persist_episode(self, episode_idx, n_steps, n_intervals):
        if int(episode_idx) != self.current_episode_idx:
            raise ValueError("episode_id does not match the active metric collection episode")
        self.result_handler.save_episode(int(episode_idx), int(n_steps), int(n_intervals))

    def finalize(self):
        self.result_handler.finalize_run()
