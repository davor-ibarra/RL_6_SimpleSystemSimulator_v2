from runtime_arrays import reset_runtime_arrays


class SimulationManager:
    def __init__(
        self,
        runtime,
        config,
        dynamic_system,
        controller_manager,
        agent,
        metric_processor,
        reward_calculator,
        metric_collector,
        result_handler,
    ):
        self.runtime = runtime
        self.step = runtime.step
        self.interval = runtime.interval
        self.episode = runtime.episode
        self.config = config
        self.dynamic_system = dynamic_system
        self.controller_manager = controller_manager
        self.agent = agent
        self.metric_processor = metric_processor
        self.reward_calculator = reward_calculator
        self.metric_collector = metric_collector
        self.result_handler = result_handler

        simulation_config = config["simulation"]
        self.n_episodes = int(simulation_config["n_episodes"])
        self.episode_duration_sec = float(simulation_config["episode_duration_sec"])
        self.dt_sec = float(simulation_config["dt_sec"])
        self.decision_interval_sec = float(simulation_config["decision_interval_sec"])
        if self.n_episodes <= 0:
            raise ValueError("n_episodes must be positive")
        if self.episode_duration_sec <= 0.0:
            raise ValueError("episode_duration_sec must be positive")
        if self.dt_sec <= 0.0:
            raise ValueError("dt_sec must be positive")
        if self.decision_interval_sec <= 0.0:
            raise ValueError("decision_interval_sec must be positive")

        self.max_steps = int(round(self.episode_duration_sec / self.dt_sec))
        self.steps_per_interval = max(1, int(round(self.decision_interval_sec / self.dt_sec)))
        self.termination_config = config["termination"]
        self.reason_codes = self.termination_config["reason_codes"]
        self.reason_names_by_code = {}
        for reason_name, reason_code in self.reason_codes.items():
            self.reason_names_by_code[int(reason_code)] = reason_name
        self.data_config = config["data"]
        self.save_agent_state_enabled = bool(self.data_config["save_agent_state"])
        self.agent_state_save_frequency = int(self.data_config["agent_state_save_frequency"])
        if self.agent_state_save_frequency <= 0:
            raise ValueError("agent_state_save_frequency must be positive")

        self.controller_reward_totals = []
        self.controller_progress_items = []
        for controller_config in config["control"]["controllers"]:
            controller_id = controller_config["id"]
            reward_array = getattr(self.interval, f"reward_controller_{controller_id}")
            total_reward_signal = f"total_reward_{controller_id}"
            self.controller_reward_totals.append(
                (
                    reward_array,
                    total_reward_signal,
                )
            )
            self.controller_progress_items.append(
                (
                    controller_id,
                    total_reward_signal,
                    getattr(self.step, f"kp_{controller_id}"),
                    getattr(self.step, f"ki_{controller_id}"),
                    getattr(self.step, f"kd_{controller_id}"),
                )
            )

        self.current_step_idx = 0
        self.current_interval_idx = 0
        self.terminated = False
        self.termination_reason_code = int(self.reason_codes["none"])

    def run(self):
        for episode_idx in range(self.n_episodes):
            self.run_episode(episode_idx)
        self.metric_collector.finalize()

    def run_episode(self, episode_idx):
        reset_runtime_arrays(self.runtime)
        self.terminated = False
        self.termination_reason_code = int(self.reason_codes["none"])
        self.current_step_idx = 0
        self.current_interval_idx = 0

        self.dynamic_system.reset_episode(0)
        self.controller_manager.reset_episode(0)
        self.agent.reset_episode()
        self.metric_processor.reset_episode()
        self.reward_calculator.reset_episode()
        self.metric_collector.start_episode(episode_idx)

        while not self.terminated and self.current_step_idx < self.max_steps:
            self.run_interval(self.current_interval_idx)
            self.current_interval_idx += 1

        self.close_episode(episode_idx)

    def run_interval(self, interval_idx):
        interval_start_step = self.current_step_idx
        self.interval.interval_id[interval_idx] = interval_idx
        self.interval.interval_start_step_idx[interval_idx] = interval_start_step
        self.interval.interval_t_start_sec[interval_idx] = self.step.t_sec[self.current_step_idx]

        self.agent.select_actions(interval_idx)
        self.controller_manager.apply_gain_actions(interval_idx)

        steps_in_interval = 0
        while steps_in_interval < self.steps_per_interval and self.current_step_idx < self.max_steps:
            control_step_idx = self.current_step_idx
            next_state_step_idx = control_step_idx + 1
            self.run_step(control_step_idx, next_state_step_idx)
            self.current_step_idx = next_state_step_idx
            steps_in_interval += 1
            if self.check_termination(next_state_step_idx):
                break
        
        if not self.terminated and self.current_step_idx >= self.max_steps:
            self.terminated = True
            self.termination_reason_code = int(self.reason_codes["time_limit"])

        interval_end_step = self.current_step_idx - 1
        self.interval.interval_end_step_idx[interval_idx] = interval_end_step
        self.interval.interval_n_steps[interval_idx] = steps_in_interval

        self.interval.terminated[interval_idx] = 1.0 if self.terminated else 0.0
        self.interval.termination_reason_code[interval_idx] = self.termination_reason_code

        self.metric_processor.calculate_metrics(interval_idx, interval_start_step, interval_end_step)
        self.metric_processor.aggregate_interval(interval_idx, interval_start_step, interval_end_step)
        self.reward_calculator.evaluate_interval(interval_idx)
        
        terminal_for_learning = self.is_terminal_for_learning()
        self.agent.learn(interval_idx, terminal_for_learning)

    def run_step(self, control_step_idx, next_state_step_idx):
        u_total = self.controller_manager.compute_control_step(control_step_idx)
        self.dynamic_system.integrate_step(next_state_step_idx, u_total, self.dt_sec)

    def is_terminal_for_learning(self):
        reason_code = int(self.termination_reason_code)
        return reason_code in (int(self.reason_codes["limit_exceeded"]), int(self.reason_codes["stabilization_success"]))

    def check_termination(self, step_idx):
        boundary = self.termination_config["boundary"]
        stabilization = self.termination_config["stabilization"]

        cart_position = self.step.cart_position_raw[step_idx]
        cart_velocity = self.step.cart_velocity_raw[step_idx]
        pendulum_angle = self.step.pendulum_angle_raw[step_idx]
        pendulum_velocity = self.step.pendulum_velocity_raw[step_idx]

        if (
            cart_position < float(boundary["cart_position"][0])
            or cart_position > float(boundary["cart_position"][1])
            or pendulum_angle < float(boundary["pendulum_angle"][0])
            or pendulum_angle > float(boundary["pendulum_angle"][1])
        ):
            self.terminated = True
            self.termination_reason_code = int(self.reason_codes["limit_exceeded"])
            return True

        if (
            float(stabilization["pendulum_angle"][0]) <= pendulum_angle <= float(stabilization["pendulum_angle"][1])
            and float(stabilization["pendulum_velocity"][0])
            <= pendulum_velocity
            <= float(stabilization["pendulum_velocity"][1])
            and float(stabilization["cart_position"][0]) <= cart_position <= float(stabilization["cart_position"][1])
            and float(stabilization["cart_velocity"][0]) <= cart_velocity <= float(stabilization["cart_velocity"][1])
        ):
            self.terminated = True
            self.termination_reason_code = int(self.reason_codes["stabilization_success"])
            return True

        if step_idx >= self.max_steps:
            self.terminated = True
            self.termination_reason_code = int(self.reason_codes["time_limit"])
            return True

        return False

    def close_episode(self, episode_idx):
        n_intervals = self.current_interval_idx
        final_step_idx = self.current_step_idx
        last_control_step_idx = max(0, final_step_idx - 1)
        n_steps = last_control_step_idx + 1

        self.episode.end_terminated = bool(self.terminated)
        self.episode.end_termination_reason_code = int(self.termination_reason_code)
        self.episode.end_termination_reason = self.reason_names_by_code[int(self.termination_reason_code)]
        self.episode.final_t_sec = float(self.step.t_sec[final_step_idx])
        self.episode.final_cart_position_raw = float(self.step.cart_position_raw[final_step_idx])
        self.episode.final_cart_velocity_raw = float(self.step.cart_velocity_raw[final_step_idx])
        self.episode.final_pendulum_angle_raw = float(self.step.pendulum_angle_raw[final_step_idx])
        self.episode.final_pendulum_velocity_raw = float(self.step.pendulum_velocity_raw[final_step_idx])
        self.episode.final_cart_position_norm = float(self.step.cart_position_norm[final_step_idx])
        self.episode.final_cart_velocity_norm = float(self.step.cart_velocity_norm[final_step_idx])
        self.episode.final_pendulum_angle_norm = float(self.step.pendulum_angle_norm[final_step_idx])
        self.episode.final_pendulum_velocity_norm = float(self.step.pendulum_velocity_norm[final_step_idx])
        self.episode.total_reward = float(self.interval.reward_global_signed[:n_intervals].sum())
        for reward_array, episode_signal_name in self.controller_reward_totals:
            setattr(self.episode, episode_signal_name, float(reward_array[:n_intervals].sum()))
        for controller_id, total_reward_signal, kp_array, ki_array, kd_array in self.controller_progress_items:
            del total_reward_signal
            setattr(self.episode, f"final_kp_{controller_id}", float(kp_array[last_control_step_idx]))
            setattr(self.episode, f"final_ki_{controller_id}", float(ki_array[last_control_step_idx]))
            setattr(self.episode, f"final_kd_{controller_id}", float(kd_array[last_control_step_idx]))
        self.episode.total_agent_decisions = int(n_intervals * len(self.agent.tracks))

        self.episode.epsilon = float(self.agent.epsilon)
        self.episode.learning_rate = float(self.agent.learning_rate)
        self.agent.decay_learning_parameters()

        self.metric_collector.persist_episode(episode_idx, n_steps, n_intervals)

        final_episode = episode_idx == self.n_episodes - 1
        frequency_episode = (episode_idx + 1) % self.agent_state_save_frequency == 0
        if self.save_agent_state_enabled and (frequency_episode or final_episode):
            self.result_handler.save_agent_state(episode_idx, self.agent.export_state())

        self.print_episode_progress(episode_idx, final_step_idx)

    def print_episode_progress(self, episode_idx, final_step_idx):
        duration_sec = float(self.episode.final_t_sec)
        reason_name = self.episode.end_termination_reason
        reward_parts = []
        gain_parts = []
        for controller_id, total_reward_signal, kp_array, ki_array, kd_array in self.controller_progress_items:
            del kp_array, ki_array, kd_array
            reward_value = float(getattr(self.episode, total_reward_signal))
            reward_parts.append(f"{controller_id}={reward_value:.3f}")
            gain_parts.append(
                f"{controller_id}(kp={getattr(self.episode, f'final_kp_{controller_id}'):.3f}, "
                f"ki={getattr(self.episode, f'final_ki_{controller_id}'):.3f}, "
                f"kd={getattr(self.episode, f'final_kd_{controller_id}'):.3f})"
            )
        print(
            f"Episode {episode_idx + 1}/{self.n_episodes} | "
            f"duration={duration_sec:.3f}s | "
            f"termination={reason_name} | "
            f"controller_rewards: {', '.join(reward_parts)} | ",
            flush=True,
        )
        print(
            f"final_gains: {'; '.join(gain_parts)}",
            flush=True,
        )
