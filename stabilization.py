from dataclasses import dataclass


@dataclass
class StabilizationTrack:
    controller_id: str
    variables: tuple
    inside_array: object
    duration_array: object
    qualified_array: object
    consecutive_steps: int = 0
    inside_bands: bool = False
    qualified: bool = False


class PhysicalStabilization:
    def __init__(self, runtime, control_config, termination_config, dt_sec):
        stabilization_config = termination_config["stabilization"]
        self.mode = stabilization_config["mode"]
        self.persistence_sec = float(stabilization_config["persistence_sec"])
        self.dt_sec = dt_sec
        self.tracks = []
        for controller_config in control_config["controllers"]:
            controller_id = controller_config["id"]
            variables = tuple(
                (
                    getattr(runtime.step, f"{variable}_raw"),
                    float(stabilization_config[variable][0]),
                    float(stabilization_config[variable][1]),
                )
                for variable in controller_config["vars"]
            )
            self.tracks.append(
                StabilizationTrack(
                    controller_id=controller_id,
                    variables=variables,
                    inside_array=getattr(runtime.interval, f"terminal_inside_bands_{controller_id}"),
                    duration_array=getattr(runtime.interval, f"terminal_band_duration_sec_{controller_id}"),
                    qualified_array=getattr(runtime.interval, f"terminal_band_qualified_{controller_id}"),
                )
            )

    def reset_episode(self):
        for track in self.tracks:
            track.consecutive_steps = 0
            track.inside_bands = False
            track.qualified = False

    def update(self, step_idx):
        for track in self.tracks:
            track.inside_bands = all(
                lower <= values[step_idx] <= upper
                for values, lower, upper in track.variables
            )
            if track.inside_bands:
                track.consecutive_steps += 1
            else:
                track.consecutive_steps = 0
            if self.mode == "instantaneous":
                track.qualified = track.inside_bands
            elif self.mode == "persistence":
                track.qualified = (
                    track.inside_bands
                    and track.consecutive_steps * self.dt_sec >= self.persistence_sec
                )

    def jointly_qualified(self):
        return all(track.qualified for track in self.tracks)

    def write_interval(self, interval_idx):
        for track in self.tracks:
            track.inside_array[interval_idx] = float(track.inside_bands)
            track.duration_array[interval_idx] = track.consecutive_steps * self.dt_sec
            track.qualified_array[interval_idx] = float(track.qualified)
