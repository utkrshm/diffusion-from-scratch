# noqa: INP001
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Literal

from scheduler import NoiseScheduler


def plot_scheduler(
    plot_axis: Axes,
    scheduler_type: Literal["linear", "log-linear", "sigmoid", "cosine"],
    total_timesteps: int,
    timesteps_to_show: torch.IntTensor,
    original_data: torch.Tensor,
    noise: torch.Tensor,
) -> None:
    _scheduler = NoiseScheduler(0.0001, 0.002, total_timesteps, scheduler_type)
    x_t = _scheduler.add_noise(original_data, noise, timesteps_to_show)

    signal_length = original_data.shape[-1]
    x_axis = list(range(signal_length))

    # Plot the uncorrupted signal (first batch element).
    plot_axis.plot(x_axis, original_data[0].tolist(), color="#4263eb", linewidth=3, label="x_0 (Original)")

    colors = ["#12b886", "#fab005", "#f76707", "#f03e3e", "#ae3ec9"]

    for i, t_val in enumerate(timesteps_to_show):
        # Note: alphas_cumprod[t_val] corresponds to the math step t = t_val + 1
        label = f"x_{t_val + 1}"
        plot_axis.plot(
            x_axis,
            x_t[i].tolist(),
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )

    plot_axis.set_title(f"Schedule: {scheduler_type}")
    plot_axis.grid(alpha=0.2)
    plot_axis.legend(loc="upper right", fontsize=8)


if __name__ == "__main__":
    signal_length = 100
    x = torch.sin(torch.linspace(0, 4*torch.pi, signal_length))
    x = x.float().unsqueeze(0)      # Add batch dimension

    total_timesteps = 1000
    timesteps_to_show = torch.IntTensor([0, 100, 250, 500, 750, 999])

    # Batch the x tensor
    x0 = x.repeat(len(timesteps_to_show), 1)
    eps = torch.randn((len(timesteps_to_show), *x.shape[1:]))

    fig, ax = plt.subplots(2, 2)
    axes = ax.flatten()

    schedules = ["linear", "log-linear", "sigmoid", "cosine"]

    for ax, sch_type in zip(axes, schedules, strict=True):
        plot_scheduler(ax, sch_type, total_timesteps, timesteps_to_show, x0, eps)

    plt.show()
