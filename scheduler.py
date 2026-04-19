from torch import nn
import torch
import math
from typing import Literal


class NoiseScheduler(nn.Module):

    def __init__(self, beta_start: float, beta_end: float, timesteps: int, schedule: Literal["linear", "log-linear", "sigmoid", "cosine"] = "linear", offset: float = 0.008):
        super().__init__()

        SCHEDULE_MAPPING = {
            "linear": self._linear_schedule,
            "log-linear": self._log_linear_schedule,
            "cosine": self._cosine_schedule,
            "sigmoid": self._sigmoid_schedule,
        }

        assert schedule in SCHEDULE_MAPPING, f'The parameter "schedule" should be one of: {SCHEDULE_MAPPING.keys()}'

        self.cosine_offset = offset

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self.betas = None
        self.alphas = None
        self.alphas_cumprod = None

        SCHEDULE_MAPPING[schedule]()

        if not self.alphas:
            self.alphas = 1 - self.betas
        if not self.alphas_cumprod:
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)


    def _linear_schedule(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)


    def _log_linear_schedule(self):
        self.betas = torch.logspace(math.log10(self.beta_start), math.log10(self.beta_end), self.timesteps)


    def _cosine_schedule(self):
        assert self.cosine_offset, "Please set a non-zero value to offset if you have chosen the cosine noise schedule"

        t = torch.linspace(0, self.timesteps, self.timesteps + 1)
        T = self.timesteps
        s = self.cosine_offset

        f_t = torch.pow(torch.cos(((t/T) + s) / (1+s) * (torch.pi / 2)), 2)
        f_0 = f_t[0]

        self.alphas_cumprod = f_t / f_0
        self.alphas = self.alphas_cumprod[:-1] / self.alphas_cumprod[1:]
        self.betas = torch.clamp(1 - self.alphas, min=0.000, max=0.999)


    def _sigmoid_schedule(self):
        t = torch.linspace(0, self.timesteps, self.timesteps + 1)
        self.betas = self.beta_start + (self.beta_end - self.beta_start) * torch.sigmoid(((t / self.timesteps) - 0.05) / self.timesteps)


    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.IntTensor) -> torch.Tensor:
        return self.sqrt_alphas_cumprod[timestep] * sample + self.sqrt_one_minus_alphas_cumprod[timestep] * noise


    def prev_timestep(self, sample_at_time_t: torch.Tensor, pred_noise: torch.Tensor, timestep: torch.IntTensor):
        raise NotImplementedError

