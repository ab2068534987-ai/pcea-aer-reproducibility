from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PowerBreakdown:
    idle: float = 0.0
    cpu: float = 0.0
    gpu: float = 0.0
    interaction_signed: float = 0.0

    @property
    def total(self) -> float:
        return self.idle + self.cpu + self.gpu + self.interaction_signed

    def scale(self, dt: float) -> "PowerBreakdown":
        return PowerBreakdown(
            idle=self.idle * dt,
            cpu=self.cpu * dt,
            gpu=self.gpu * dt,
            interaction_signed=self.interaction_signed * dt,
        )


def coupled_power(
    p_idle: float,
    p_cpu0: float,
    p_gpu0: float,
    k1: float,
    k2: float,
    u_cpu: float,
    u_gpu: float,
    active: bool = True,
) -> PowerBreakdown:
    """Whole-server power model with a signed CPU-GPU interaction residual.

    If active is False, returns zero. This is the active-server energy convention:
    unused machines do not contribute to active energy, while machines inside their
    active window contribute idle power even when no task is running.
    """
    if not active:
        return PowerBreakdown()
    u_cpu = max(0.0, min(1.0, float(u_cpu)))
    u_gpu = max(0.0, min(1.0, float(u_gpu)))
    cpu = p_cpu0 * u_cpu
    gpu = p_gpu0 * u_gpu
    cross = u_cpu * u_gpu
    interaction = k1 * p_cpu0 * p_gpu0 * cross + k2 * p_cpu0 * p_gpu0 * (cross**2)
    return PowerBreakdown(idle=p_idle, cpu=cpu, gpu=gpu, interaction_signed=interaction)
