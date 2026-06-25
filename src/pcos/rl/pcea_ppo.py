from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except Exception as exc:  # pragma: no cover
    torch = None
    nn = object
    F = None
    Categorical = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class PCEAConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 32
    grad_clip: float = 1.0
    lambda_power_init: float = 0.1
    lambda_deadline_init: float = 1.0
    dual_lr_power: float = 0.02
    dual_lr_deadline: float = 0.05
    epsilon_power_violation: float = 0.02
    epsilon_deadline_miss: float = 0.01
    max_power_violation_soft: float = 0.02
    lambda_max: float = 10.0
    lambda_power_max: float = 10.0
    lambda_deadline_max: float = 10.0
    scalar_gae: bool = False
    fixed_dual: bool = False
    cpeg_enabled: bool = False
    cpeg_logit_penalty: float = 8.0
    anti_idle_enabled: bool = False
    anti_idle_logit_penalty: float = 4.0
    energy_guard_enabled: bool = False
    energy_guard_logit_bonus: float = 3.0
    activation_guard_enabled: bool = False
    activation_logit_penalty: float = 4.0
    deterministic_repair_enabled: bool = False


def _require_torch():
    if torch is None:
        raise RuntimeError("PCEA-PPO requires torch. Install with `pip install torch`.") from _TORCH_IMPORT_ERROR


class PairActorCritic(nn.Module):
    """Shared global encoder with per-pair actor and three value heads."""

    def __init__(self, global_dim: int, pair_dim: int, hidden: int = 128):
        _require_torch()
        super().__init__()
        self.global_encoder = nn.Sequential(nn.Linear(global_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pair_encoder = nn.Sequential(nn.Linear(pair_dim + hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pair_head = nn.Linear(hidden, 1)
        self.defer_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.value_energy = nn.Linear(hidden, 1)
        self.value_power = nn.Linear(hidden, 1)
        self.value_deadline = nn.Linear(hidden, 1)
        self.scalar_value = nn.Linear(hidden, 1)

    def forward(self, obs: Dict[str, np.ndarray]):
        g = torch.as_tensor(obs["global"], dtype=torch.float32).unsqueeze(0)
        pairs = torch.as_tensor(obs["pairs"], dtype=torch.float32)
        mask = torch.as_tensor(obs["mask"], dtype=torch.float32)
        h = self.global_encoder(g).squeeze(0)
        if pairs.numel() > 0:
            h_rep = h.unsqueeze(0).expand(pairs.shape[0], -1)
            pair_h = self.pair_encoder(torch.cat([pairs, h_rep], dim=-1))
            pair_logits = self.pair_head(pair_h).squeeze(-1)
        else:
            pair_logits = torch.empty((0,), dtype=torch.float32)
        logits = pair_logits
        if len(mask) > len(pair_logits):
            logits = torch.cat([pair_logits, self.defer_head(h).view(1)], dim=0)
        # Hard mask for illegal actions.
        logits = logits.masked_fill(mask <= 0, -1e9)
        values = {
            "energy": self.value_energy(h).squeeze(-1),
            "power": self.value_power(h).squeeze(-1),
            "deadline": self.value_deadline(h).squeeze(-1),
            "scalar": self.scalar_value(h).squeeze(-1),
        }
        return logits, values


class RolloutBuffer:
    def __init__(self):
        self.rows: List[dict] = []

    def add(self, **kwargs) -> None:
        self.rows.append(kwargs)

    def clear(self) -> None:
        self.rows.clear()

    def __len__(self):
        return len(self.rows)


def compute_gae(costs, values, dones, gamma, lam):
    adv = np.zeros(len(costs), dtype=np.float32)
    last = 0.0
    next_value = 0.0
    for t in reversed(range(len(costs))):
        nonterminal = 1.0 - float(dones[t])
        delta = costs[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        next_value = values[t]
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def norm_adv(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return a
    return (a - a.mean()) / (a.std() + 1e-8)


class PCEAPPOAgent:
    def __init__(self, global_dim: int, pair_dim: int, config: PCEAConfig | None = None, hidden: int = 128, device: str = "cpu"):
        _require_torch()
        self.cfg = config or PCEAConfig()
        self.device = torch.device(device)
        self.model = PairActorCritic(global_dim, pair_dim, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.lambda_power = self.cfg.lambda_power_init
        self.lambda_deadline = self.cfg.lambda_deadline_init

    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False):
        self.model.eval()
        with torch.no_grad():
            mask = np.asarray(obs.get("mask", []), dtype=np.float32)
            if len(mask) == 0:
                raise RuntimeError("PCEA-PPO received an empty action mask. This indicates env did not ensure actionability.")
            if float(mask.sum()) <= 0.0:
                raise RuntimeError("PCEA-PPO received no positive legal actions in the action mask. This indicates env did not ensure actionability.")
            logits, values = self.model(obs)
            logits = self._apply_cpeg_penalty(logits, obs)
            logits = self._apply_anti_idle_penalty(logits, obs)
            logits = self._apply_energy_guard_penalty(logits, obs)
            logits = self._apply_activation_guard_penalty(logits, obs)
            if logits.numel() == 0:
                raise RuntimeError("PCEA-PPO received zero legal actions. This indicates env did not ensure actionability.")
            dist = Categorical(logits=logits)
            action = torch.argmax(logits) if deterministic else dist.sample()
            if deterministic and self.cfg.deterministic_repair_enabled:
                action = self._repair_action(action, logits, obs)
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), {k: float(v.item()) for k, v in values.items()}

    def _apply_cpeg_penalty(self, logits, obs: Dict[str, np.ndarray]):
        if not self.cfg.cpeg_enabled or logits.numel() == 0:
            return logits
        flags = obs.get("pair_cpeg_slow_critical")
        if flags is None:
            return logits
        flag_t = torch.as_tensor(flags, dtype=torch.float32, device=logits.device)
        n = min(int(flag_t.numel()), int(logits.numel()))
        if n <= 0:
            return logits
        out = logits.clone()
        out[:n] = out[:n] - self.cfg.cpeg_logit_penalty * flag_t[:n]
        return out

    def _apply_anti_idle_penalty(self, logits, obs: Dict[str, np.ndarray]):
        if not self.cfg.anti_idle_enabled or logits.numel() == 0:
            return logits
        flags = obs.get("pair_anti_idle_slow_finish")
        if flags is None:
            return logits
        flag_t = torch.as_tensor(flags, dtype=torch.float32, device=logits.device)
        n = min(int(flag_t.numel()), int(logits.numel()))
        if n <= 0:
            return logits
        out = logits.clone()
        out[:n] = out[:n] - self.cfg.anti_idle_logit_penalty * flag_t[:n]
        return out

    def _apply_energy_guard_penalty(self, logits, obs: Dict[str, np.ndarray]):
        if not self.cfg.energy_guard_enabled or logits.numel() == 0:
            return logits
        flags = obs.get("pair_is_energy_guard_preferred")
        if flags is None:
            return logits
        flag_t = torch.as_tensor(flags, dtype=torch.float32, device=logits.device)
        n = min(int(flag_t.numel()), int(logits.numel()))
        if n <= 0:
            return logits
        out = logits.clone()
        if float(flag_t[:n].sum().item()) > 0.0:
            out[:n] = out[:n] - self.cfg.energy_guard_logit_bonus * (1.0 - flag_t[:n])
        return out

    def _apply_activation_guard_penalty(self, logits, obs: Dict[str, np.ndarray]):
        if not self.cfg.activation_guard_enabled or logits.numel() == 0:
            return logits
        flags = obs.get("pair_activation_guard_penalized")
        if flags is None:
            return logits
        flag_t = torch.as_tensor(flags, dtype=torch.float32, device=logits.device)
        n = min(int(flag_t.numel()), int(logits.numel()))
        if n <= 0:
            return logits
        out = logits.clone()
        out[:n] = out[:n] - self.cfg.activation_logit_penalty * flag_t[:n]
        return out

    def _repair_action(self, action, logits, obs: Dict[str, np.ndarray]):
        pair_penalty = obs.get("pair_energy_guard_penalized")
        activation_penalty = obs.get("pair_activation_guard_penalized")
        if pair_penalty is None and activation_penalty is None:
            return action
        action_idx = int(action.item())
        pair_count = 0
        if pair_penalty is not None:
            pair_count = max(pair_count, int(np.asarray(pair_penalty).shape[0]))
        if activation_penalty is not None:
            pair_count = max(pair_count, int(np.asarray(activation_penalty).shape[0]))
        if action_idx >= pair_count:
            return action
        energy_pen = np.asarray(pair_penalty if pair_penalty is not None else np.zeros(pair_count, dtype=np.float32), dtype=np.float32)
        act_pen = np.asarray(activation_penalty if activation_penalty is not None else np.zeros(pair_count, dtype=np.float32), dtype=np.float32)
        if energy_pen[action_idx] <= 0.0 and act_pen[action_idx] <= 0.0:
            return action
        mask = np.asarray(obs.get("mask", []), dtype=np.float32)[:pair_count] > 0.0
        if mask.shape[0] != pair_count:
            return action
        candidate_sets = [
            mask & (energy_pen <= 0.0) & (act_pen <= 0.0),
            mask & (act_pen <= 0.0),
            mask & (energy_pen <= 0.0),
        ]
        replacement = None
        for cand in candidate_sets:
            if not np.any(cand):
                continue
            cand_t = torch.as_tensor(cand, dtype=torch.bool, device=logits.device)
            repair_logits = logits[:pair_count].masked_fill(~cand_t, float("-inf"))
            replacement = int(torch.argmax(repair_logits).item())
            break
        if replacement is None:
            return action
        return torch.as_tensor(replacement, dtype=torch.long, device=logits.device)

    def update_dual(self, mean_power_violation: float, mean_deadline_miss: float) -> None:
        if self.cfg.fixed_dual:
            return
        power_excess = mean_power_violation - self.cfg.max_power_violation_soft
        self.lambda_power = float(np.clip(self.lambda_power + self.cfg.dual_lr_power * power_excess, 0.0, self.cfg.lambda_power_max))
        self.lambda_deadline = float(np.clip(self.lambda_deadline + self.cfg.dual_lr_deadline * (mean_deadline_miss - self.cfg.epsilon_deadline_miss), 0.0, self.cfg.lambda_deadline_max))

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if not buffer.rows:
            return {}
        rows = buffer.rows
        cE = np.array([r["cost_energy"] for r in rows], dtype=np.float32)
        cP = np.array([r["cost_power"] for r in rows], dtype=np.float32)
        cD = np.array([r["cost_deadline"] for r in rows], dtype=np.float32)
        vE = np.array([r["value_energy"] for r in rows], dtype=np.float32)
        vP = np.array([r["value_power"] for r in rows], dtype=np.float32)
        vD = np.array([r["value_deadline"] for r in rows], dtype=np.float32)
        vS = np.array([r["value_scalar"] for r in rows], dtype=np.float32)
        dones = np.array([r["done"] for r in rows], dtype=np.float32)
        aE, rE = compute_gae(cE, vE, dones, self.cfg.gamma, self.cfg.gae_lambda)
        aP, rP = compute_gae(cP, vP, dones, self.cfg.gamma, self.cfg.gae_lambda)
        aD, rD = compute_gae(cD, vD, dones, self.cfg.gamma, self.cfg.gae_lambda)
        if self.cfg.scalar_gae:
            scalar_cost = cE + self.lambda_power * cP + self.lambda_deadline * cD
            aS, rS = compute_gae(scalar_cost, vS, dones, self.cfg.gamma, self.cfg.gae_lambda)
            adv_policy = -norm_adv(aS)
        else:
            adv_policy = -norm_adv(aE) - self.lambda_deadline * norm_adv(aD) - self.lambda_power * norm_adv(aP)
            rS = rE + self.lambda_power * rP + self.lambda_deadline * rD
        old_logp = torch.as_tensor([r["logp"] for r in rows], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([r["action"] for r in rows], dtype=torch.long, device=self.device)
        adv_t = torch.as_tensor(adv_policy, dtype=torch.float32, device=self.device)
        retE = torch.as_tensor(rE, dtype=torch.float32, device=self.device)
        retP = torch.as_tensor(rP, dtype=torch.float32, device=self.device)
        retD = torch.as_tensor(rD, dtype=torch.float32, device=self.device)
        retS = torch.as_tensor(rS, dtype=torch.float32, device=self.device)
        idxs = np.arange(len(rows))
        stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.cfg.minibatch_size):
                mb = idxs[start:start + self.cfg.minibatch_size]
                losses = []
                entropies = []
                new_logps = []
                v_losses = []
                for i in mb:
                    logits, vals = self.model(rows[i]["obs"])
                    logits = self._apply_cpeg_penalty(logits, rows[i]["obs"])
                    logits = self._apply_anti_idle_penalty(logits, rows[i]["obs"])
                    logits = self._apply_energy_guard_penalty(logits, rows[i]["obs"])
                    logits = self._apply_activation_guard_penalty(logits, rows[i]["obs"])
                    dist = Categorical(logits=logits)
                    a = actions[i]
                    new_logps.append(dist.log_prob(a))
                    entropies.append(dist.entropy())
                    if self.cfg.scalar_gae:
                        v_losses.append(F.mse_loss(vals["scalar"], retS[i]))
                    else:
                        v_losses.extend([
                            F.mse_loss(vals["energy"], retE[i]),
                            F.mse_loss(vals["power"], retP[i]),
                            F.mse_loss(vals["deadline"], retD[i]),
                        ])
                new_logps = torch.stack(new_logps)
                entropy = torch.stack(entropies).mean()
                ratio = torch.exp(new_logps - old_logp[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.stack(v_losses).mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                stats = {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()), "entropy": float(entropy.item())}
        return stats
