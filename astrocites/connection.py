import torch
import numpy as np
from astrocites.learning import NoOp, WeightDependentPostPre


class Connection(torch.nn.Module):
    def __init__(self, source, target, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                 impulse_length=40, impulse_shape_factor=0.9, invert=False,
                 update_rule=NoOp, w=None, nu=None, wmin=0, wmax=1,
                 weight_decay=0, post_spike_weight_decay=0,
                 baseline_decay=0.01, policy_mix_beta=0.5, **kwargs):
        super().__init__()
        self.source = source
        self.target = target
        self.wmin = wmin
        self.wmax = wmax
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)
        self.w = torch.nn.Parameter(w, False)
        self.update_rule = update_rule(self, nu=nu, weight_decay=weight_decay,
                                        post_spike_weight_decay=post_spike_weight_decay)
        self.impulse_amplitude = impulse_amplitude
        self.impulse_amplitude_2 = impulse_amplitude_2
        self.impulse_length = impulse_length
        self.impulse_shape_factor = impulse_shape_factor
        self.invert = invert
        self.register_buffer("a_pre", torch.zeros(source.n))
        self.register_buffer("impulse_state", torch.zeros(source.n))
        self.baseline_decay = baseline_decay
        self.policy_mix_beta = policy_mix_beta
        self.running_baseline = 0.0
        self.step_buffer = []

    def impulse_curve(self):
        k = self.impulse_shape_factor
        if self.invert:
            impulse_value_2 = self.impulse_amplitude / (self.impulse_length * k - 1)
            impulse_value_1 = self.impulse_amplitude / (self.impulse_length * (1 - k))
            impulse_bias = 2 * self.impulse_amplitude * (self.impulse_state > (self.impulse_length * (1 - k) + 0.5)).float() * (self.impulse_state <= (self.impulse_length * (1 - k) + 1.5)).float()
            impulse = (-impulse_value_1) * (self.impulse_state > 0).float() * (self.impulse_state <= (self.impulse_length * (1 - k) + 0.5)).float() + (-impulse_value_2) * (self.impulse_state > (self.impulse_length * (1 - k) + 1.5)).float() + impulse_bias
            return impulse
        else:
            impulse_value_2 = self.impulse_amplitude / (self.impulse_length * k - 1)
            impulse_value_1 = self.impulse_amplitude_2 / (self.impulse_length * (1 - k))
            impulse_bias = (self.impulse_amplitude + self.impulse_amplitude_2) * (self.impulse_state >= (self.impulse_length * k)).float() * (self.impulse_state < (self.impulse_length * k + 1)).float()
            impulse = (impulse_value_1) * (self.impulse_state > (self.impulse_length * k)).float() + (impulse_value_2) * (self.impulse_state > 0).float() * (self.impulse_state < (self.impulse_length * k)).float() - impulse_bias
            return impulse

    def update_impulse_state(self, s):
        self.impulse_state += (self.impulse_state > 0).float()
        s_modified = s.clone()
        if len(s_modified.shape) == 1:
            s_modified = s_modified.unsqueeze(0)
        s_modified[:, self.impulse_state > 0] = 0
        self.impulse_state += (self.impulse_state == 0).float() * s_modified.float().view(-1)
        impulse = self.impulse_curve()
        self.impulse_state *= (self.impulse_state < self.impulse_length).float()
        return impulse

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        impulse = self.update_impulse_state(s)
        self.a_pre += impulse
        self.a_pre *= (self.impulse_state > 0).float()
        a_post = self.a_pre @ self.w
        return a_post.view(1, *self.target.shape)

    def update(self, **kwargs):
        self.update_rule.update(**kwargs)

    def store_step(self, s_idx, adj_positions, probs, chosen_idx, step_reward):
        self.step_buffer.append({
            's_idx': s_idx,
            'adj_positions': list(adj_positions),
            'probs': probs.detach().clone(),
            'chosen_idx': chosen_idx,
            'step_reward': step_reward,
        })

    def compute_and_apply_reinforce_update(self, lr, gamma, temperature, mask=None):
        T = len(self.step_buffer)
        if T == 0:
            return

        rewards = [step['step_reward'] for step in self.step_buffer]
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        self.running_baseline = (1 - self.baseline_decay) * self.running_baseline + self.baseline_decay * returns[0]
        baseline = self.running_baseline

        delta_w = torch.zeros_like(self.w)
        beta = self.policy_mix_beta

        for t, step in enumerate(self.step_buffer):
            s_idx = step['s_idx']
            adj_positions = step['adj_positions']
            probs = step['probs']
            chosen_idx = step['chosen_idx']
            advantage = returns[t] - baseline

            for i, a in enumerate(adj_positions):
                indicator = 1.0 if i == chosen_idx else 0.0
                score = beta / temperature * (indicator - probs[i].item())
                delta_w[s_idx, a] += advantage * score

        self.w.data += lr * delta_w
        self.w.data.clamp_(self.wmin, self.wmax)
        if mask is not None:
            self.w.data *= mask

        self.step_buffer = []

    def reset_episode(self):
        self.step_buffer = []

    def reset_(self):
        self.a_pre.zero_()
        self.impulse_state.zero_()