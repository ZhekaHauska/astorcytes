import torch
import os
import numpy as np


class LearningRule:
    def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0, **kwargs):
        self.connection = connection
        self.nu = nu if nu is not None else [0.0, 0.0]
        self.reduction = reduction if reduction is not None else torch.mean
        self.weight_decay = weight_decay

    def update(self, **kwargs):
        if self.weight_decay != 0:
            self.connection.w.data *= (1 - self.weight_decay)


class WeightDependentPostPre(LearningRule):
    def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0,
                 post_spike_weight_decay=0.0, tc_trace=20, tc_trace_neg=20, **kwargs):
        super().__init__(connection, nu, reduction, weight_decay, **kwargs)
        self.post_spike_weight_decay = post_spike_weight_decay
        self.tc_trace = tc_trace
        self.tc_trace_neg = tc_trace_neg
        self.interval = 100
        stdp_path = os.path.join(os.path.dirname(__file__), "..", "STDP.txt")
        try:
            with open(stdp_path, 'r') as fl:
                self.STDP_base = torch.zeros([101, 120])
                i = 0
                for line in fl:
                    k = 0
                    for sym in line.split():
                        self.STDP_base[i][k] = float(sym)
                        k += 1
                    i += 1
        except FileNotFoundError:
            self.STDP_base = torch.randn(101, 120) * 0.1
        self.wmin = getattr(connection, 'wmin', 0.001)
        self.wmax = getattr(connection, 'wmax', 1.0)

    def delta_w_custom_single(self, weight, delta_val):
        first_index = int(round(float(weight / self.nu[0] * 100)))
        if torch.isinf(delta_val) or torch.isnan(delta_val):
            delta_val = torch.tensor(0.0)
        second_index = int(float(delta_val) + 60)
        if second_index > 120 or second_index < 0:
            second_index = 0
        if first_index < 0:
            first_index = -first_index
        if first_index > 100:
            first_index = 100
        return self.STDP_base[first_index][second_index]

    def update(self, current_position=None, adjacent_positions=None, **kwargs):
        batch_size = self.connection.source.batch_size
        if current_position is not None and adjacent_positions is not None:
            update = torch.zeros_like(self.connection.w)
            source_s_current = self.connection.source.s[:, current_position].unsqueeze(1).unsqueeze(2).float()
            source_x_current = self.connection.source.x[:, current_position].unsqueeze(1).unsqueeze(2)
            for post_idx in adjacent_positions:
                target_s_adj = self.connection.target.s[:, post_idx].unsqueeze(1).unsqueeze(1).float()
                target_x_adj = self.connection.target.x_neg[:, post_idx].unsqueeze(1).unsqueeze(1)
                outer_product_pre = self.reduction(torch.bmm(source_s_current, target_x_adj), dim=0)
                outer_product_pre = torch.clamp(outer_product_pre, min=1e-10)
                delta_pre = self.tc_trace_neg * torch.log(outer_product_pre)
                update_pre_val = self.nu[0] * self.delta_w_custom_single(
                    self.connection.w[current_position, post_idx], delta_pre[0, 0])
                update[current_position, post_idx] += update_pre_val
                outer_product_post = self.reduction(torch.bmm(source_x_current, target_s_adj), dim=0)
                outer_product_post = torch.clamp(outer_product_post, min=1e-10)
                delta_post = -self.tc_trace * torch.log(outer_product_post)
                update_post_val = self.nu[1] * self.delta_w_custom_single(
                    self.connection.w[current_position, post_idx], delta_post[0, 0])
                update[current_position, post_idx] += update_post_val
                decay_factor = self.reduction(torch.bmm(torch.ones_like(source_x_current), target_s_adj), dim=0)
                update[current_position, post_idx] += (-self.post_spike_weight_decay) * self.connection.w[current_position, post_idx] * decay_factor[0, 0]
            self.connection.w += update
        else:
            source_s = self.connection.source.s.view(batch_size, -1).unsqueeze(2).float()
            source_x = self.connection.source.x.view(batch_size, -1).unsqueeze(2)
            target_s = self.connection.target.s.view(batch_size, -1).unsqueeze(1).float()
            target_x = self.connection.target.x_neg.view(batch_size, -1).unsqueeze(1)
            update = 0
            outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
            outer_product = torch.clamp(outer_product, min=1e-10)
            update += self.nu[0] * self.delta_w_custom(self.tc_trace_neg * torch.log(outer_product))
            outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
            outer_product = torch.clamp(outer_product, min=1e-10)
            update += self.nu[1] * self.delta_w_custom(-self.tc_trace * torch.log(outer_product))
            update += (-self.post_spike_weight_decay) * self.connection.w * self.reduction(
                torch.bmm(torch.ones(source_x.shape), target_s), dim=0)
            self.connection.w += update
        super().update()

    def delta_w_custom(self, delta):
        raise NotImplementedError("delta_w_custom not implemented for batch mode")


class NoOp(LearningRule):
    def update(self, **kwargs):
        super().update()