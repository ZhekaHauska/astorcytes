import torch


class Nodes(torch.nn.Module):
    def __init__(self, n=None, shape=None, traces=False, traces_additive=False,
                 tc_trace=20.0, tc_trace_neg=20.0, trace_scale=1.0, sum_input=False):
        super().__init__()
        self.n = n
        self.shape = (n,) if shape is None else shape
        self.batch_size = 1
        self.traces = traces
        self.traces_additive = traces_additive
        self.register_buffer("tc_trace", torch.tensor(tc_trace, dtype=torch.float))
        self.register_buffer("tc_trace_neg", torch.tensor(tc_trace_neg, dtype=torch.float))
        self.trace_scale = trace_scale
        self.sum_input = sum_input
        if traces:
            self.register_buffer("x", torch.zeros(1, *self.shape))
            self.register_buffer("x_neg", torch.zeros(1, *self.shape))
            self.register_buffer("trace_decay_pre", torch.exp(-1.0 / self.tc_trace))
            self.register_buffer("trace_decay_neg", torch.exp(-1.0 / self.tc_trace_neg))
        else:
            self.x = None
            self.x_neg = None
        self.register_buffer("s", torch.zeros(1, *self.shape, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> None:
        if self.traces:
            if self.trace_scale == 1.0:
                s_f = self.s.float()
            else:
                s_f = self.s.float() * self.trace_scale
            self.x.mul_(self.trace_decay_pre).add_(s_f)
            self.x_neg.mul_(self.trace_decay_neg).add_(s_f)

    def reset_(self) -> None:
        self.s.zero_()
        if self.traces:
            self.x.zero_()
            self.x_neg.zero_()

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def compute_decays(self, dt: float) -> None:
        pass


class Input(Nodes):
    def __init__(self, n=None, shape=None, traces=False, traces_additive=False,
                 tc_trace=20.0, tc_trace_neg=20.0, trace_scale=1.0, sum_input=False,
                 thresh=-52.0, rest=-65.0, reset=-65.0, refrac=60, **kwargs):
        super().__init__(n, shape, traces, traces_additive, tc_trace, tc_trace_neg, trace_scale, sum_input)
        self.register_buffer("rest", torch.tensor(rest, dtype=torch.float))
        self.register_buffer("reset", torch.tensor(reset, dtype=torch.float))
        self.register_buffer("thresh", torch.tensor(thresh, dtype=torch.float))
        self.register_buffer("refrac", torch.tensor(refrac))
        self.register_buffer("refrac_count", torch.zeros(1, n))

    def forward(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        max_val, neuron_idx = torch.max(x[0], 0)
        if self.s.shape != x.shape:
            self.s = torch.zeros_like(x, dtype=torch.bool)
        else:
            self.s.zero_()
        refrac_active = self.refrac_count > 0
        self.refrac_count.sub_(1.0)
        self.refrac_count.mul_(refrac_active.float())
        if max_val > 0:
            batch_idx = 0
            if self.refrac_count[batch_idx, neuron_idx] == 0:
                self.s[batch_idx, neuron_idx] = True
                self.refrac_count[batch_idx, neuron_idx] = self.refrac
        super().forward(x)

    def reset_(self) -> None:
        super().reset_()
        self.refrac_count.zero_()

    def set_batch_size(self, batch_size: int) -> None:
        super().set_batch_size(batch_size)
        self.refrac_count = torch.zeros((batch_size, self.n), device=self.refrac_count.device)


class LIFNodes(Nodes):
    def __init__(self, n=None, shape=None, traces=False, traces_additive=False,
                 tc_trace=20.0, tc_trace_neg=20.0, trace_scale=1.0, sum_input=False,
                 thresh=-52.0, rest=-65.0, reset=-65.0, refrac=5, tc_decay=150.0,
                 dt=1.0, lbound=None, enable_astrocyte=False, alpha=None, k=None, **kwargs):
        super().__init__(n, shape, traces, traces_additive, tc_trace, tc_trace_neg, trace_scale, sum_input)
        self.dt = dt
        self.enable_astrocyte = enable_astrocyte
        self.register_buffer("rest", torch.tensor(rest, dtype=torch.float))
        self.register_buffer("reset", torch.tensor(reset, dtype=torch.float))
        self.register_buffer("thresh", torch.tensor(thresh, dtype=torch.float))
        self.register_buffer("refrac", torch.tensor(refrac))
        self.register_buffer("tc_decay", torch.tensor(tc_decay, dtype=torch.float))
        self.register_buffer("decay", torch.zeros(*self.shape))
        self.register_buffer("v", torch.zeros(1, *self.shape))
        self.register_buffer("refrac_count", torch.zeros(1, *self.shape))
        self.lbound = lbound
        self.register_buffer("G", torch.zeros(n))
        self.register_buffer("Ca", torch.zeros(n))
        self.Ca[:] = 0
        self.k = k
        self.alpha = alpha
        self.initial_thresh = thresh
        self.G_thr = 0.2
        self.Ca_duration = 1000 * 100
        self.prev_layer_s = None
        # (position, adjacency) -> active neuron indices; rebuilt only when the
        # navigation step changes, reused across the timestep loop
        self._idx_cache = (None, None)

    def astrocyte_input(self):
        if self.prev_layer_s is not None:
            if self.prev_layer_s.dim() > 1:
                prev_s_flat = self.prev_layer_s.view(-1, self.n).float().mean(dim=0)
            else:
                prev_s_flat = self.prev_layer_s.float()
            self.G.sub_(self.dt * (self.alpha * self.G - self.k * prev_s_flat))
        self.Ca[self.G > self.G_thr] = self.Ca_duration
        self.Ca.sub_(1.0)
        self.Ca.clamp_(min=0)
        return 1.0 / (1.0 + 5.0 * (self.Ca > 0).float())

    def forward(self, x: torch.Tensor, current_position: int = None, adjacent_positions: list = None) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.enable_astrocyte:
            self.thresh.copy_(self.initial_thresh * self.astrocyte_input())
        if current_position is None and adjacent_positions is None:
            idx_tensor = None
        else:
            key = (current_position,
                   tuple(adjacent_positions) if adjacent_positions is not None else None)
            if self._idx_cache[0] == key:
                idx_tensor = self._idx_cache[1]
            else:
                idx = [] if current_position is None else [current_position]
                if adjacent_positions is not None:
                    idx.extend(adjacent_positions)
                idx = list(dict.fromkeys(idx))
                idx_tensor = (torch.as_tensor(idx, dtype=torch.long, device=self.v.device)
                              if idx else None)
                self._idx_cache = (key, idx_tensor)

        if self.s.shape != x.shape:
            self.s = torch.zeros_like(x, dtype=torch.bool)
        else:
            self.s.zero_()

        if idx_tensor is None:
            v = self.v
            rc = self.refrac_count
            x_act = x
            thresh = self.thresh
        else:
            v = self.v[:, idx_tensor]
            rc = self.refrac_count[:, idx_tensor]
            x_act = x[:, idx_tensor]
            thresh = self.thresh[idx_tensor] if self.thresh.dim() > 0 else self.thresh

        v = self.decay * (v - self.rest) + self.rest
        rc = (rc > 0).float() * (rc - self.dt)
        rc_free = rc == 0
        v = v + rc_free.float() * x_act
        spike = v >= thresh
        rc = torch.where(spike, self.refrac, rc)
        v = torch.where(spike, self.reset, v)
        if self.lbound is not None:
            v = v.clamp_min(self.lbound)

        if idx_tensor is None:
            self.v.copy_(v)
            self.refrac_count.copy_(rc)
            self.s.copy_(spike)
        else:
            self.v[:, idx_tensor] = v
            self.refrac_count[:, idx_tensor] = rc
            self.s[:, idx_tensor] = spike
        super().forward(x)

    def reset_(self) -> None:
        super().reset_()
        self.v.fill_(self.rest)
        self.refrac_count.zero_()
        self.G.zero_()

    def compute_decays(self, dt) -> None:
        self.decay = torch.exp(-dt / self.tc_decay)

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros(batch_size, *self.shape, device=self.refrac_count.device)
        if self.traces:
            self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
            self.x_neg = torch.zeros(batch_size, *self.shape, device=self.x_neg.device)
        self.s = torch.zeros(batch_size, *self.shape, dtype=torch.bool, device=self.s.device)