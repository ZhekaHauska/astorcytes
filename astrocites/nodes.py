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
        else:
            self.x = None
            self.x_neg = None
        self.register_buffer("s", torch.zeros(1, *self.shape, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> None:
        if self.traces:
            decay_factor_pre = torch.exp(-1.0 / self.tc_trace)
            decay_factor_post = torch.exp(-1.0 / self.tc_trace_neg)
            self.x = self.x * decay_factor_pre + self.trace_scale * self.s.float()
            self.x_neg = self.x_neg * decay_factor_post + self.trace_scale * self.s.float()

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
        batch_size, n_neurons = x.shape
        max_val, neuron_idx = torch.max(x[0], 0)
        self.s = torch.zeros_like(x, dtype=torch.bool)
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - 1.0)
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

    def astrocyte_input(self):
        response = torch.ones_like(self.thresh)
        if self.prev_layer_s is not None:
            if self.prev_layer_s.dim() > 1:
                prev_s_flat = self.prev_layer_s.view(-1, self.n).float().mean(dim=0)
            else:
                prev_s_flat = self.prev_layer_s.float()
            self.G = self.G - self.dt * (self.alpha * self.G - self.k * prev_s_flat)
        activated_neurons = self.G > self.G_thr
        self.Ca[activated_neurons] = self.Ca_duration
        self.Ca = torch.clamp(self.Ca - 1, min=0)
        imp_astro = torch.zeros_like(self.thresh)
        active_effect = self.Ca > 0
        imp_astro[active_effect] = 5
        response = response / (1 + imp_astro)
        return response

    def forward(self, x: torch.Tensor, current_position: int = None, adjacent_positions: list = None) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, n_neurons = x.shape
        if self.enable_astrocyte:
            self.thresh = self.initial_thresh * self.astrocyte_input()
        active_neurons = set()
        if current_position is not None:
            active_neurons.add(current_position)
        if adjacent_positions is not None:
            active_neurons.update(adjacent_positions)
        if not active_neurons:
            active_neurons = set(range(n_neurons))
        active_mask = torch.zeros(n_neurons, dtype=torch.bool, device=self.v.device)
        for idx in active_neurons:
            active_mask[idx] = True
        self.s = torch.zeros_like(x, dtype=torch.bool)
        self.v[:, active_mask] = self.decay * (self.v[:, active_mask] - self.rest) + self.rest
        self.refrac_count[:, active_mask] = (self.refrac_count[:, active_mask] > 0).float() * (self.refrac_count[:, active_mask] - self.dt)
        for neuron_idx in active_neurons:
            refrac_mask = self.refrac_count[:, neuron_idx] == 0
            self.v[:, neuron_idx] += refrac_mask.float() * x[:, neuron_idx]
            spike_mask = self.v[:, neuron_idx] >= self.thresh[neuron_idx]
            self.s[:, neuron_idx] = spike_mask
            if spike_mask.any():
                self.refrac_count[:, neuron_idx][spike_mask] = self.refrac
                self.v[:, neuron_idx][spike_mask] = self.reset
        if self.lbound is not None:
            self.v[:, active_mask] = torch.where(self.v[:, active_mask] < self.lbound, self.lbound, self.v[:, active_mask])
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