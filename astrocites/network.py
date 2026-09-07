import torch


class NetworkMonitor:
    def __init__(self, network, state_vars=('v', 's', 'w')):
        self.network = network
        self.state_vars = state_vars
        self.recording = {}
        self._targets = None
        self.reset_()

    def _build_targets(self):
        targets = []
        for name, layer in self.network.layers.items():
            for var in self.state_vars:
                if hasattr(layer, var):
                    targets.append((name, layer, var))
        for conn_key, conn in self.network.connections.items():
            for var in self.state_vars:
                if hasattr(conn, var):
                    targets.append((conn_key, conn, var))
        self._targets = targets

    def record(self):
        if self._targets is None:
            self._build_targets()
        recording = self.recording
        for name, obj, var in self._targets:
            data = getattr(obj, var)
            if var == 's':
                data = data.float()
            recording.setdefault(name, {}).setdefault(var, []).append(data.detach().clone())

    def get(self):
        result = {}
        for key, val_dict in self.recording.items():
            result[key] = {}
            for var, data_list in val_dict.items():
                if data_list:
                    result[key][var] = torch.stack(data_list)
        return result

    def reset_(self):
        self.recording = {}


class Network(torch.nn.Module):
    def __init__(self, dt=1.0, batch_size=1, learning=True):
        super().__init__()
        self.dt = dt
        self.batch_size = batch_size
        self.learning = learning
        self.layers = torch.nn.ModuleDict()
        self.connections = torch.nn.ModuleDict()
        self.monitors = {}

    def add_layer(self, layer, name):
        self.layers[name] = layer
        if hasattr(layer, 'compute_decays'):
            layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(self, connection, source, target):
        key = f"{source}_{target}"
        self.connections[key] = connection

    def add_monitor(self, monitor, name):
        self.monitors[name] = monitor

    def _get_inputs(self, layers=None):
        inpts = {}
        if layers is None:
            layers = self.layers.keys()
        for layer_name in layers:
            if layer_name not in inpts:
                layer = self.layers[layer_name]
                inpts[layer_name] = torch.zeros(self.batch_size, *layer.shape)
        for conn_key, connection in self.connections.items():
            parts = conn_key.split('_')
            if len(parts) >= 2:
                src = parts[0]
                tgt = parts[1]
                if tgt in layers:
                    source_output = connection.compute(self.layers[src].s)
                    if source_output.dim() == 1:
                        source_output = source_output.unsqueeze(0)
                    inpts[tgt] += source_output
        return inpts

    def run(self, inpts, time, injects_v=None, current_position=None, adjacent_positions=None, conn_XY=None, enable_stdp=True, **kwargs):
        timesteps = int(time / self.dt)
        injects_v = injects_v or {}
        layer_items = list(self.layers.items())
        # resolve connection routing once instead of re-parsing keys every timestep
        routes = []
        for conn_key, connection in self.connections.items():
            parts = conn_key.split('_')
            if len(parts) >= 2 and parts[1] in self.layers:
                routes.append((connection, self.layers[parts[0]], parts[1]))
        # input buffers are reused across timesteps; layers never mutate their input
        current_inpts = {name: torch.zeros(self.batch_size, *layer.shape) for name, layer in layer_items}
        for t_step in range(timesteps):
            for buf in current_inpts.values():
                buf.zero_()
            for connection, source_layer, tgt in routes:
                source_output = connection.compute(source_layer.s)
                if source_output.dim() == 1:
                    source_output = source_output.unsqueeze(0)
                current_inpts[tgt] += source_output
            for layer_name, input_data in inpts.items():
                if layer_name in current_inpts:
                    if len(input_data.shape) == 3:
                        current_inpts[layer_name] += input_data[t_step]
                    else:
                        current_inpts[layer_name] += input_data
            for name, layer in layer_items:
                if name in injects_v:
                    inject_voltage = injects_v[name]
                    if len(inject_voltage.shape) == 1:
                        layer.v += inject_voltage
                    else:
                        layer.v += inject_voltage[t_step]
                if name in ['Y', 'I']:
                    layer.forward(current_inpts[name], current_position=current_position, adjacent_positions=adjacent_positions)
                else:
                    layer.forward(current_inpts[name])
            for connection, source_layer, _tgt in routes:
                if getattr(connection.target, 'enable_astrocyte', False):
                    # clone: layer spike buffers are reused across timesteps, but the
                    # astrocyte must see the spikes as of when they were stored
                    connection.target.__dict__['prev_layer_s'] = source_layer.s.clone()
                if connection is conn_XY:
                    if enable_stdp:
                        connection.update(current_position=current_position,
                                          adjacent_positions=adjacent_positions,
                                          learning=self.learning, **kwargs)
                else:
                    connection.update(learning=self.learning, **kwargs)
            for monitor in self.monitors.values():
                monitor.record()

    def reset_(self):
        for layer in self.layers.values():
            layer.reset_()
        for connection in self.connections.values():
            connection.reset_()
        for monitor in self.monitors.values():
            monitor.reset_()