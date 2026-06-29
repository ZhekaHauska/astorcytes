# Architecture and Weight Update Formulas

## 1. Network Architecture

The network is a three-layer spiking neural network for grid navigation. Each of the `NA = N²` neurons corresponds to a grid cell. The agent's current position drives the input layer; the output layer's spikes determine the next action.

### Layers

| Layer | Type | Role |
|-------|------|------|
| **X** (input) | `Input` | Winner-take-all encoder. Receives Bernoulli spike trains at the current position. |
| **Y** (output) | `LIFNodes` | Action selection via spike counts over a `time_steps` window. |
| **I** (inhibitor) | `LIFNodes` | Lateral inhibition via astrocyte-modulated thresholds; receives constant voltage injection. |

### Connections

| Connection | Rule | Role |
|------------|------|------|
| **X → Y** (`conn_XY`) | `WeightDependentPostPre` (STDP) + REINFORCE | Learned excitatory weights; the only trainable connection. |
| **X → I** (`conn_XI`) | `NoOp` (fixed) | Drives the inhibitor layer; weights initialized to identity. |
| **I → Y** (`conn_IY`) | `NoOp` (fixed, sign-flipped) | Inhibitory feedback: weights are the negation of `conn_XI`. |

### Forward Pass (per timestep)

```
Network.run(time=time_steps):
    for t in range(time_steps):
        1. inpts = _get_inputs()           # compute(s) for each connection
        2. inpts['X'] += bernoulli_input[t] # add external drive
        3. layer_I.v += 0.02                # constant voltage injection to inhibitor
        4. for each layer: layer.forward(inpts[layer])
        5. for each connection: connection.update()   # STDP or NoOp
        6. monitors.record()
```


## 2. Neuron Dynamics

### 2.1 Input Layer (X) — Winner-Take-All

The input layer does **not** perform LIF integration. It spikes the single neuron with the highest input, subject to a refractory period (`refrac = 60` timesteps):

```
neuron_idx = argmax(x)
if x[neuron_idx] > 0 and refrac_count[neuron_idx] == 0:
    s[neuron_idx] = 1
    refrac_count[neuron_idx] = refrac
refrac_count = max(refrac_count - 1, 0)
```

The external input is a Bernoulli spike train with per-timestep firing probability `intensity / time_steps` at the current position, producing approximately `intensity` spikes over the window.

### 2.2 LIF Neurons (Y, I)

**Parameters** (default values from config):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Threshold | $\theta$ | 7 | Spike threshold voltage |
| Rest / Reset | $V_{\text{rest}} = V_{\text{reset}}$ | 0 | Resting and post-spike reset voltage |
| Refractory | $t_{\text{ref}}$ | 40 | Refractory period (timesteps) |
| Time constant | $\tau$ | 150 | Membrane decay time constant |
| Decay factor | $d$ | $e^{-dt/\tau} \approx 0.9934$ | Per-timestep voltage decay |
| Timestep | $dt$ | 1 | Simulation timestep |

**Voltage update** (only for neurons in `{current_position} ∪ adjacent_positions`):

$$
v_i[t] = d \cdot (v_i[t-1] - V_{\text{rest}}) + V_{\text{rest}} + \mathbb{1}[\text{not refractory}] \cdot I_i[t]
$$

With $V_{\text{rest}} = 0$ this simplifies to:

$$
v_i[t] = d \cdot v_i[t-1] + \mathbb{1}[\text{refrac\_count}_i = 0] \cdot I_i[t]
$$

**Spike condition** (hard Heaviside — non-differentiable):

$$
s_i[t] = \mathbb{1}[v_i[t] \geq \theta_i] \qquad \text{(boolean, detached from autograd)}
$$

**Reset and refractory on spike:**

$$
v_i[t] \leftarrow V_{\text{reset}}, \qquad \text{refrac\_count}_i \leftarrow t_{\text{ref}}
$$

### 2.3 Eligibility Traces

Each LIF neuron maintains positive and negative traces used by STDP:

$$
x_i[t] = x_i[t-1] \cdot e^{-1/\tau_{\text{trace}}} + \text{trace\_scale} \cdot s_i[t]
$$

$$
x_i^{\text{neg}}[t] = x_i^{\text{neg}}[t-1] \cdot e^{-1/\tau_{\text{trace,neg}}} + \text{trace\_scale} \cdot s_i[t]
$$

with $\tau_{\text{trace}} = \tau_{\text{trace,neg}} = 20$.


## 3. Synaptic Impulse Waveform

Each pre-synaptic spike triggers a **biphasic post-synaptic current waveform** (Dale-type, `invert=True`) lasting `impulse_length - 1` timesteps. The waveform is parameterized by:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Amplitude $A$ | 0.5 | Peak amplitude |
| Length $L$ | 40 | Waveform duration (timesteps) |
| Shape factor $k$ | 0.9 | Fraction of the waveform in the second phase |

The waveform consists of three phases applied to the pre-synaptic activation buffer `a_pre`:

1. **Inhibitory phase** ($L(1-k) = 4$ timesteps): impulse $= -A / (L(1-k))$ each
2. **Bias pulse** (1 timestep): impulse $= +2A$
3. **Decaying phase** ($Lk - 1 = 35$ timesteps): impulse $= -A/(Lk - 1)$ each

The post-synaptic current into neuron $j$ at each timestep is:

$$
I_j[t] = \mathbf{a}_{\text{pre}}[t] \cdot \mathbf{w}[:, j]
$$

where $\mathbf{a}_{\text{pre}}$ accumulates the impulse waveform and $\mathbf{w}$ is the weight matrix.

### Total Charge per Spike (Impulse Integral)

The cumulative `a_pre` summed over the full waveform window has a closed form:

$$
\mathcal{A} = \sum_{t=1}^{L-1} a_{\text{pre}}[t] = \frac{A \cdot (L(2k - 1) - 1)}{2}
$$

For the default parameters: $\mathcal{A} = 0.5 \cdot (40 \cdot 0.8 - 1) / 2 = 7.75$.


## 4. Astrocyte Modulation (Inhibitor Layer I)

When `enable_astrocyte = True`, the inhibitor layer's spike threshold is dynamically modulated by gliotransmitter dynamics driven by pre-synaptic (layer X) activity.

### Gliotransmitter Dynamics

$$
G_i[t] = G_i[t-1] - dt \cdot (\alpha \cdot G_i[t-1] - k \cdot \bar{s}_{\text{pre},i}[t])
$$

where $\bar{s}_{\text{pre},i}$ is the mean pre-synaptic spike rate for neuron $i$, $\alpha = 0.001$, $k = 0.2$.

### Calcium Events

When $G_i > G_{\text{thr}}$ ($= 0.2$), a calcium event is triggered lasting $Ca_{\text{duration}}$ timesteps:

$$
\text{Ca}_i \leftarrow Ca_{\text{duration}} \quad \text{if } G_i > G_{\text{thr}}
$$

$$
\text{Ca}_i[t] = \max(\text{Ca}_i[t-1] - 1, \, 0)
$$

### Threshold Modulation

During a calcium event, the effective threshold is **lowered** (increased excitability):

$$
\theta_i^{\text{eff}} = \theta_i^{\text{initial}} \cdot \frac{1}{1 + 5 \cdot \mathbb{1}[\text{Ca}_i > 0]}
$$

This divides the threshold by 6 during calcium events, making those neurons more excitable.


## 5. Learning: STDP (Weight-Dependent Post-Pre)

The `conn_XY` connection uses a weight-dependent STDP rule with a precomputed lookup table (`STDP.txt`, shape $101 \times 120$). The update is computed per `(current_position → adjacent_position)` pair.

### Timing Differences

For a pre-synaptic neuron $i$ (in layer X) and post-synaptic neuron $j$ (in layer Y):

**Depression (pre before post):**

$$
\Delta_{\text{pre}} = \tau_{\text{trace,neg}} \cdot \ln\bigl(\text{clamp}(s_i \cdot x_j^{\text{neg}}, \, \epsilon)\bigr)
$$

**Potentiation (post before pre):**

$$
\Delta_{\text{post}} = -\tau_{\text{trace}} \cdot \ln\bigl(\text{clamp}(x_i \cdot s_j, \, \epsilon)\bigr)
$$

### Lookup Table Indexing

Each timing difference is mapped to a weight change via the lookup table, indexed by the current weight value and the timing:

$$
\text{idx}_{\text{weight}} = \text{round}\!\left(\frac{w_{ij}}{\nu_0} \cdot 100\right), \qquad \text{idx}_{\text{time}} = \lfloor \Delta + 60 \rfloor
$$

$$
\Delta w_{ij} = \nu_0 \cdot \text{STDP\_base}[\text{idx}_{\text{weight}}][\text{idx}_{\text{time,pre}}] + \nu_1 \cdot \text{STDP\_base}[\text{idx}_{\text{weight}}][\text{idx}_{\text{time,post}}]
$$

with learning rates $\nu_0 = \nu_1 = 10$.

### Post-Spike Weight Decay

An additional decay is applied when the post-synaptic neuron spikes:

$$
\Delta w_{ij} \mathrel{-}= \lambda_{\text{post}} \cdot w_{ij} \cdot s_j
$$

with $\lambda_{\text{post}} = 0.005$.

### Weight Clamping and Masking

After each navigation step, weights are masked to the adjacency matrix (zeroing non-adjacent connections). After clamping, $w \in [w_{\min}, w_{\max}] = [0.001, 1]$.


## 6. Learning: REINFORCE with Surrogate Gradients

### 6.1 Problem

Action selection is based on **actual SNN spike counts** (a spike-based policy). However, spikes are produced by a hard threshold function and are `torch.bool` — there is no differentiable path from weights to spike counts. Without intervention, the REINFORCE eligibility gradient $\nabla_w \log \pi(a)$ cannot flow through the spike signal.

### 6.2 Solution: Decoupled Surrogate Rate

A **differentiable surrogate** of the LIF firing rate $r(w)$ is used solely for gradient computation. The SNN forward pass (hard spikes) is untouched and still drives action selection.

**Action selection** uses actual recorded spike counts:

$$
\text{logits}_a = \beta \cdot w_{s,a} + (1 - \beta) \cdot n_a^{\text{spike}}
$$

$$
\pi_{\text{action}}(a) = \text{softmax}\!\left(\frac{\text{logits}_a}{T}\right)
$$

where $\beta$ is `policy_mix_beta` (default 0.0 = pure spike-based), $T$ is temperature, $s$ is the current position, and $n_a^{\text{spike}}$ is the recorded spike count for output neuron $a$ over the `time_steps` window.

**Gradient computation** uses the surrogate rate $\hat{r}(a)$, a smooth differentiable function of the weight $w_{s,a}$:

$$
\pi_{\text{grad}}(a) = \text{softmax}\!\left(\frac{\hat{r}(I_{\text{eff}}(a))}{T}\right)
$$

### 6.3 Effective Input Current

The per-timestep input current into output neuron $a$ is linear in the weight (only the current position spikes in layer X):

$$
I_{\text{eff}}(a) = c \cdot w_{s,a}
$$

where the scale constant $c$ is derived analytically from the impulse waveform:

$$
c = \frac{\mathcal{A} \cdot \text{intensity}}{\text{time\_steps}}
$$

For default parameters: $c = 7.75 \times 15 / 1000 = 0.116$.

### 6.4 Surrogate Rate Functions

Two surrogates are available (config-selectable via `surrogate.type`):

#### LIF Analytical Rate (`"lif"`)

Derived from the steady-state solution of the discrete LIF equation under constant current $I$:

$$
I_\theta = \theta \cdot (1 - d) \qquad \text{(threshold current)}
$$

$$
\hat{r}(I) = \frac{1}{t_{\text{ref}} + \tau \cdot \ln\!\left(\dfrac{I}{I - I_\theta}\right)} \quad \text{for } I > I_\theta, \quad \text{else } 0
$$

Its derivative:

$$
\hat{r}'(I) = \frac{\tau \cdot I_\theta}{I \cdot (I - I_\theta) \cdot \bigl(t_{\text{ref}} + \tau \cdot \ln\!\left(\frac{I}{I - I_\theta}\right)\bigr)^2}
$$

The function is continuous ($\hat{r} \to 0$ as $I \to I_\theta^+$) but the derivative diverges near threshold.

#### Softplus Proxy (`"softplus"`)

A simpler smooth surrogate with no singularity:

$$
\hat{r}(I) = \text{softplus}\bigl(\text{scale} \cdot (I - I_\theta)\bigr) = \ln\!\bigl(1 + e^{\text{scale} \cdot (I - I_\theta)}\bigr)
$$

$$
\hat{r}'(I) = \text{scale} \cdot \sigma\bigl(\text{scale} \cdot (I - I_\theta)\bigr)
$$

### 6.5 Eligibility Gradient

The REINFORCE score function through the surrogate rate, per candidate action $a$:

$$
\frac{\partial \log \pi(a^*)}{\partial w_{s,a}} = \frac{1}{T} \cdot \bigl(\mathbb{1}[a = a^*] - \pi_{\text{grad}}(a)\bigr) \cdot \hat{r}'(I_{\text{eff}}(a)) \cdot c
$$

This is accumulated into the eligibility trace $\mathbf{E}$ each step with exponential decay:

$$
\mathbf{E}[t] = \lambda_{\text{trace}} \cdot \mathbf{E}[t-1] + \nabla_w \log \pi(a^*[t])
$$

where $\lambda_{\text{trace}} = 0.95$ is `trace_decay`.

### 6.6 Returns and Baseline

The discounted return is accumulated over the episode:

$$
G = \sum_{t=0}^{T-1} \gamma^t \cdot r[t]
$$

with discount $\gamma = 0.99$. Rewards: $r = +10$ for reaching the goal, $r = -0.1$ per step.

A running baseline (exponential moving average) reduces variance:

$$
\bar{b} \leftarrow (1 - \beta_b) \cdot \bar{b} + \beta_b \cdot G
$$

with $\beta_b = 0.01$ (`baseline_decay`). The advantage is $A = G - \bar{b}$.

### 6.7 REINFORCE Weight Update

Applied **once per episode**:

$$
\mathbf{w} \leftarrow \text{clip}\!\left(\mathbf{w} + \eta_{\text{RL}} \cdot A \cdot \mathbf{E}, \; w_{\min}, \; w_{\max}\right)
$$

where $\eta_{\text{RL}} = 0.01$ is the REINFORCE learning rate. The weight matrix is then re-masked to the adjacency structure.


## 7. Combined Weight Update Summary

At each training cycle, `conn_XY.w` is modified by two mechanisms:

### Per-Timestep (during simulation)

$$
w_{ij} \mathrel{+}= \underbrace{\nu \cdot \text{STDP}(\Delta_{\text{pre}}, \Delta_{\text{post}}, w_{ij})}_{\text{weight-dependent STDP}} - \underbrace{\lambda_{\text{post}} \cdot w_{ij} \cdot s_j}_{\text{post-spike decay}}
$$

(STDP is applied each timestep when `enable_stdp_during_training = True`.)

### Per-Episode (after navigation completes)

$$
w_{ij} \mathrel{+}= \eta_{\text{RL}} \cdot (G - \bar{b}) \cdot E_{ij}
$$

where $E_{ij} = \sum_t \lambda_{\text{trace}}^{T-t} \cdot \nabla_{w_{ij}} \log \pi(a^*[t])$ is the eligibility trace accumulated via the surrogate gradient.

### After Each Step

$$
w_{ij} \leftarrow w_{ij} \cdot M_{ij} \qquad \text{(adjacency mask)}
$$


## 8. Parameters Reference

### Neuron Parameters

| Config Key | Symbol | Default | Description |
|------------|--------|---------|-------------|
| `neuron.thresh` | $\theta$ | 7 | Spike threshold |
| `neuron.reset` | $V_{\text{reset}}$ | 0 | Post-spike reset voltage |
| `neuron.refrac` | $t_{\text{ref}}$ | 40 | Refractory period (Y, I); 60 for X |
| `neuron.wmin` | $w_{\min}$ | 0.001 | Minimum weight |
| `neuron.wmax` | $w_{\max}$ | 1 | Maximum weight |
| `neuron.weight_decay` | — | 0 | Global weight decay (unused if 0) |
| `neuron.post_spike_weight_decay` | $\lambda_{\text{post}}$ | 0.005 | Post-spike weight decay |

### Astrocyte Parameters

| Config Key | Symbol | Default | Description |
|------------|--------|---------|-------------|
| `astrocyte.enable` | — | true | Enable astrocyte modulation |
| `astrocyte.alpha` | $\alpha$ | 0.001 | Gliotransmitter decay rate |
| `astrocyte.k` | $k$ | 0.2 | Gliotransmitter coupling gain |

### Simulation Parameters

| Config Key | Symbol | Default | Description |
|------------|--------|---------|-------------|
| `simulation.intensity` | — | 15.0 | Expected input spikes per window |
| `simulation.time_steps` | — | 1000 | Timesteps per navigation step |
| `simulation.dt` | $dt$ | 1 | Simulation timestep |

### REINFORCE Parameters

| Config Key | Symbol | Default | Description |
|------------|--------|---------|-------------|
| `reinforce.learning_rate` | $\eta_{\text{RL}}$ | 0.01 | REINFORCE learning rate |
| `reinforce.temperature` | $T$ | 1.0 | Softmax temperature |
| `reinforce.reward_goal` | — | 10.0 | Reward for reaching goal |
| `reinforce.step_penalty` | — | −0.1 | Per-step penalty |
| `reinforce.gamma` | $\gamma$ | 0.99 | Discount factor |
| `reinforce.baseline_decay` | $\beta_b$ | 0.01 | Baseline EMA decay |
| `reinforce.policy_mix_beta` | $\beta$ | 0.0 | Action-selection mix (0 = pure spikes) |
| `reinforce.trace_decay` | $\lambda_{\text{trace}}$ | 0.95 | Eligibility trace decay |
| `reinforce.enable_stdp_during_training` | — | true | Apply STDP during REINFORCE training |

### Surrogate Parameters

| Config Key | Default | Description |
|------------|---------|-------------|
| `reinforce.surrogate.type` | `"lif"` | Surrogate kind: `"lif"` or `"softplus"` |
| `reinforce.surrogate.tc_decay` | 150.0 | LIF membrane time constant (LIF only) |
| `reinforce.surrogate.scale` | 1.0 | Softplus steepness (softplus only) |
| `reinforce.surrogate.I_theta` | derived | Softplus knee current (softplus only) |


## 9. Key File Locations

| File | Contents |
|------|----------|
| `astrocites/nodes.py` | `Input` (WTA), `LIFNodes` (voltage dynamics, astrocyte modulation, spike generation at line 145) |
| `astrocites/connection.py` | `Connection` (impulse waveform, `compute()`, `accumulate_trace()`, `compute_and_apply_reinforce_update()`) |
| `astrocites/network.py` | `Network` (forward pass orchestration), `NetworkMonitor` |
| `astrocites/learning.py` | `WeightDependentPostPre` (STDP), `NoOp` |
| `astrocites/surrogate.py` | Surrogate rate functions, eligibility gradient, impulse integral derivation |
| `astrocites/experiment.py` | `setup_and_run_simulation()`, `setup_and_run_simulation_reinforce()`, `run_experiment()` |
| `configs/default.yaml` | Default parameter configuration |
| `tests/test_surrogate.py` | Autograd verification of surrogate derivatives and eligibility gradient |
