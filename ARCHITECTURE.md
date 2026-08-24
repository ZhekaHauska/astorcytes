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

Three surrogates are available (config-selectable via `surrogate.type`):

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

#### NMDA Baseline-Calcium (`"nmda"`)

A bio-plausible surrogate where the rate derivative is modeled as **calcium level above baseline**, motivated by NMDA receptor Mg²⁺ unblock dynamics (Jahr & Stevens, 1990) and the Ca²⁺/calmodulin cooperativity cascade (Chin & Means, 2000). The calcium level is a generalized sigmoid in input current:

$$
\text{Ca}(I) = \sigma\!\left(\frac{n_H \cdot (I - I_{1/2})}{k}\right)
$$

with $I_{1/2} = I_\theta$ (half-activation at the LIF threshold current), $k = 0.5 \cdot I_\theta$ (slope factor), and $n_H$ the **Hill cooperativity coefficient**. $n_H = 1$ gives the simple NMDA Mg²⁺ unblock sigmoid; $n_H \approx 3$–4 approximates the cooperativity of Ca²⁺/calmodulin binding to CaMKII (four Ca²⁺ ions bind cooperatively to one calmodulin molecule), sharpening the calcium-to-plasticity threshold toward the all-or-nothing regime observed experimentally (Lisman et al., 2002).

The rate function is defined as the integral of calcium-above-baseline:

$$
\hat{r}(I) = \frac{k}{n_H} \cdot \text{softplus}\!\left(\frac{n_H \cdot (I - I_{1/2})}{k}\right) - \text{Ca}_0 \cdot I
$$

so that its derivative is exactly the calcium excess:

$$
\hat{r}'(I) = \text{Ca}(I) - \text{Ca}_0
$$

With the defaults $\text{Ca}_0 = 0.5$, $I_{1/2} = I_\theta$, $n_H = 1$, the derivative simplifies to $\frac{1}{2}\tanh\!\left(\frac{I - I_\theta}{2k}\right)$ and the rate to $k \cdot \ln\cosh\!\left(\frac{I - I_\theta}{2k}\right)$ — a smooth threshold function.

**Key bio-plausible properties:**

- **Monotonically increasing derivative**: unlike the LIF or softplus derivatives (which decrease for strong inputs due to rate saturation), $\hat{r}'(I)$ increases monotonically — stronger synapses produce more calcium and get more eligibility. This matches the biological fact that calcium levels do not decrease for stronger inputs.
- **Sign change at baseline**: $\hat{r}'(I) < 0$ when $\text{Ca}(I) < \text{Ca}_0$ (subthreshold, LTD direction); $\hat{r}'(I) > 0$ when $\text{Ca}(I) > \text{Ca}_0$ (suprathreshold, LTP direction). This is a BCM-like calcium threshold mechanism (Bienenstock, Cooper & Munro, 1982), with the modification threshold set by the baseline $\text{Ca}_0$.
- **Cooperativity via $n_H$**: increasing $n_H$ sharpens the calcium transition around $I_{1/2}$ without shifting the midpoint, modeling the cooperative Ca²⁺/CaM → CaMKII activation. At $n_H = 4$, the threshold approaches the steep, switch-like behavior of CaMKII T286 autophosphorylation.
- **No rate saturation**: the rate grows approximately linearly for $I \gg I_\theta$ (no refractory-period ceiling), unlike the LIF rate.

**Comparison** (operating range $w \in [0.5, 0.8]$, $I_{1/2} = I_\theta$, $k = 0.5 I_\theta$, $\text{Ca}_0 = 0.5$):

| Weight | $I_{\text{eff}}$ | LIF $\hat{r}'(I)$ | NMDA $\hat{r}'(I)$, $n_H{=}1$ | NMDA $\hat{r}'(I)$, $n_H{=}4$ |
|--------|------|------|------|------|
| 0.50 | 0.058 | 0.130 (↓ with $I$) | +0.121 | +0.018 |
| 0.65 | 0.076 | 0.095 | +0.280 | +0.392 |
| 0.80 | 0.093 | 0.078 | +0.380 | +0.496 |

At $n_H = 4$, the derivative is near-zero for barely-suprathreshold weights ($w = 0.5$) and large for clearly suprathreshold weights ($w = 0.8$), producing sharper differentiation between weak and strong synapses.

#### Spike-Based Eligibility (`"spike"`)

A reward-modulated Hebbian mode that bypasses the analytical surrogate entirely. Instead of computing $\hat{r}'(I_{\text{eff}}) \cdot c$, the eligibility signal uses the **actual spike count** $n_a$ recorded from the SNN simulation:

$$
\text{eligibility}_a = \bigl(\mathbb{1}[a = a^*] - \pi(a)\bigr) \cdot \frac{n_a}{\text{time\_steps}}
$$

The score function $(\mathbb{1}[a=a^*] - \pi(a))$ is retained for credit assignment (chosen actions strengthened, non-chosen weakened). The spike count — normalized to a firing rate — replaces the analytical derivative as the magnitude signal.

**Tradeoffs vs surrogate modes:**

| Property                   | Surrogate (`lif`, `nmda`, ...)         | Spike-based                                               |
| -------------------------- | -------------------------------------- | --------------------------------------------------------- |
| Eligibility signal         | Analytical $\hat{r}'(I) \cdot c$       | Empirical $n_a / \text{time\_steps}$                      |
| Captures full SNN dynamics | No (first-order $I_{\text{eff}}$ only) | **Yes** (impulse waveform, refrac, astrocyte, inhibition) |
| Differentiable w.r.t. $w$  | Yes                                    | No                                                        |
| Policy gradient guarantee  | Yes                                    | No (score-weighted Hebbian)                               |
| Subthreshold synapses      | Can strengthen silent synapses         | Cannot (zero spikes → zero eligibility)                   |
| Extra computation          | Surrogate formula                      | None (spikes already computed)                            |

The spike-based mode is the most bio-plausible option — the eligibility signal IS the post-synaptic activity, gated by the score function for credit assignment. It captures all network dynamics (including astrocyte modulation, inhibition, refractory effects) that the analytical $I_{\text{eff}}$ approximation ignores. However, it cannot reinforce synapses whose neurons did not fire ($n_a = 0 \Rightarrow \text{eligibility} = 0$), limiting its ability to discover new pathways.

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
| `reinforce.surrogate.type` | `"lif"` | Surrogate kind: `"lif"`, `"softplus"`, `"nmda"`, or `"spike"` |
| `reinforce.surrogate.tc_decay` | 150.0 | LIF membrane time constant (LIF only) |
| `reinforce.surrogate.scale` | 1.0 | Softplus steepness (softplus only) |
| `reinforce.surrogate.I_theta` | derived | Softplus knee current (softplus only) |
| `reinforce.surrogate.I_half` | $I_\theta$ | NMDA calcium half-activation current (NMDA only) |
| `reinforce.surrogate.k` | $0.5 I_\theta$ | NMDA calcium slope factor (NMDA only) |
| `reinforce.surrogate.ca_baseline` | 0.5 | NMDA baseline calcium level (NMDA only) |
| `reinforce.surrogate.n_hill` | 1.0 | NMDA Hill cooperativity coefficient (NMDA only) |


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


## 10. Biological Interpretation

This section speculates on how the mathematical machinery above maps onto known neurobiology. These are analogies, not claims about the implementation's biological accuracy — but they motivate the architecture and suggest directions for future refinement.

### 10.1 The Three-Factor Rule

The REINFORCE update:

$$
\Delta w_{ij} = \eta_{\text{RL}} \cdot \underbrace{(G - \bar{b})}_{\text{factor 3: neuromodulator}} \cdot \underbrace{E_{ij}}_{\text{factors 1+2: Hebbian eligibility}}
$$

is an instance of a **three-factor learning rule**, a framework widely proposed for biological reinforcement learning (Frémaux & Gerstner, 2016). The three factors are:

1. **Pre-synaptic activity** — encoded in $I_{\text{eff}}(a) = c \cdot w_{s,a}$, which depends on whether the pre-synaptic neuron (at the current position) is spiking.
2. **Post-synaptic activity** — encoded in the surrogate rate $\hat{r}(I_{\text{eff}}(a))$, which predicts the post-synaptic neuron's firing probability.
3. **Neuromodulatory signal** — the advantage $A = G - \bar{b}$, broadcast globally to all synapses.

In the mammalian brain, this architecture is most clearly realized in the **striatum**: cortico-striatal synapses carry factors 1 and 2 (glutamatergic pre/post coincidence), while dopaminergic projections from the VTA/SNc deliver factor 3. Synaptic plasticity is gated by the convergence of cortical input and dopamine — without dopamine, Hebbian coincidence alone produces weak or transient changes; with dopamine, those changes are consolidated.

### 10.2 Eligibility Traces as Synaptic Tags

The eligibility trace:

$$
E_{ij}[t] = \lambda_{\text{trace}} \cdot E_{ij}[t-1] + \nabla_{w_{ij}} \log \pi(a^*[t])
$$

serves as a **temporary synaptic tag** that marks a synapse as "recently relevant" for future modulation. This is directly analogous to the molecular eligibility traces hypothesized in biological synapses. The three operations — decay, gradient increment, and reward-gated readout — each map onto distinct biochemical processes, organized as a **cascade of increasing timescale**:

$$
\text{NMDA}/\text{Ca}^{2+} \xrightarrow{\sim 50\text{ ms}} \text{CaMKII} \xrightarrow{\sim 30\text{ s}} \text{PKM-}\zeta \xrightarrow{\text{hours}} \text{Structural plasticity}
$$

#### Exponential decay: calcium extrusion and CaMKII dephosphorylation

The decay $\lambda_{\text{trace}} \cdot E[t-1]$ is implemented by two parallel first-order processes:

- **Fast (per-spike, ~50–500 ms):** Calcium entering through NMDA receptors is extruded by PMCA (plasma membrane Ca²⁺-ATPase) and NCX (Na⁺–Ca²⁺ exchanger). The extrusion follows approximately first-order kinetics, $[\text{Ca}^{2+}][t] \propto e^{-t/\tau_{\text{ext}}}$, providing the short-timescale decay.
- **Slow (per-step, ~10–60 s):** CaMKII, once autophosphorylated at T286 by a strong calcium transient, remains autonomously active after calcium returns to baseline. It is gradually dephosphorylated by PP1 (protein phosphatase 1) with first-order kinetics, giving $P_{\text{CaMKII}}[t] \propto e^{-t/\tau_{\text{PP1}}}$. This is the molecular analog of our $\lambda_{\text{trace}} = 0.95$ per step (effective half-life ≈ 14 steps).

The two-tier decay means the biological eligibility signal is actually a **sum of two exponentials** (fast calcium + slow CaMKII), not a single exponential as in our model. A double-exponential trace could capture both intra-step (spike-level coincidence) and inter-step (action-level credit) timescales — a potential model refinement.

#### Gradient increment: NMDA-mediated calcium influx

Each time the pre-synaptic neuron spikes and the post-synaptic membrane is depolarized, NMDA receptors admit calcium into the spine. The influx magnitude at synapse $i$ is:

$$
\Delta\text{Ca}_i \propto g_{\text{NMDA}}(V_{\text{post}}) \cdot s_{\text{pre},i}
$$

Since $g_{\text{NMDA}}(V)$ is steepest near threshold (Section 10.5), the calcium influx naturally encodes the surrogate derivative $\hat{r}'(I_{\text{eff}})$: synapses whose input brings the post-synaptic neuron near threshold produce maximal calcium, while far-subthreshold synapses produce little. The calcium is added to the spine's residual pool, incrementing the eligibility tag. This is the biophysical implementation of the accumulation $+ \hat{r}'(I_{\text{eff}}) \cdot c$.

#### The score function $(\mathbb{1}[a=a^*] - \pi(a))$: Hebbian coincidence and competition

This term has two components requiring different mechanisms:

**Positive component $\mathbb{1}[a=a^*]$:** The post-synaptic neuron that drove the chosen action was highly active → strong depolarization → strong NMDA activation → large calcium influx → strong positive eligibility increment. This is standard Hebbian coincidence detection, implemented by every NMDA-dependent plasticity mechanism.

**Negative component $-\pi(a)$:** This is the competitive normalization — all candidate synapses receive a negative (depressive) contribution proportional to their action probability. Three candidate mechanisms:

- **Lateral inhibition (our layer I):** The inhibitor layer provides recurrent inhibition to Y proportional to overall population activity. This inhibition hyperpolarizes non-chosen neurons, reducing their NMDA activation and calcium influx. The net calcium signal at each synapse becomes $g_{\text{NMDA}}(V_a^{\text{exc}} - V_a^{\text{inh}}) \cdot s_{\text{pre}}$, approximating the subtraction of expected activity.
- **Tonic dopamine baseline:** If tonic dopamine sets a zero-advantage baseline, the expected weight change averages to zero across episodes. The $-\pi(a)$ term ensures this cancellation at the trace level; tonic dopamine achieves a similar normalization in expectation.
- **Short-term synaptic depression:** Frequently-active release sites deplete readily releasable vesicle pools. A synapse supporting a high-$\pi(a)$ neuron has been releasing frequently → depleted → reduced effective calcium per spike → naturally reduced eligibility, implementing an implicit $-\pi(a)$ scaling through vesicle dynamics on a seconds timescale.

#### Cross-step accumulation: PKM-ζ as the slow trace

The deepest challenge is that our eligibility trace accumulates across **navigation steps** (each separated by seconds of real time), but calcium and CaMKII decay too fast to bridge this gap. The biological candidate for the slow accumulating trace is **PKM-ζ (protein kinase M zeta)**:

- PKM-ζ is a constitutively active fragment of PKC that is *synthesized* (not just activated) in response to strong plasticity induction. Its expression follows the cascade CaMKII → MAPK → CREB → local PKM-ζ mRNA translation at the synapse.
- Once synthesized, PKM-ζ persistently increases AMPA receptor trafficking to the potentiated synapse, maintaining LTP for hours.
- PKM-ζ synthesis rate is proportional to the CaMKII tag strength — so the PKM-ζ level *integrates* the history of recent CaMKII activation across multiple events, with synthesis adding and degradation (proteasome) providing the leak.
- This maps directly to $E[t] = \lambda \cdot E[t-1] + \nabla_w \log\pi$: PKM-ζ protein level is the eligibility state $E$, its degradation is $\lambda$, and CaMKII-driven synthesis is the gradient increment.

#### Reward readout: the dopamine × tag AND gate

When the episode ends and the advantage $A = G - \bar{b}$ arrives as a dopamine signal, it gates consolidation of the molecular tag:

- **$A > 0$ (dopamine burst):** D1 receptor activation → cAMP ↑ → PKA → DARPP-32 → PP1 inhibition → CaMKII protected → PKM-ζ synthesis proceeds → LTP consolidated. The magnitude is proportional to both dopamine ($A$) and the tag ($E$) — a **biochemical AND gate**.
- **$A < 0$ (dopamine dip):** Reduced D1 activation → PP1 active → CaMKII dephosphorylated → calcineurin (PP2B) activated → AMPAR dephosphorylation → LTD.
- **$A \approx 0$:** Tag decays normally → no PKM-ζ synthesis → no lasting change.

Neither signal alone produces lasting change; both must be present — the molecular implementation of the multiplication $A \cdot E$.

#### Summary: a multi-tier cascade, not a single number

The eligibility trace is **not stored as a single scalar** at each synapse. It is distributed across a cascade of molecular states with increasing timescale and decreasing reversibility:

| Model variable | Biological analog | Timescale | Mechanism |
|----------------|-------------------|-----------|-----------|
| Per-spike gradient $\hat{r}'(I) \cdot c$ | Ca²⁺ influx | ~50 ms | NMDA conductance × Mg²⁺ unblock |
| Per-step tag (fast decay) | CaMKII T286 phosphorylation | ~30 s | Autophosphorylation / PP1 dephosphorylation |
| Cross-step trace $E$ (slow decay) | PKM-ζ protein level | minutes–hours | mRNA translation / proteasome degradation |
| $\lambda_{\text{trace}}$ decay | Tag decay | seconds–minutes | Ca²⁺ extrusion, PP1, proteasome |
| $(\mathbb{1}[a=a^*])$ positive score | Hebbian coincidence | per-spike | Pre × post depolarization → Ca²⁺ |
| $(-\pi(a))$ negative score | Competitive normalization | per-step | Lateral inhibition, vesicle depletion |
| $A \cdot E$ reward gating | DA × tag AND gate | episode end | D1/cAMP/PKA/DARPP-32 × CaMKII/PKM-ζ |
| Reset after REINFORCE update | Tag consumption | per-episode | Consolidation stabilizes AMPAR; tag role complete |

The single-exponential trace in our implementation is a mathematical simplification of this multi-tier biochemical cascade.

### 10.3 Advantage as Dopamine Reward Prediction Error

The advantage $A = G - \bar{b}$ is the **reward prediction error** (RPE). Its biological counterpart is the phasic firing of midbrain dopamine neurons:

- **Positive advantage** ($A > 0$): outcome exceeds expectation → dopamine burst → potentiates eligible synapses (LTP). In our model: $w \mathrel{+}= \eta \cdot A^+ \cdot E$, strengthening connections that contributed to the rewarded trajectory.
- **Negative advantage** ($A < 0$): outcome falls short → dopamine dip (pause in tonic firing) → depresses eligible synapses (LTD). In our model: $w \mathrel{+}= \eta \cdot A^- \cdot E$, weakening connections that led to the penalized trajectory.
- **Zero advantage** ($A \approx 0$): outcome matches expectation → no phasic dopamine → no net change. The synapse's Hebbian tag decays uneventfully.

The running baseline $\bar{b}$ (updated with $\beta_b = 0.01$, a slow EMA) plays the role of the brain's **learned reward expectation** — analogous to the reward prediction encoded in the activity of striatal medium spiny neurons and orbitofrontal cortex, which gradually adapts to the statistics of the environment. The fact that dopamine neurons signal RPE (not absolute reward) is one of the most robust findings in systems neuroscience (Schultz, 1998), and our advantage formulation replicates this exactly.

### 10.4 Surrogate Gradient as f-I Curve Sensitivity

The surrogate rate $\hat{r}(I_{\text{eff}})$ and its derivative $\hat{r}'(I_{\text{eff}})$ have a natural biological reading:

- The **firing rate vs. current (f-I) curve** of a biological neuron is smooth and sigmoidal — it does not have the discontinuous step of an idealized spike threshold. This smoothness arises from noise (ion channel stochasticity, synaptic noise), adaptation currents, and the fact that biological circuits operate in a fluctuation-driven regime where the concept of a fixed threshold is an approximation.
- The **surrogate rate** $\hat{r}(I)$ approximates this smooth f-I relationship. By using it for gradient computation while keeping hard spikes for dynamics, we are effectively saying: "the neuron's output is a spike train, but the *sensitivity* of its output to input changes is well-approximated by the slope of its f-I curve."
- The **derivative** $\hat{r}'(I_{\text{eff}}(a))$ in the eligibility gradient encodes how much a small increase in weight $w_{s,a}$ would increase the post-synaptic firing rate. Biologically, this is related to the concept of **synaptic efficacy** — how effectively a pre-synaptic input can drive the post-synaptic neuron. Synapses far below threshold ($I_{\text{eff}} \ll I_\theta$) have near-zero $\hat{r}'$ (subthreshold, negligible influence on output), while synapses near threshold have high $\hat{r}'$ (maximal influence). This mirrors the experimentally observed gradient of synaptic influence around the post-synaptic neuron's firing threshold.

The divergence of $\hat{r}'(I)$ near $I_\theta$ (Section 6.4) is biologically interesting: it implies that synapses operating right at the post-synaptic threshold are the most "educable" — small weight changes produce maximal changes in output. This resonates with the notion of **metaplasticity** (Abraham, 2008), where synapses near threshold exhibit the highest plasticity because they are at the boundary between silent and active states, where the neuron is most sensitive to synaptic modifications.

### 10.5 How the Surrogate Derivative Could Be Implemented in the Brain

The surrogate derivative $\hat{r}'(I_{\text{eff}}) \cdot c$ answers a specific question at each synapse: *"how much does a change in my strength alter the post-synaptic neuron's output?"* The brain never evaluates this expression symbolically. Instead, the same sensitivity structure **emerges from biophysics** through several convergent mechanisms.

#### NMDA receptor voltage dependence — the built-in $\hat{r}'$ gate

The NMDA receptor conductance is magnesium-blocked at rest and progressively unblocks as the membrane depolarizes:

$$
g_{\text{NMDA}}(V) \propto \frac{1}{1 + [\text{Mg}^{2+}] \cdot e^{-V / 16.13}}
$$

This sigmoid is steepest right around spike threshold — precisely where $\hat{r}'(I)$ peaks in the surrogate. A synapse whose input brings the post-synaptic neuron *near* threshold produces maximal NMDA-mediated calcium influx; a far-subthreshold synapse produces almost none. The NMDA receptor doesn't "differentiate" the rate function — its biophysics gates calcium (and thus plasticity) in a pattern that naturally tracks the sensitivity profile of $\hat{r}'(I)$. This is the closest single-molecule analog to the surrogate derivative.

#### Calcium as the analog of $\hat{r}'$ magnitude

Post-synaptic calcium is the master variable controlling plasticity direction and magnitude (BCM theory; Bienenstock, Cooper & Munro, 1982):

- Low $[\text{Ca}^{2+}]$ → no change
- Moderate $[\text{Ca}^{2+}]$ → LTD (phosphatase pathway: calcineurin → PP1)
- High $[\text{Ca}^{2+}]$ → LTP (kinase pathway: CaMKII → AMPA insertion)

The calcium level at a given synapse depends on both glutamate binding and post-synaptic voltage (via NMDA receptors and voltage-gated Ca²⁺ channels). Near threshold, voltage fluctuations are amplified into large calcium transients — the calcium signal *embodies* $\hat{r}'(I)$: it encodes how effectively the synapse can influence the output. A subthreshold synapse sees little calcium (weak eligibility); a threshold-proximal synapse sees maximal calcium (strong eligibility). The sign (LTP vs LTD) is then set by the reward signal — the $(G - \bar{b})$ gating.

#### Dendritic nonlinearities compute the local rate curve

Pyramidal neuron dendrites are not passive cables — they possess active conductances (voltage-gated Na⁺, Ca²⁺ channels, NMDA spikes) that produce **local nonlinear integration**. Each dendritic branch computes a smooth input-output function:

$$
V_{\text{dendrite}} = f\!\left(\sum_i w_i \cdot s_i\right)
$$

where $f$ is differentiable (softened by channel noise and the spatially distributed NMDA receptors along the branch). Plasticity at each spine depends on the local dendritic voltage — a differentiable function of the local input. The dendrite thus implements, in analog hardware, exactly the kind of smooth input-output curve the surrogate models. The "derivative" is computed implicitly by the voltage dependence of NMDA receptors and Ca²⁺ channels within that branch, without any symbolic operation.

#### Membrane voltage as a rate proxy

At the single-synapse level, there is no access to the instantaneous firing rate. But there *is* access to the **post-synaptic membrane voltage** — a low-pass filtered integral of recent spike history. The voltage naturally encodes the recent rate, and voltage-dependent plasticity mechanisms (NMDA unblock, voltage-gated Ca²⁺ channels) effectively compute $d(\text{rate}) / d(\text{input})$ by responding to how much the voltage changes when input changes. The surrogate rate $\hat{r}(I)$ is an analytical model of what the voltage-encoded rate looks like at steady state.

#### The eligibility trace as molecular memory

Our eligibility trace accumulates the derivative over time with exponential decay ($\lambda_{\text{trace}} = 0.95$). The biological analog is the **residual calcium / CaMKII autophosphorylation** tag:

- NMDA-mediated calcium entry creates a transient tag that persists after the pre/post spikes have passed.
- CaMKII transitions to an autonomously active (autophosphorylated) state during high calcium, maintaining activity for seconds after the calcium decays — functioning as the "memory" of the derivative magnitude.
- The tag strength is proportional to the calcium transient amplitude, which (via the mechanisms above) is proportional to $\hat{r}'(I)$.
- When the reward / dopamine signal arrives later, it interacts with this tag — the three-factor convergence at the biochemical level.

#### Summary: the derivative is embodied, not computed

| Mathematical model | Biological analog | Mechanism |
|--------------------|-------------------|-----------|
| $\hat{r}'(I)$ magnitude → eligibility strength | $[\text{Ca}^{2+}]$ amplitude → tag strength | NMDA + VGCC calcium influx |
| $\hat{r}'(I) = \text{Ca}(I) - \text{Ca}_0$ (NMDA surrogate) | Calcium above tonic baseline | NMDA Mg²⁺ unblock minus resting Ca²⁺ |
| Sign change at $I_\theta$ (LTD ↔ LTP) | BCM modification threshold | Calcineurin (low Ca) vs CaMKII (high Ca) |
| $\hat{r}(I)$ smooth f-I curve | Dendritic spike rate | Active dendritic conductances |
| $c$ (weight → current) | Synaptic conductance → EPSP amplitude | Ohm's law at the synapse |
| $(G - \bar{b}) \cdot E$ three-factor rule | DA × Ca²⁺ tag convergence | D1/D2 receptor + CaMKII interaction |
| Trace decay $\lambda_{\text{trace}}$ | Tag lifetime | CaMKII dephosphorylation, Ca²⁺ extrusion |

The brain does not evaluate $\hat{r}'(I_{\text{eff}}) \cdot c$ as a formula. Ion channel biophysics produce a gating signal (calcium) whose amplitude naturally tracks the sensitivity of the neuron's output to each synapse's input. The NMDA baseline-calcium surrogate (`"nmda"` type) makes this mapping explicit: the derivative $\hat{r}'(I) = \text{Ca}(I) - \text{Ca}_0$ is modeled directly as calcium above a tonic baseline, with a BCM-like sign change at threshold driven by the baseline subtraction rather than by a sigmoid-saturation artifact. The mathematical surrogate is a **phenomenological model** of processes the brain implements through membrane dynamics and molecular signaling cascades.

### 10.6 Decoupled Surrogate and the Forward-Model Hypothesis

Our architecture uses **actual spikes for action selection** but a **surrogate rate for gradient computation** — the two are decoupled. This has an intriguing biological parallel in the **forward model / efference copy** framework:

- In motor control, the cerebellum maintains forward models that predict the sensory consequences of motor commands. These predictions enable rapid error-based learning (the error between predicted and actual outcome drives plasticity at parallel fiber–Purkinje cell synapses) without waiting for the full sensory feedback loop.
- Similarly, our surrogate rate predicts how a change in synaptic strength will alter the post-synaptic neuron's output, enabling gradient-based credit assignment through the non-differentiable spike threshold. The surrogate is a **local forward model** of the neuron's input-output mapping, maintained alongside the actual spiking dynamics.
- The fact that the surrogate is approximate (analytical LIF rate under constant-current assumptions, while the real input is a stochastic, time-varying impulse train) mirrors the inherent imprecision of biological forward models — they are good enough to guide learning, but not exact replicas of the plant.

### 10.7 STDP + REINFORCE: Local Hebbian Learning Gated by Global Neuromodulation

The coexistence of STDP (per-timestep, local, unsupervised) and REINFORCE (per-episode, global, reward-modulated) on the same synapses mirrors a major theme in computational neuroscience:

- **STDP as local correlation detection**: Spike-timing-dependent plasticity is a purely local mechanism — each synapse "measures" the relative timing of its pre- and post-synaptic spikes and adjusts accordingly. This is biologically implemented via NMDA receptor-mediated calcium dynamics and is well-documented in hippocampal and neocortical synapses.
- **REINFORCE as global modulation**: The reward signal arrives at the synapse as a diffuse, broadcast neuromodulator (dopamine, serotonin, or acetylcholine), carrying information that is not locally available. It cannot tell individual synapses what to do; it can only scale the changes that local mechanisms (STDP, eligibility) have already proposed.
- **Gating**: In our implementation, STDP proposes weight changes continuously, and REINFORCE applies a multiplicative gating (advantage × eligibility) at episode end. Biologically, dopamine is known to gate plasticity in the striatum: the same cortico-striatal input pattern can produce LTP, LTD, or no change depending on the timing and magnitude of dopamine receptor activation. The interaction between local Hebbian traces and global modulatory signals is considered a cornerstone of biological reinforcement learning.

The observed asymmetry in our system — STDP with $\nu = 10$ produces larger per-step weight changes than REINFORCE with $\eta_{\text{RL}} = 0.01$ — mirrors the biological observation that Hebbian plasticity is the "workhorse" of synaptic change, while neuromodulation plays a modulatory (amplifying, suppressing, or sign-flipping) role on top of it.

### 10.8 Astrocytes as Slow Eligibility Integrators

The astrocyte dynamics in this network — integrating pre-synaptic activity over long timescales ($\alpha = 0.001$, giving an effective time constant of $\sim$1000 timesteps) and modulating neuronal excitability via threshold changes — resonate with emerging views of astrocytes in neuroscience:

- **Slow integration**: Astrocytes are known to integrate synaptic activity over seconds to minutes via calcium signaling, far slower than neuronal dynamics. Our gliotransmitter variable $G$ plays this role: it slowly accumulates evidence of pre-synaptic activity and triggers a calcium event when a threshold is crossed.
- **Heterosynaptic modulation**: By lowering the spike threshold (increasing excitability) during calcium events, astrocytes effectively broaden the set of neurons that can participate in a given computation. This is conceptually related to **heterosynaptic plasticity**, where glial cells release gliotransmitters (glutamate, ATP, D-serine) that modulate synaptic strength at nearby synapses, independent of their individual pre/post activity.
- **Eligibility on a slower timescale**: The astrocyte calcium event ($Ca_{\text{duration}} = 100{,}000$ timesteps) creates a very slow "eligibility" window — once a region is activated, it remains excitable for a long time. This could be interpreted as a **homeostatic or meta-learning signal**: astrocytes maintain a slow memory of which regions of the network have been active, biasing future computation toward or away from those regions. In RL terms, this resembles a form of **intrinsic motivation** or **exploration bonus**: recently-active regions become more excitable, encouraging the agent to revisit and refine previously-used pathways.

### 10.9 The Temperature Parameter as Cortical State

The softmax temperature $T$ in the action-selection policy controls exploration vs. exploitation. In biological terms, this maps to **cortical state** — the balance between desynchronized (high T, exploratory, broad neural ensemble activation) and synchronized (low T, exploitative, sparse winner-take-all) cortical dynamics. Acetylcholine and norepinephrine are known to shift cortical state in this way, with high cholinergic tone promoting desynchronization and exploratory behavior. The temperature parameter can thus be viewed as a simplified model of these ascending arousal systems.

### 10.10 Tracing the Surrogate Derivative Backward: From Eligibility Trace to Ion Concentration

Starting from the question *"what biological variable stores the eligibility trace?"* and working backward at each step, we arrive at a chain of well-characterized molecular processes. Each link in the chain determines the next.

#### The backward chain

$$
\boxed{\hat{r}'(I) \;\longleftrightarrow\; \text{CaMKII activation} \;\leftarrow\; \text{Ca}^{2+}\text{/CaM} \;\leftarrow\; [\text{Ca}^{2+}]_{\text{spine}} \;\leftarrow\; g_{\text{NMDA}}(V) \;\leftarrow\; V_{\infty}(I) \;\leftarrow\; I_{\text{eff}}(w)}
$$

#### Step 1: Eligibility trace → CaMKII T286 phosphorylation

The eligibility trace must persist for seconds-to-minutes (the duration of a navigation episode). Among molecular candidates — free [Ca²⁺] (50–500 ms, too fast), cAMP/PKA (minutes, not synapse-specific), PKM-ζ (hours, too slow to accumulate) — **CaMKII T286 autophosphorylation** is the best match (Lisman et al., 2002):

- Autophosphorylated CaMKII remains active after calcium returns to baseline, persisting for 10–60 seconds — matching our $\lambda_{\text{trace}} = 0.95$ per step (half-life ≈ 14 steps).
- It is synapse-specific: restricted to the active spine by the spatial confinement of calcium and CaMKII anchoring at the postsynaptic density.
- It is reward-gated: dopamine → D1 → cAMP/PKA → DARPP-32 → PP1 inhibition protects the tag; dopamine dips → PP1 active → tag erased (Frémaux & Gerstner, 2016).

#### Step 2: CaMKII activation → Ca²⁺/calmodulin cooperativity

CaMKII is activated by Ca²⁺/calmodulin (CaM). The binding is highly cooperative — four Ca²⁺ ions bind to one CaM molecule — producing a steep nonlinearity described by a Hill function (Chin & Means, 2000):

$$
[\text{CaM}^*] \propto \frac{[\text{Ca}^{2+}]^{n_H}}{[\text{Ca}^{2+}]^{n_H} + K_d^{n_H}}, \qquad n_H \approx 3\text{--}4
$$

This cooperativity means CaMKII activation is nearly all-or-nothing: below a calcium threshold, barely active; above it, fully active. In our model, this is captured by the `n_hill` parameter: $n_H = 1$ gives a simple sigmoid; $n_H = 4$ approximates the CaM cooperativity, sharpening the BCM threshold.

The **sign of plasticity** is determined by the competition between two calcium-activated enzymes with different affinities (Graupner & Brunel, 2007):
- **Calcineurin (PP2B)**: high Ca²⁺ affinity, low cooperativity → activated at moderate calcium → **LTD** (dephosphorylates AMPARs via PP1).
- **CaMKII**: lower Ca²⁺ affinity, high cooperativity (through CaM) → activated only at high calcium → **LTP** (phosphorylates GluA1 at S831, increasing AMPAR conductance).

The balance point — where calcineurin gives way to CaMKII — is the BCM modification threshold, modeled by our `ca_baseline` parameter.

#### Step 3: Calcium level → NMDA receptor voltage dependence

The spine calcium concentration is primarily set by NMDA receptor-mediated influx during pre/post coincidence (Jahr & Stevens, 1990):

$$
\Delta[\text{Ca}^{2+}] \propto g_{\text{NMDA}}(V) \cdot [\text{glutamate}], \qquad g_{\text{NMDA}}(V) = \frac{1}{1 + [\text{Mg}^{2+}] \cdot e^{-\gamma V}}
$$

where $[\text{Mg}^{2+}] \approx 1$ mM and $\gamma \approx 0.062$ mV⁻¹. This conductance is the coincidence detector: it requires both pre-synaptic glutamate release AND post-synaptic depolarization (to unblock the Mg²⁺ pore). It is steepest near spike threshold — exactly where our $I_{1/2} = I_\theta$ places the calcium half-activation.

#### Step 4: Voltage → input current

The steady-state membrane voltage is set by the input current through the membrane equation:

$$
V_{\infty} = \frac{I_{\text{eff}}}{1 - d}, \qquad I_{\text{eff}} = c \cdot w_{s,a}
$$

This closes the loop: the weight $w_{s,a}$ determines the current, which determines the voltage, which determines the NMDA conductance, which determines the calcium influx, which determines the CaMKII activation, which determines the eligibility trace.

#### Summary: what biological quantity represents $\hat{r}'(I)$?

The most bio-plausible representation is **CaMKII-T286 phosphorylation level above its tonic baseline** — a real molecular state that is directly measurable (via phospho-specific antibodies), monotonically increasing with input strength, sign-determining (CaMKII vs calcineurin balance), and persistent at behavioral timescale.

Our NMDA surrogate models this entire cascade as a single parameterized function:

$$
\hat{r}'(I) = \sigma\!\left(\frac{n_H(I - I_\theta)}{k}\right) - \text{Ca}_0
$$

where each parameter maps to a biological quantity:

| Parameter | Biological analog | Source |
|-----------|-------------------|--------|
| $I_{1/2} = I_\theta$ | NMDA steepest at spike threshold | Jahr & Stevens (1990) |
| $k$ | Voltage-to-current scaling ($1/(1-d)$) | LIF membrane equation |
| $\text{Ca}_0$ | BCM modification threshold (calcineurin/CaMKII balance) | Bienenstock, Cooper & Munro (1982); Graupner & Brunel (2007) |
| $n_H$ | Ca²⁺/CaM cooperativity (Hill coefficient ≈ 4) | Chin & Means (2000); Lisman et al. (2002) |
| $\lambda_{\text{trace}}$ | PP1 dephosphorylation rate of CaMKII-T286 | Lisman et al. (2002) |


## 11. References

1. **Jahr, C. E. & Stevens, C. F.** (1990). A quantitative description of NMDA receptor-channel kinetic behavior. *Journal of Neuroscience*, 10(6), 1830–1837. — NMDA receptor Mg²⁺ unblock voltage dependence: $g_{\text{NMDA}}(V) = 1/(1 + [\text{Mg}^{2+}] \cdot e^{-0.062V})$.

2. **Bienenstock, E. L., Cooper, L. N. & Munro, P. W.** (1982). Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. *Journal of Neuroscience*, 2(1), 32–48. — BCM theory: sliding plasticity threshold $\theta_M \propto \bar{c}^p$; sign of plasticity determined by postsynaptic activity relative to threshold.

3. **Chin, D. & Means, A. R.** (2000). Calmodulin: a prototypical calcium sensor. *Trends in Cell Biology*, 10(8), 322–328. — Ca²⁺/calmodulin cooperative binding: four Ca²⁺ ions per CaM, Hill coefficient $n_H \approx 3$–4.

4. **Lisman, J., Schulman, H. & Cline, H.** (2002). The molecular basis of CaMKII function in synaptic and behavioural memory. *Nature Reviews Neuroscience*, 3(3), 175–190. — CaMKII T286 autophosphorylation as a molecular memory switch; CaMKII vs calcineurin threshold for LTP/LTD.

5. **Graupner, M. & Brunel, N.** (2007). STDP in a bistable synapse model indicates a path to depression in CA1 pyramidal cells. *PLoS Computational Biology*, 3(11), e221. — Calcium-based model of synaptic plasticity: calcineurin (LTD) vs CaMKII (LTP) pathways with distinct calcium affinities and cooperativities.

6. **Graupner, M. & Brunel, N.** (2012). Calcium-based plasticity model explains sensitivity of synaptic changes to spike timing, firing rate, synaptic conductance, and dendritic depolarization. *Physical Review E*, 84(5), 051907. — Unified calcium threshold model predicting STDP and BCM behavior from biophysical parameters.

7. **Frémaux, N. & Gerstner, W.** (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits*, 9, 85. — Three-factor learning rule framework: eligibility trace × neuromodulatory signal; dopamine gating of CaMKII-dependent plasticity.

8. **Shouval, H. Z., Bear, M. F. & Cooper, L. N.** (2002). A unified model of NMDA receptor-dependent bidirectional synaptic plasticity. *PNAS*, 99(16), 10831–10836. — Biophysical model linking calcium influx through NMDA receptors to BCM-like plasticity via CaMKII and calcineurin pathways.
