import torch


def impulse_integral(impulse_amplitude, impulse_length, impulse_shape_factor, invert=True):
    """Closed-form total ``a_pre`` area delivered by a single pre-synaptic spike.

    This is the sum of the cumulative ``a_pre`` buffer over the full impulse
    waveform window (``impulse_length - 1`` timesteps during which the impulse
    state is non-zero). For ``invert=True`` (the biphasic waveform used by
    ``conn_XY``), the derivation yields:

        impulse_integral = A * (L * (2k - 1) - 1) / 2

    where ``A`` is ``impulse_amplitude``, ``L`` is ``impulse_length`` and ``k``
    is ``impulse_shape_factor``.
    """
    A = float(impulse_amplitude)
    L = float(impulse_length)
    k = float(impulse_shape_factor)
    if invert:
        return A * (L * (2.0 * k - 1.0) - 1.0) / 2.0
    else:
        raise NotImplementedError("impulse_integral for invert=False is not used by conn_XY")


def compute_scale_c(impulse_amplitude, impulse_length, impulse_shape_factor, invert,
                    intensity, time_steps):
    """Scale constant mapping a weight ``w[s, a]`` to the effective per-timestep
    input current ``I_eff(a) = c * w[s, a]``.

    ``c = impulse_integral * intensity / time_steps``: the per-spike charge area
    times the expected number of input spikes (``intensity``) spread over the
    simulation window (``time_steps``).
    """
    ii = impulse_integral(impulse_amplitude, impulse_length, impulse_shape_factor, invert)
    return ii * float(intensity) / float(time_steps)


def lif_rate(current, thresh, decay, tc_decay, refrac, eps=1e-8):
    """Analytical steady-state firing rate of a LIF neuron under constant current.

    Neuron model (matches ``LIFNodes`` with ``rest = reset = 0``):

        v[t+1] = decay * v[t] + I ;  spike when v >= thresh ;  reset to 0

    The minimum current needed to ever reach threshold at steady state is
    ``I_theta = thresh * (1 - decay)``. For ``I > I_theta`` the inter-spike
    interval is ``refrac + tc_decay * ln(I / (I - I_theta))``, giving rate
    ``r(I) = 1 / interval``. For ``I <= I_theta`` the rate is zero. The function
    is continuous (``r -> 0`` as ``I -> I_theta+``).

    Parameters
    ----------
    current : torch.Tensor
        Effective input currents (one per candidate action).
    """
    I_theta = thresh * (1.0 - decay)
    suprathreshold = current > I_theta
    safe = torch.where(suprathreshold, current, torch.full_like(current, I_theta + eps))
    denom = (safe - I_theta).clamp(min=eps)
    ratio = safe / denom
    interval = refrac + tc_decay * torch.log(ratio)
    rate = 1.0 / interval
    return torch.where(suprathreshold, rate, torch.zeros_like(rate))


def lif_rate_deriv(current, thresh, decay, tc_decay, refrac, eps=1e-8):
    """Closed-form ``dr/dI`` for :func:`lif_rate`.

        f(I)  = refrac + tc_decay * ln(I / (I - I_theta))
        r(I)  = 1 / f(I)
        r'(I) = tc_decay * I_theta / (I * (I - I_theta) * f(I)^2)

    Zero for subthreshold inputs.
    """
    I_theta = thresh * (1.0 - decay)
    suprathreshold = current > I_theta
    safe = torch.where(suprathreshold, current, torch.full_like(current, I_theta + eps))
    diff = (safe - I_theta).clamp(min=eps)
    interval = refrac + tc_decay * torch.log(safe / diff)
    rprime = tc_decay * I_theta / (safe * diff * interval * interval)
    return torch.where(suprathreshold, rprime, torch.zeros_like(rprime))


def softplus_rate(current, I_theta, scale):
    """Smooth proxy surrogate rate: ``r(I) = softplus(scale * (I - I_theta))``."""
    return torch.nn.functional.softplus(scale * (current - I_theta))


def softplus_rate_deriv(current, I_theta, scale):
    """``dr/dI = scale * sigmoid(scale * (I - I_theta))``."""
    return scale * torch.sigmoid(scale * (current - I_theta))


def rate_and_deriv(current, kind, params):
    """Dispatch to the requested surrogate kind, returning ``(r, r')``.

    Parameters
    ----------
    current : torch.Tensor
        Effective input currents (one per candidate action).
    kind : str
        ``"lif"`` or ``"softplus"``.
    params : dict
        Keyword arguments forwarded to the chosen surrogate functions.
        For ``"lif"``: ``thresh``, ``decay``, ``tc_decay``, ``refrac``.
        For ``"softplus"``: ``I_theta``, ``scale``.
    """
    if kind == "lif":
        return lif_rate(current, **params), lif_rate_deriv(current, **params)
    elif kind == "softplus":
        return softplus_rate(current, **params), softplus_rate_deriv(current, **params)
    else:
        raise ValueError(f"Unknown surrogate kind: {kind!r}")


def eligibility_gradient(adj_positions, chosen_idx, I_eff, c, temperature, kind, params):
    """Compute the closed-form REINFORCE eligibility gradient ``d log pi / d w[s, a]``
    for each candidate action ``a``, where the policy is
    ``pi(a) = softmax(r(I_eff(a)) / T)``.

    Returns a 1-D tensor of length ``len(adj_positions)`` aligned with
    ``adj_positions``: the gradient entries to scatter into ``eligibility[s, adj]``.

        d log pi(a*) / d w[s, a] = (1/T) * (1[a=a*] - pi(a)) * r'(I_eff(a)) * c
    """
    rate, rate_deriv = rate_and_deriv(I_eff, kind, params)
    probs = torch.softmax(rate / temperature, dim=0)
    onehot = torch.zeros_like(probs)
    onehot[chosen_idx] = 1.0
    return (1.0 / temperature) * (onehot - probs) * rate_deriv * c


def build_lif_params(thresh, dt, tc_decay, refrac):
    """Convenience: pack the neuron constants into the dict expected by the
    ``"lif"`` surrogate, computing ``decay`` from ``dt`` and ``tc_decay``."""
    decay = float(torch.exp(torch.tensor(-float(dt) / float(tc_decay))).item())
    return {"thresh": float(thresh), "decay": decay, "tc_decay": float(tc_decay), "refrac": float(refrac)}
