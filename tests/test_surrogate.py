import torch
import pytest

from astrocites.connection import Connection
from astrocites.learning import NoOp
from astrocites.surrogate import (
    impulse_integral,
    compute_scale_c,
    lif_rate,
    lif_rate_deriv,
    softplus_rate,
    softplus_rate_deriv,
    eligibility_gradient,
    build_lif_params,
)


class _MockLayer:
    def __init__(self, n):
        self.n = n
        self.shape = (n,)


# ---------------------------------------------------------------------------
# 1. Impulse integral: closed form vs. numerical simulation of the waveform
# ---------------------------------------------------------------------------

def _simulate_impulse_a_pre(impulse_amplitude, impulse_length, impulse_shape_factor):
    """Drive a real ``Connection.compute`` with a single spike and sum the
    ``a_pre`` buffer over the entire impulse window."""
    source = _MockLayer(1)
    target = _MockLayer(1)
    conn = Connection(source, target,
                      impulse_amplitude=impulse_amplitude,
                      impulse_length=impulse_length,
                      impulse_shape_factor=impulse_shape_factor,
                      invert=True, update_rule=NoOp, w=torch.ones(1, 1))
    total = 0.0
    for step in range(impulse_length):
        s = torch.tensor([True]) if step == 0 else torch.tensor([False])
        conn.compute(s)
        total += float(conn.a_pre.sum())
    return total


@pytest.mark.parametrize("A, L, k", [
    (0.5, 40, 0.9),
    (0.5, 40, 0.5),
    (1.0, 40, 0.9),
    (0.3, 50, 0.8),
    (0.5, 20, 0.7),
])
def test_impulse_integral_matches_numerical(A, L, k):
    numerical = _simulate_impulse_a_pre(A, L, k)
    analytical = impulse_integral(A, L, k, invert=True)
    assert abs(numerical - analytical) < 1e-4, (
        f"impulse_integral mismatch for A={A}, L={L}, k={k}: "
        f"numerical={numerical:.6f}, analytical={analytical:.6f}"
    )


def test_impulse_integral_default_params():
    assert abs(impulse_integral(0.5, 40, 0.9, invert=True) - 7.75) < 1e-10


# ---------------------------------------------------------------------------
# 2. LIF rate derivative: closed form vs. autograd
# ---------------------------------------------------------------------------

LIF_PARAMS = dict(thresh=7.0, decay=0.9934, tc_decay=150.0, refrac=40.0)


def test_lif_rate_is_zero_subthreshold():
    I_theta = LIF_PARAMS["thresh"] * (1.0 - LIF_PARAMS["decay"])
    I_sub = torch.tensor([I_theta * 0.5, I_theta * 0.99, 0.0])
    r = lif_rate(I_sub, **LIF_PARAMS)
    assert torch.all(r == 0.0)


def test_lif_rate_deriv_autograd():
    I_theta = LIF_PARAMS["thresh"] * (1.0 - LIF_PARAMS["decay"])
    current = torch.linspace(I_theta * 1.1, I_theta * 5.0, 50, requires_grad=True)
    r = lif_rate(current, **LIF_PARAMS)
    autograd_grad = torch.autograd.grad(r.sum(), current, create_graph=False)[0]
    manual_grad = lif_rate_deriv(current.detach(), **LIF_PARAMS)
    assert torch.allclose(autograd_grad, manual_grad, rtol=1e-4, atol=1e-6), (
        f"LIF rate derivative mismatch:\n autograd={autograd_grad}\n manual  ={manual_grad}"
    )


def test_lif_rate_deriv_zero_subthreshold():
    I_theta = LIF_PARAMS["thresh"] * (1.0 - LIF_PARAMS["decay"])
    I_sub = torch.tensor([I_theta * 0.5, 0.0])
    rprime = lif_rate_deriv(I_sub, **LIF_PARAMS)
    assert torch.all(rprime == 0.0)


# ---------------------------------------------------------------------------
# 3. Softplus rate derivative: closed form vs. autograd
# ---------------------------------------------------------------------------

def test_softplus_rate_deriv_autograd():
    current = torch.linspace(-2.0, 5.0, 80, requires_grad=True)
    params = dict(I_theta=0.05, scale=2.0)
    r = softplus_rate(current, **params)
    autograd_grad = torch.autograd.grad(r.sum(), current, create_graph=False)[0]
    manual_grad = softplus_rate_deriv(current.detach(), **params)
    assert torch.allclose(autograd_grad, manual_grad, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Full eligibility gradient: closed form vs. autograd on log softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind, params", [
    ("lif", LIF_PARAMS),
    ("softplus", dict(I_theta=0.05, scale=2.0)),
])
def test_eligibility_gradient_autograd(kind, params):
    torch.manual_seed(0)
    n_actions = 5
    c = 0.12
    temperature = 1.5
    chosen_idx = 2

    w_slice = torch.linspace(0.4, 0.9, n_actions, requires_grad=True)
    I_eff = c * w_slice
    adj_positions = list(range(n_actions))

    # Closed-form eligibility gradient
    manual = eligibility_gradient(adj_positions, chosen_idx, I_eff.detach(),
                                  c, temperature, kind, params)

    # Autograd reference: d/dw log softmax(r(c*w)/T)[chosen_idx]
    if kind == "lif":
        r = lif_rate(I_eff, **params)
    else:
        r = softplus_rate(I_eff, **params)
    log_probs = torch.log_softmax(r / temperature, dim=0)
    autograd = torch.autograd.grad(log_probs[chosen_idx], w_slice, create_graph=False)[0]

    assert torch.allclose(manual, autograd, rtol=1e-4, atol=1e-6), (
        f"Eligibility gradient mismatch ({kind}):\n manual  ={manual}\n autograd={autograd}"
    )


@pytest.mark.parametrize("chosen_idx", [0, 1, 2, 3, 4])
def test_eligibility_gradient_all_actions(chosen_idx):
    n_actions = 5
    c = 0.12
    temperature = 1.0
    w_slice = torch.linspace(0.4, 0.9, n_actions, requires_grad=True)
    I_eff = c * w_slice
    adj_positions = list(range(n_actions))

    manual = eligibility_gradient(adj_positions, chosen_idx, I_eff.detach(),
                                  c, temperature, "lif", LIF_PARAMS)
    r = lif_rate(I_eff, **LIF_PARAMS)
    log_probs = torch.log_softmax(r / temperature, dim=0)
    autograd = torch.autograd.grad(log_probs[chosen_idx], w_slice, create_graph=False)[0]
    assert torch.allclose(manual, autograd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Integration: scale constant and build_lif_params helpers
# ---------------------------------------------------------------------------

def test_compute_scale_c():
    c = compute_scale_c(0.5, 40, 0.9, True, intensity=15.0, time_steps=1000)
    expected = 7.75 * 15.0 / 1000.0
    assert abs(c - expected) < 1e-10


def test_build_lif_params():
    p = build_lif_params(thresh=7.0, dt=1.0, tc_decay=150.0, refrac=40)
    assert p["thresh"] == 7.0
    assert p["refrac"] == 40.0
    assert p["tc_decay"] == 150.0
    expected_decay = float(torch.exp(torch.tensor(-1.0 / 150.0)).item())
    assert abs(p["decay"] - expected_decay) < 1e-12


# ---------------------------------------------------------------------------
# 6. End-to-end: the rates produce plausible spike counts for default params
# ---------------------------------------------------------------------------

def test_lif_rate_plausible_counts():
    """For default params and weights in [0.5, 0.8], the surrogate rate should
    predict a handful of spikes over 1000 timesteps (matching what the SNN
    would produce)."""
    p = build_lif_params(thresh=7.0, dt=1.0, tc_decay=150.0, refrac=40)
    c = compute_scale_c(0.5, 40, 0.9, True, intensity=15.0, time_steps=1000)
    time_steps = 1000
    for w_val in [0.5, 0.65, 0.8]:
        I_eff = torch.tensor([c * w_val])
        rate = lif_rate(I_eff, **p)
        expected_spikes = (rate * time_steps).item()
        assert expected_spikes > 0, f"Expected spikes for w={w_val} but got 0"
        assert expected_spikes < 100, f"Too many spikes ({expected_spikes}) for w={w_val}"
