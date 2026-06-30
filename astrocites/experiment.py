import os
import random
import numpy as np
import torch
import scipy.io as sio
from time import time as t

from astrocites.nodes import Input, LIFNodes
from astrocites.connection import Connection
from astrocites.learning import WeightDependentPostPre, NoOp
from astrocites.network import Network, NetworkMonitor
from astrocites.utils import bernoulli_loader, create_adjacency_matrix
from astrocites.surrogate import compute_scale_c, eligibility_gradient, build_lif_params
from astrocites.logs import ExperimentLogger, FileLogger


def setup_and_run_simulation(
    NA, weights_mask_XY, weights_init_XY, weights_init_XI, n_steps,
    current_position, goal, learning_rate, wmin, wmax, weight_decay,
    post_spike_weight_decay, reset, refrac, thresh, intensity, time_steps, dt,
    enable_astrocyte, alpha, k
):
    network = Network(dt=dt)
    input_layer = Input(n=NA, traces=True, thresh=thresh, rest=reset, reset=reset, refrac=refrac)
    output_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac)
    inhibitor_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac, dt=dt,
                               enable_astrocyte=enable_astrocyte, alpha=alpha, k=k)
    conn_XY = Connection(input_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True,
                         update_rule=WeightDependentPostPre, w=weights_init_XY, nu=[10, 10],
                         wmin=wmin, wmax=wmax, weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay)
    conn_XI = Connection(input_layer, inhibitor_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                         w=weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                         weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    conn_IY = Connection(inhibitor_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                          impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                          w=-weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                          weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    network.add_layer(input_layer, 'X')
    network.add_layer(output_layer, 'Y')
    network.add_layer(inhibitor_layer, 'I')
    network.add_connection(conn_XY, 'X', 'Y')
    network.add_connection(conn_XI, 'X', 'I')
    network.add_connection(conn_IY, 'I', 'Y')
    state_vars = ('v', 's', 'w', 'G', 'Ca') if enable_astrocyte else ('v', 's', 'w')
    global_monitor = NetworkMonitor(network, state_vars=state_vars)
    network.add_monitor(global_monitor, 'Network')
    start = t()
    positions = []
    for i in range(n_steps):
        positions.append(current_position)
        if current_position == goal:
            print("success")
            break
        adjacent_positions = []
        for j in range(NA):
            if weights_mask_XY[current_position, j] > 0 and j != current_position:
                adjacent_positions.append(j)
        input_data = torch.zeros(1, NA)
        input_data[0, current_position] = intensity
        sample = next(bernoulli_loader(data=(input_data / time_steps) * dt, time=time_steps)).float()
        inpts = {'X': sample}
        injects_v = {'I': torch.full((NA,), 0.02)}
        network.run(inpts=inpts, time=time_steps, injects_v=injects_v,
                    current_position=current_position, adjacent_positions=adjacent_positions,
                    conn_XY=conn_XY)
        recordings = network.monitors['Network'].get()
        spikes = np.asarray(recordings['Y']['s'])
        summed = np.squeeze(np.sum(spikes, axis=0))
        summed = summed + weights_mask_XY[current_position, :]
        max_val = np.max(summed)
        candidates = np.where((summed == max_val) & (weights_mask_XY[current_position, :] == 1))[0]
        candidates = [c for c in candidates if c != current_position]
        new_position = np.random.choice(candidates)
        network.connections['X_Y'].w.data *= torch.Tensor(weights_mask_XY).float()
        weights_2d = network.connections['X_Y'].w.detach().cpu().numpy()
        current_position = new_position
        network.reset_()
    elapsed = t() - start
    reached_goal = int(current_position == goal)
    return positions, weights_2d, reached_goal, elapsed


def setup_and_run_simulation_reinforce(
    NA, weights_mask_XY, weights_init_XY, weights_init_XI, n_steps,
    current_position, goal, learning_rate, wmin, wmax, weight_decay,
    post_spike_weight_decay, reset, refrac, thresh, intensity, time_steps, dt,
    enable_astrocyte, alpha, k,
    reinforce_lr, temperature, reward_goal, step_penalty,
    gamma, baseline_decay, policy_mix_beta, trace_decay,
    enable_stdp_during_training,
    surrogate_kind="lif", surrogate_params=None,
    running_baseline=0.0,
):
    network = Network(dt=dt)
    input_layer = Input(n=NA, traces=True, thresh=thresh, rest=reset, reset=reset, refrac=refrac)
    output_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac)
    inhibitor_layer = LIFNodes(n=NA, traces=True, thresh=thresh * torch.ones(NA), rest=reset, reset=reset, refrac=refrac, dt=dt,
                               enable_astrocyte=enable_astrocyte, alpha=alpha, k=k)
    conn_XY = Connection(input_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True,
                         update_rule=WeightDependentPostPre, w=weights_init_XY, nu=[10, 10],
                         wmin=wmin, wmax=wmax, weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                         baseline_decay=baseline_decay, policy_mix_beta=policy_mix_beta,
                         gamma=gamma, trace_decay=trace_decay, temperature=temperature)
    conn_XI = Connection(input_layer, inhibitor_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                         impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                         w=weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                         weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    conn_IY = Connection(inhibitor_layer, output_layer, impulse_amplitude=0.5, impulse_amplitude_2=0.5,
                          impulse_length=40, impulse_shape_factor=0.9, invert=True, update_rule=NoOp,
                          w=-weights_init_XI, nu=[learning_rate, learning_rate], wmin=-100, wmax=wmax,
                          weight_decay=0, post_spike_weight_decay=post_spike_weight_decay)
    network.add_layer(input_layer, 'X')
    network.add_layer(output_layer, 'Y')
    network.add_layer(inhibitor_layer, 'I')
    network.add_connection(conn_XY, 'X', 'Y')
    network.add_connection(conn_XI, 'X', 'I')
    network.add_connection(conn_IY, 'I', 'Y')
    state_vars = ('v', 's', 'w', 'G', 'Ca') if enable_astrocyte else ('v', 's', 'w')
    global_monitor = NetworkMonitor(network, state_vars=state_vars)
    network.add_monitor(global_monitor, 'Network')

    conn_XY.running_baseline = running_baseline
    mask_tensor = torch.Tensor(weights_mask_XY).float()

    use_surrogate = surrogate_kind != "spike"
    if use_surrogate:
        surrogate_c = compute_scale_c(conn_XY.impulse_amplitude, conn_XY.impulse_length,
                                      conn_XY.impulse_shape_factor, conn_XY.invert,
                                      intensity, time_steps)
        if surrogate_params is None:
            surrogate_params = build_lif_params(thresh, dt, float(output_layer.tc_decay), refrac)

    start = t()
    positions = []
    for i in range(n_steps):
        positions.append(current_position)
        if current_position == goal:
            print("success")
            break
        adjacent_positions = []
        for j in range(NA):
            if weights_mask_XY[current_position, j] > 0 and j != current_position:
                adjacent_positions.append(j)
        input_data = torch.zeros(1, NA)
        input_data[0, current_position] = intensity
        sample = next(bernoulli_loader(data=(input_data / time_steps) * dt, time=time_steps)).float()
        inpts = {'X': sample}
        injects_v = {'I': torch.full((NA,), 0.02)}
        network.run(inpts=inpts, time=time_steps, injects_v=injects_v,
                    current_position=current_position, adjacent_positions=adjacent_positions,
                    conn_XY=conn_XY, enable_stdp=enable_stdp_during_training)
        recordings = network.monitors['Network'].get()
        spikes = np.asarray(recordings['Y']['s'])
        summed = np.squeeze(np.sum(spikes, axis=0))
        w = network.connections['X_Y'].w.detach()
        weight_prefs = torch.tensor([w[current_position, a] for a in adjacent_positions], dtype=torch.float32)
        spike_prefs = torch.tensor([summed[a] for a in adjacent_positions], dtype=torch.float32)
        logits = policy_mix_beta * weight_prefs + (1 - policy_mix_beta) * spike_prefs
        probs = torch.softmax(logits / temperature, dim=0)
        probs = probs / probs.sum()
        dist = torch.distributions.Categorical(probs)
        chosen_idx = dist.sample().item()
        new_position = adjacent_positions[chosen_idx]

        step_reward = step_penalty
        if new_position == goal:
            step_reward += reward_goal

        if use_surrogate:
            w_slice = network.connections['X_Y'].w.data[current_position, adjacent_positions]
            I_eff = surrogate_c * w_slice
            grad_slice = eligibility_gradient(adjacent_positions, chosen_idx, I_eff, surrogate_c,
                                              temperature, surrogate_kind, surrogate_params)
        else:
            spike_rates = spike_prefs / time_steps
            onehot = torch.zeros_like(probs)
            onehot[chosen_idx] = 1.0
            grad_slice = (onehot - probs) * spike_rates
        conn_XY.accumulate_trace(current_position, adjacent_positions, grad_slice, step_reward)
        network.connections['X_Y'].w.data *= mask_tensor
        current_position = new_position
        network.reset_()

    elapsed = t() - start
    reached_goal = (current_position == goal)

    conn_XY.compute_and_apply_reinforce_update(lr=reinforce_lr, mask=mask_tensor)

    weights_2d = network.connections['X_Y'].w.detach().cpu().numpy()
    new_baseline = conn_XY.running_baseline
    return positions, weights_2d, int(reached_goal), elapsed, new_baseline


def run_experiment(config: dict, logger: ExperimentLogger = None):
    if logger is None:
        logger = FileLogger(output_dir=config.get("output_dir", "results"))

    N = config["grid_size"]
    NA = N * N
    weights_mask_XY = create_adjacency_matrix(N)
    n_steps = config["n_steps"]
    current_position = config["start_position"]
    goal = config["goal_position"]

    neuron_cfg = config.get("neuron", {})
    learning_rate = neuron_cfg.get("learning_rate", config.get("learning_rate", 1))
    wmin = neuron_cfg.get("wmin", config.get("wmin", 0.001))
    wmax = neuron_cfg.get("wmax", config.get("wmax", 1))
    weight_decay = neuron_cfg.get("weight_decay", config.get("weight_decay", 0))
    post_spike_weight_decay = neuron_cfg.get("post_spike_weight_decay", config.get("post_spike_weight_decay", 0.005))
    reset = neuron_cfg.get("reset", config.get("reset", 0))
    refrac = neuron_cfg.get("refrac", config.get("refrac", 40))
    thresh = neuron_cfg.get("thresh", config.get("thresh", 7))

    astro_cfg = config.get("astrocyte", {})
    enable_astrocyte = astro_cfg.get("enable", config.get("enable_astrocyte", True))
    alpha = astro_cfg.get("alpha", config.get("alpha", 0.001))
    k = astro_cfg.get("k", config.get("k", 0.2))

    sim_cfg = config.get("simulation", {})
    intensity = sim_cfg.get("intensity", config.get("intensity", 15.0))
    time_steps = sim_cfg.get("time_steps", config.get("time_steps", 1000))
    dt = sim_cfg.get("dt", config.get("dt", 1))

    reinf_cfg = config.get("reinforce", {})
    reinf_enable = reinf_cfg.get("enable", False)
    reinf_lr = reinf_cfg.get("learning_rate", 0.01)
    temperature = reinf_cfg.get("temperature", 1.0)
    reward_goal = reinf_cfg.get("reward_goal", 10.0)
    step_penalty = reinf_cfg.get("step_penalty", -0.1)
    gamma = reinf_cfg.get("gamma", 0.99)
    baseline_decay = reinf_cfg.get("baseline_decay", 0.01)
    policy_mix_beta = reinf_cfg.get("policy_mix_beta", 0.0)
    enable_stdp_during_training = reinf_cfg.get("enable_stdp_during_training", True)
    trace_decay = reinf_cfg.get("trace_decay", 0.95)

    surr_cfg = reinf_cfg.get("surrogate", {})
    surrogate_kind = surr_cfg.get("type", "lif")
    if surrogate_kind == "lif":
        decay = float(np.exp(-dt / surr_cfg.get("tc_decay", 150.0)))
        surrogate_params = {
            "thresh": thresh,
            "decay": surr_cfg.get("decay", decay),
            "tc_decay": surr_cfg.get("tc_decay", 150.0),
            "refrac": refrac,
        }
    elif surrogate_kind == "softplus":
        I_theta_default = thresh * (1.0 - float(np.exp(-dt / 150.0)))
        surrogate_params = {
            "I_theta": surr_cfg.get("I_theta", I_theta_default),
            "scale": surr_cfg.get("scale", 1.0),
        }
    elif surrogate_kind == "nmda":
        I_theta_default = thresh * (1.0 - float(np.exp(-dt / 150.0)))
        surrogate_params = {
            "I_half": surr_cfg.get("I_half", I_theta_default),
            "k": surr_cfg.get("k", 0.5 * I_theta_default),
            "ca_baseline": surr_cfg.get("ca_baseline", 0.5),
            "n_hill": surr_cfg.get("n_hill", 1.0),
        }
    elif surrogate_kind == "spike":
        surrogate_params = None
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_kind!r}")

    exp_cfg = config.get("experiment", {})
    num_cycles = exp_cfg.get("num_cycles", config.get("num_cycles", 20))
    num_experiments = exp_cfg.get("num_experiments", config.get("num_experiments", 5))

    all_experiment_results = []

    for exp_idx in range(num_experiments):
        experiment_num = exp_idx + 1
        print(f"\nEXPERIMENT {experiment_num}/{num_experiments}")
        print("-" * 20)

        seeds = exp_cfg.get("seeds", [])
        if seeds and exp_idx < len(seeds):
            seed = seeds[exp_idx]
        else:
            seed = random.randint(0, 2**31 - 1)

        config['seed'] = seed
        logger.log_params(config)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        weights_rand_dist_XY = np.random.normal(0.65, 0.1, size=(NA, NA))
        weights_rand_dist_XY[weights_rand_dist_XY > 0.8] = 0.8
        weights_rand_dist_XY[weights_rand_dist_XY < 0.5] = 0.5
        weights_init_XY = torch.Tensor(weights_mask_XY * weights_rand_dist_XY).float()
        weights_init_XI = torch.eye(NA)

        all_routes = []
        all_weight_matrices = []
        route_lengths = []

        QAZ_initial = weights_rand_dist_XY * weights_mask_XY
        all_weight_matrices.append(QAZ_initial.copy())

        print(f"\n{'=' * 60}")
        print("INITIAL RUN (BASELINE)")
        print(f"{'=' * 60}")

        positions, weights_2d, goal_reached, elapsed_time = setup_and_run_simulation(
            NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
            weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
            goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
            weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
            reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
            time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
        )
        all_routes.append(f"Route (initial baseline): {positions}")
        route_lengths.append(len(positions) - 1)

        logger.log_metrics({
            "initial/route_length": len(positions) - 1,
            "initial/goal_reached": goal_reached,
            "initial/elapsed_time": elapsed_time
        }, step=0)

        running_baseline = 0.0

        for cycle_idx in range(num_cycles):
            cycle_num = cycle_idx + 1

            print(f"\n{'=' * 60}")
            print(f"TRAINING CYCLE {cycle_num}/{num_cycles}")
            print(f"{'=' * 60}")

            if reinf_enable:
                positions_train, weights_2d_train, goal_reached_train, elapsed_time_train, running_baseline = setup_and_run_simulation_reinforce(
                    NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                    weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                    goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                    weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                    reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                    time_steps=time_steps, dt=dt, enable_astrocyte=enable_astrocyte, alpha=alpha, k=k,
                    reinforce_lr=reinf_lr, temperature=temperature,
                    reward_goal=reward_goal, step_penalty=step_penalty,
                    gamma=gamma, baseline_decay=baseline_decay,
                    policy_mix_beta=policy_mix_beta, trace_decay=trace_decay,
                    enable_stdp_during_training=enable_stdp_during_training,
                    surrogate_kind=surrogate_kind, surrogate_params=surrogate_params,
                    running_baseline=running_baseline,
                )
            else:
                positions_train, weights_2d_train, goal_reached_train, elapsed_time_train = setup_and_run_simulation(
                    NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                    weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                    goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                    weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                    reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                    time_steps=time_steps, dt=dt, enable_astrocyte=enable_astrocyte, alpha=alpha, k=k,
                )

            QAZ_after_train = weights_2d_train * weights_mask_XY
            all_weight_matrices.append(QAZ_after_train.copy())
            weights_init_XY = torch.Tensor(weights_2d_train).float()
            all_routes.append(f"Route (training, cycle {cycle_num}): {positions_train}")
            route_lengths.append(len(positions_train) - 1)

            print(f"\n{'=' * 60}")
            print(f"VERIFICATION CYCLE {cycle_num}/{num_cycles}")
            print(f"{'=' * 60}")

            positions_verify, weights_2d_verify, goal_reached_verify, elapsed_time_verify = setup_and_run_simulation(
                NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
            )
            all_routes.append(f"Route (verification, cycle {cycle_num}): {positions_verify}")
            route_lengths.append(len(positions_verify) - 1)

            metrics = {
                "train/route_length": len(positions_train) - 1,
                "train/goal_reached": goal_reached_train,
                "train/elapsed_time": elapsed_time_train,
                "verify/route_length": len(positions_verify) - 1,
                "verify/goal_reached": goal_reached_verify,
                "verify/elapsed_time": elapsed_time_verify,
            }
            logger.log_metrics(metrics, step=cycle_num)

        logger.start_experiment(f"experiment_{experiment_num}")
        results_dir = str(logger.active_exp_dir)

        weights_filename = os.path.join(results_dir, "weight_matrices.txt")
        with open(weights_filename, 'w') as f:
            f.write(f"WEIGHT MATRICES: EXPERIMENT RESULTS #{experiment_num}\n")
            f.write("=" * 50 + "\n\n")
            for i, weight_matrix in enumerate(all_weight_matrices):
                if i == 0:
                    f.write("INITIAL WEIGHT MATRIX:\n")
                else:
                    f.write(f"WEIGHT MATRIX AFTER TRAINING CYCLE {i}:\n")
                np.savetxt(f, weight_matrix, fmt='%.4f')
                f.write("\n" + "-" * 30 + "\n\n")

        routes_filename = os.path.join(results_dir, "navigation_routes.txt")
        with open(routes_filename, 'w') as f:
            f.write(f"NAVIGATION ROUTES: EXPERIMENT RESULTS #{experiment_num}\n")
            f.write("=" * 50 + "\n\n")
            f.write("ALL ROUTES:\n")
            f.write("=" * 30 + "\n")
            for i, route in enumerate(all_routes):
                f.write(f"{i + 1:2d}. {route}\n")
            f.write("\n")
            f.write("ROUTE LENGTHS:\n")
            f.write("=" * 30 + "\n")
            for i, length in enumerate(route_lengths):
                f.write(f"{i + 1:2d}. Route length: {length}\n")

        mat_filename = os.path.join(results_dir, "weight_matrices.mat")
        sio.savemat(mat_filename, {'weight_matrices': np.array(all_weight_matrices)})
        lengths_mat_filename = os.path.join(results_dir, "route_lengths.mat")
        sio.savemat(lengths_mat_filename, {'route_lengths': np.array(route_lengths)})

        all_experiment_results.append({
            "experiment_num": experiment_num,
            "routes": all_routes,
            "route_lengths": route_lengths,
            "weight_matrices": all_weight_matrices,
        })

    print("\n" + "=" * 50)
    print(f"ALL {num_experiments} EXPERIMENTS COMPLETED!")

    logger.finish()
    return all_experiment_results