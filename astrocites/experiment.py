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
from astrocites.logs import ExperimentLogger, FileLogger


def setup_and_run_simulation(
    NA, weights_mask_XY, weights_init_XY, weights_init_XI, n_steps,
    current_position, goal, learning_rate, wmin, wmax, weight_decay,
    post_spike_weight_decay, reset, refrac, thresh, intensity, time_steps, dt,
    enable_astrocyte, alpha, k, logger=None,
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
        weights_before = network.connections['X_Y'].w.detach().clone()
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
        delta = network.connections['X_Y'].w.detach().clone() - weights_before
        modulated_delta = delta.clone()
        for adj in adjacent_positions:
            if adj == new_position:
                coef = 1.5
            else:
                coef = 1.0
            modulated_delta[current_position, adj] = delta[current_position, adj] * coef
        network.connections['X_Y'].w.data = weights_before + modulated_delta
        network.connections['X_Y'].w.data *= torch.Tensor(weights_mask_XY).float()
        weights_2d = network.connections['X_Y'].w.detach().cpu().numpy()
        current_position = new_position
        network.reset_()
    elapsed = t() - start
    if logger is not None:
        logger.log_metrics({"elapsed_time": elapsed, "steps_taken": len(positions), "reached_goal": current_position == goal})
    return positions, weights_2d


def run_experiment(config: dict, logger: ExperimentLogger = None):
    if logger is None:
        logger = FileLogger(output_dir=config.get("output_dir", "results"))

    logger.log_params(config)

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
        print("INITIAL RUN WITHOUT ASTROCYTES")
        print(f"{'=' * 60}")

        positions, weights_2d = setup_and_run_simulation(
            NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
            weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
            goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
            weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
            reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
            time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
            logger=logger,
        )
        all_routes.append(f"Route without astrocytes (initial): {positions}")
        route_lengths.append(len(positions) - 1)

        logger.log_metrics({
            f"exp{experiment_num}.initial_no_astro_route_length": len(positions) - 1,
        }, step=0)

        for cycle_idx in range(num_cycles):
            cycle_num = cycle_idx + 1

            if enable_astrocyte:
                print(f"\n{'=' * 60}")
                print(f"RUN WITH ASTROCYTES {cycle_num} OUT OF {num_cycles}")
                print(f"{'=' * 60}")

                positions_astro, weights_2d_astro = setup_and_run_simulation(
                    NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                    weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                    goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                    weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                    reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                    time_steps=time_steps, dt=dt, enable_astrocyte=True, alpha=alpha, k=k,
                    logger=logger,
                )
                QAZ_after_astro = weights_2d_astro * weights_mask_XY
                all_weight_matrices.append(QAZ_after_astro.copy())
                weights_init_XY = torch.Tensor(weights_2d_astro).float()
                all_routes.append(f"Route with astrocytes (cycle {cycle_num}): {positions_astro}")
                route_lengths.append(len(positions_astro) - 1)

            print(f"\n{'=' * 60}")
            print(f"CYCLE {cycle_num}/{num_cycles}: VERIFICATION WITHOUT ASTROCYTES")
            print(f"{'=' * 60}")

            positions_no_astro, weights_2d_no_astro = setup_and_run_simulation(
                NA=NA, weights_mask_XY=weights_mask_XY, weights_init_XY=weights_init_XY,
                weights_init_XI=weights_init_XI, n_steps=n_steps, current_position=current_position,
                goal=goal, learning_rate=learning_rate, wmin=wmin, wmax=wmax,
                weight_decay=weight_decay, post_spike_weight_decay=post_spike_weight_decay,
                reset=reset, refrac=refrac, thresh=thresh, intensity=intensity,
                time_steps=time_steps, dt=dt, enable_astrocyte=False, alpha=alpha, k=k,
                logger=logger,
            )
            all_routes.append(f"Route without astrocytes (cycle {cycle_num}): {positions_no_astro}")
            route_lengths.append(len(positions_no_astro) - 1)

            metrics = {
                f"exp{experiment_num}.no_astro_route_length": len(positions_no_astro) - 1,
            }
            if enable_astrocyte:
                metrics[f"exp{experiment_num}.astro_route_length"] = len(positions_astro) - 1
            logger.log_metrics(metrics, step=cycle_num)

        results_dir = os.path.join(str(logger.exp_dir), f"experiment_{experiment_num}")
        os.makedirs(results_dir, exist_ok=True)

        weights_filename = os.path.join(results_dir, "weight_matrices.txt")
        with open(weights_filename, 'w') as f:
            f.write(f"WEIGHT MATRICES: EXPERIMENT RESULTS #{experiment_num}\n")
            f.write("=" * 50 + "\n\n")
            for i, weight_matrix in enumerate(all_weight_matrices):
                if i == 0:
                    f.write("INITIAL WEIGHT MATRIX:\n")
                else:
                    f.write(f"WEIGHT MATRIX AFTER CYCLE {i} WITH ASTROCYTES:\n")
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