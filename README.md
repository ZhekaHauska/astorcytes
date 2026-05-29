# Astrocites

Spiking neural network navigation experiments with astrocyte modulation.

## Overview

This project implements a biologically-inspired spiking neural network (SNN) where **astrocyte cells** modulate neuron thresholds, influencing pathfinding behavior on a grid. The network learns navigation routes through STDP (Spike-Timing-Dependent Plasticity) with astrocyte-mediated modulation.

### Architecture

- **X (Input)**: Encodes the current position as a spike
- **Y (Output)**: Produces spike patterns used for action selection
- **I (Inhibitor)**: Competitive inhibition layer, optionally modulated by astrocytes
- **Connections**: X→Y (learned via STDP), X→I and I→Y (fixed inhibitory)

The astrocyte model increases firing thresholds of neurons that have been recently active via their neighbors, preventing stagnation and encouraging exploration of new paths.

## Installation

Requires [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Or with optional logging backends
uv sync --extra comet   # Comet.ml logging
uv sync --extra aim     # Aim logging
uv sync --extra dev     # Development tools (pytest, ruff)
```

## Running Experiments

### Using the default config

```bash
uv run astrocites
```

### Using a specific config file

```bash
uv run astrocites -c configs/large_grid.yaml
```

### Command-line overrides

```bash
uv run astrocites -c configs/default.yaml --num-experiments 3 --num-cycles 10
```

Override individual config values:

```bash
uv run astrocites -c configs/default.yaml -o grid_size=4 neuron.thresh=10
```

### Choosing a logger

```bash
# File-based logging (default)
uv run astrocites -l file

# Comet.ml
uv run astrocites -l comet

# Aim
uv run astrocites -l aim
```

## Configuration

All experiment parameters are specified in YAML files under `configs/`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `grid_size` | Grid dimension (N×N) | 3 |
| `start_position` | Starting cell index | 4 |
| `goal_position` | Goal cell index | 6 |
| `n_steps` | Max steps per navigation run | 101 |
| `neuron.thresh` | Firing threshold | 7 |
| `neuron.refrac` | Refractory period | 40 |
| `astrocyte.enable` | Whether astrocytes are active | true |
| `astrocyte.alpha` | Astrocyte decay rate | 0.001 |
| `astrocyte.k` | Astrocyte sensitivity | 0.2 |
| `experiment.num_experiments` | Number of independent experiments | 5 |
| `experiment.num_cycles` | Astrocyte/no-astrocyte cycles per experiment | 20 |

## Project Structure

```
astrocites/
├── __init__.py
├── nodes.py          # Neuron models (Input, LIFNodes)
├── learning.py       # STDP learning rules
├── connection.py      # Synaptic connections with impulse curves
├── network.py        # Network and monitor classes
├── utils.py          # Data loading and adjacency matrix
├── logging.py        # Logging backends (File, Comet, Aim)
├── experiment.py     # Experiment orchestration
└── run.py            # CLI entry point
configs/
├── default.yaml
├── large_grid.yaml
└── no_astrocyte_baseline.yaml
STDP.txt              # STDP weight-change lookup table
```

## Logging Backends

- **File**: Saves params, metrics (JSONL), and artifacts to a local directory (default)
- **Comet.ml**: Logs to [Comet](https://www.comet.ml/) for cloud-based experiment tracking
- **Aim**: Logs to [Aim](https://aimstack.io/) for local experiment comparison