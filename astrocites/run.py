import argparse
import yaml
import json

from astrocites.experiment import run_experiment
from astrocites.logs import get_logger


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            return yaml.safe_load(f)
        elif path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")


def merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description="Astrocyte-modulated spiking neural network navigation experiments")
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("-o", "--overrides", nargs="*", default=[],
                        help="Override config values as key=value pairs")
    parser.add_argument("-l", "--logger", type=str, default="file",
                        choices=["file", "comet", "aim"],
                        help="Logging backend")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--num-experiments", type=int, default=None,
                        help="Override number of experiments")
    parser.add_argument("--num-cycles", type=int, default=None,
                        help="Override number of astrocyte cycles per experiment")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.overrides:
        override_dict = {}
        for item in args.overrides:
            key, value = item.split("=", 1)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            keys = key.split(".")
            d = override_dict
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        config = merge_configs(config, override_dict)

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.num_experiments is not None:
        config["num_experiments"] = args.num_experiments
    if args.num_cycles is not None:
        config["num_cycles"] = args.num_cycles

    logger = get_logger(
        logger_type=args.logger,
        output_dir=config.get("output_dir", "results"),
        experiment_name=f"grid{config['grid_size']}_{config.get('experiment_name', 'run')}",
    )

    print("START OF EXPERIMENT SERIES")
    print("=" * 50)
    run_experiment(config, logger=logger)


if __name__ == "__main__":
    main()