from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ==================== Manifest ====================

def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        manifest = yaml.safe_load(f)

    if "experiments" not in manifest:
        raise ValueError("Manifest must contain 'experiments' key")

    for i, exp in enumerate(manifest["experiments"]):
        for required in ("name", "command", "base_config"):
            if required not in exp:
                raise ValueError(
                    f"Experiment {i} missing required field '{required}'"
                )

    return manifest


# ==================== Grid expansion & override formatting ====================

def expand_grid(grid: dict | None) -> list[dict]:
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = []
    for k in keys:
        v = grid[k]
        values.append(v if isinstance(v, list) else [v])

    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        if value.lower() in ("true", "false", "none"):
            return value
        return value
    elif value is None:
        return "null"
    elif isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    else:
        return str(value)


def format_override(key: str, value) -> str:
    return f"{key}={format_value(value)}"


CLI_FLAGS = ("logger", "output_dir", "num_experiments", "num_cycles")

CLI_FLAG_ARGS = {
    "logger": lambda v: ["-l", str(v)],
    "output_dir": lambda v: ["--output-dir", str(v)],
    "num_experiments": lambda v: ["--num-experiments", str(v)],
    "num_cycles": lambda v: ["--num-cycles", str(v)],
}


def build_commands(experiment: dict) -> list[list[str]]:
    base_cmd = experiment["command"].split()
    base_config = experiment["base_config"]

    overrides = dict(experiment.get("overrides") or {})
    grid_combos = expand_grid(experiment.get("grid"))

    base_cli = {}
    for flag in CLI_FLAGS:
        if flag in experiment:
            base_cli[flag] = experiment[flag]
        if flag in overrides:
            base_cli[flag] = overrides.pop(flag)

    commands = []
    for combo in grid_combos:
        merged = {**overrides, **combo}
        combo_cli = {}
        for flag in CLI_FLAGS:
            if flag in merged:
                combo_cli[flag] = merged.pop(flag)

        effective_cli = {**base_cli, **combo_cli}

        parts = base_cmd + ["-c", base_config]
        if merged:
            parts.append("-o")
            for k, v in merged.items():
                parts.append(format_override(k, v))
        for flag, value in effective_cli.items():
            parts.extend(CLI_FLAG_ARGS[flag](value))
        commands.append(parts)

    return commands


# ==================== Checkpointing ====================

def checkpoint_path(manifest_path: str | Path) -> Path:
    return Path(str(manifest_path) + ".checkpoint.json")


def compute_manifest_hash(experiments: list[tuple[str, list[list[str]], int]]) -> str:
    h = hashlib.sha256()
    for name, commands, _ in experiments:
        h.update(name.encode())
        h.update(str(len(commands)).encode())
        for cmd in commands:
            for arg in cmd:
                h.update(arg.encode())
    return h.hexdigest()[:16]


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Could not load checkpoint {path}: {e}")
        print("Starting from scratch.")
        return None


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    tmp = path.with_suffix(".checkpoint.json.tmp")
    with tmp.open("w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.rename(path)


def init_checkpoint(
    experiments: list[tuple[str, list[list[str]], int]],
    manifest_hash: str,
) -> dict:
    runs = {}
    for name, commands, _ in experiments:
        for i, cmd in enumerate(commands):
            runs[f"{name}/{i}"] = {
                "command": cmd,
                "status": "pending",
            }
    return {"manifest_hash": manifest_hash, "runs": runs}


def merge_checkpoint(
    checkpoint: dict,
    experiments: list[tuple[str, list[list[str]], int]],
    manifest_hash: str,
) -> tuple[dict, list[str]]:
    old_runs = checkpoint.get("runs", {})
    warnings = []
    new_runs = {}

    for name, commands, _ in experiments:
        n_kept = 0
        n_changed = 0
        n_new = 0
        for i, cmd in enumerate(commands):
            key = f"{name}/{i}"
            if key in old_runs:
                if old_runs[key].get("command", []) == cmd:
                    new_runs[key] = old_runs[key]
                    n_kept += 1
                else:
                    new_runs[key] = {"command": cmd, "status": "pending"}
                    n_changed += 1
            else:
                new_runs[key] = {"command": cmd, "status": "pending"}
                n_new += 1

        if n_changed or n_new:
            parts = []
            if n_changed:
                parts.append(f"{n_changed} changed")
            if n_new:
                parts.append(f"{n_new} new")
            warnings.append(
                f"  {name}: {', '.join(parts)} run(s) reset to pending"
            )

    removed = len(set(old_runs.keys()) - set(new_runs.keys()))
    if removed:
        warnings.append(
            f"  {removed} run(s) removed from checkpoint "
            f"(no longer in manifest)"
        )

    return {"manifest_hash": manifest_hash, "runs": new_runs}, warnings


# ==================== Execution ====================

def run_experiment(
    name: str,
    commands: list[list[str]],
    max_parallel: int,
    checkpoint: dict,
    cp_path: Path,
    restart_failed: bool = False,
) -> None:
    total = len(commands)
    skip_ok = 0
    skip_fail = 0
    queue: list[int] = []

    for i in range(total):
        key = f"{name}/{i}"
        status = checkpoint["runs"][key].get("status", "pending")
        if status == "ok":
            skip_ok += 1
        elif status == "fail":
            if restart_failed:
                queue.append(i)
            else:
                skip_fail += 1
        else:
            queue.append(i)

    print(f"\n{'='*60}")
    print(f"Experiment: {name} | {total} runs | max_parallel={max_parallel}")
    if skip_ok or skip_fail:
        parts = []
        if skip_ok:
            parts.append(f"{skip_ok} ok")
        if skip_fail:
            parts.append(f"{skip_fail} failed")
        print(f"  Skipping: {', '.join(parts)}")
        if skip_fail and not restart_failed:
            print(
                f"  (use --restart-failed to retry "
                f"{skip_fail} failed run(s))"
            )
    print(f"  Running: {len(queue)}/{total}")
    print(f"{'='*60}")

    if not queue:
        print(f"  Experiment '{name}' — nothing to run.")
        return

    running: dict[int, tuple[subprocess.Popen, float, str, int]] = {}

    while queue or running:
        while queue and len(running) < max_parallel:
            idx = queue.pop(0)
            cmd = commands[idx]
            key = f"{name}/{idx}"
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
            checkpoint["runs"][key] = {
                "command": cmd,
                "status": "running",
                "start_time": start_iso,
            }
            save_checkpoint(checkpoint, cp_path)
            proc = subprocess.Popen(cmd)
            running[proc.pid] = (proc, time.time(), start_iso, idx)
            print(f"  [{idx+1}/{total}] Started (PID {proc.pid})")

        finished = []
        for pid, (proc, start, start_iso, idx) in running.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start
                end_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                key = f"{name}/{idx}"
                status = "ok" if ret == 0 else "fail"
                checkpoint["runs"][key] = {
                    "command": commands[idx],
                    "status": status,
                    "exit_code": ret,
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "duration_s": round(elapsed, 1),
                }
                save_checkpoint(checkpoint, cp_path)
                label = "OK" if ret == 0 else f"FAIL(code={ret})"
                print(
                    f"  [{idx+1}/{total}] {label} "
                    f"({elapsed:.1f}s) (PID {pid})"
                )
                finished.append(pid)

        for pid in finished:
            del running[pid]

        if running:
            time.sleep(0.5)

    print(f"  Experiment '{name}' complete.")


def run_interleaved(
    experiments: list[tuple[str, list[list[str]], int]],
    global_max_parallel: int,
    checkpoint: dict,
    cp_path: Path,
    restart_failed: bool = False,
) -> None:
    total_runs = sum(len(cmds) for _, cmds, _ in experiments)

    skip_ok_total = 0
    skip_fail_total = 0
    run_total = 0

    for name, cmds, _ in experiments:
        for i in range(len(cmds)):
            key = f"{name}/{i}"
            status = checkpoint["runs"][key].get("status", "pending")
            if status == "ok":
                skip_ok_total += 1
            elif status == "fail":
                if restart_failed:
                    run_total += 1
                else:
                    skip_fail_total += 1
            else:
                run_total += 1

    print(f"\n{'='*60}")
    print(
        f"Interleaved schedule | {total_runs} total runs | "
        f"global max_parallel={global_max_parallel}"
    )
    if skip_ok_total or skip_fail_total:
        parts = []
        if skip_ok_total:
            parts.append(f"{skip_ok_total} ok")
        if skip_fail_total:
            parts.append(f"{skip_fail_total} failed")
        print(f"  Skipping: {', '.join(parts)}")
        if skip_fail_total and not restart_failed:
            print(
                f"  (use --restart-failed to retry "
                f"{skip_fail_total} failed run(s))"
            )
    print(f"  Running: {run_total}/{total_runs}")
    print(f"{'='*60}")

    n_exp = len(experiments)
    queues: list[list[int]] = []
    exp_max: list[int] = []

    for name, cmds, mp in experiments:
        q = []
        for i in range(len(cmds)):
            key = f"{name}/{i}"
            status = checkpoint["runs"][key].get("status", "pending")
            if status == "ok":
                continue
            elif status == "fail" and not restart_failed:
                continue
            q.append(i)
        queues.append(q)
        exp_max.append(mp)

    per_exp_running: list[int] = [0] * n_exp
    per_exp_ok: list[int] = [0] * n_exp
    per_exp_fail: list[int] = [0] * n_exp

    running: dict[int, tuple[subprocess.Popen, float, str, int, int]] = {}

    def try_launch(rr_idx: int) -> bool:
        for offset in range(n_exp):
            i = (rr_idx + offset) % n_exp
            if not queues[i]:
                continue
            if per_exp_running[i] >= exp_max[i]:
                continue
            idx = queues[i].pop(0)
            name, cmds, _ = experiments[i]
            cmd = cmds[idx]
            key = f"{name}/{idx}"
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
            checkpoint["runs"][key] = {
                "command": cmd,
                "status": "running",
                "start_time": start_iso,
            }
            save_checkpoint(checkpoint, cp_path)
            proc = subprocess.Popen(cmd)
            running[proc.pid] = (proc, time.time(), start_iso, i, idx)
            per_exp_running[i] += 1
            print(
                f"  [{name} {idx+1}/{len(cmds)}] "
                f"Started (PID {proc.pid})"
            )
            return True
        return False

    if run_total == 0:
        print("  Nothing to run.")
        print_summary(experiments, checkpoint)
        return

    rr = 0
    while any(q for q in queues) or running:
        while sum(per_exp_running) < global_max_parallel:
            if not try_launch(rr):
                break
            rr = (rr + 1) % n_exp

        finished = []
        for pid, (proc, start, start_iso, exp_i, idx) in running.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start
                end_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                name, cmds, _ = experiments[exp_i]
                key = f"{name}/{idx}"
                per_exp_running[exp_i] -= 1
                status = "ok" if ret == 0 else "fail"
                if ret == 0:
                    per_exp_ok[exp_i] += 1
                else:
                    per_exp_fail[exp_i] += 1
                checkpoint["runs"][key] = {
                    "command": cmds[idx],
                    "status": status,
                    "exit_code": ret,
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "duration_s": round(elapsed, 1),
                }
                save_checkpoint(checkpoint, cp_path)
                label = "OK" if ret == 0 else f"FAIL(code={ret})"
                print(
                    f"  [{name} {idx+1}/{len(cmds)}] {label} "
                    f"({elapsed:.1f}s) (PID {pid})"
                )
                finished.append(pid)

        for pid in finished:
            del running[pid]

        if running:
            time.sleep(0.5)

    print_summary(experiments, checkpoint)


def print_summary(
    experiments: list[tuple[str, list[list[str]], int]],
    checkpoint: dict,
) -> None:
    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"{'='*60}")
    for name, cmds, _ in experiments:
        total = len(cmds)
        ok = fail = pending = running = 0
        for i in range(total):
            key = f"{name}/{i}"
            status = checkpoint["runs"][key].get("status", "pending")
            if status == "ok":
                ok += 1
            elif status == "fail":
                fail += 1
            elif status == "running":
                running += 1
            else:
                pending += 1
        parts = [f"{ok}/{total} OK"]
        if fail:
            parts.append(f"{fail} FAIL")
        if pending:
            parts.append(f"{pending} pending")
        if running:
            parts.append(f"{running} interrupted")
        print(f"  {name}: {', '.join(parts)}")


# ==================== Main ====================

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m astrocites.run_scheduler "
            "<manifest.yaml> [--dry-run] [--restart-failed] [--force]"
        )
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv
    restart_failed = "--restart-failed" in sys.argv
    force = "--force" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not positional:
        print("Error: manifest path is required")
        sys.exit(1)
    manifest_path = positional[0]

    manifest = load_manifest(manifest_path)
    global_max_parallel = manifest.get("max_parallel", 1)
    schedule = manifest.get("schedule", "sequential")

    if schedule not in ("sequential", "interleaved"):
        raise ValueError(
            f"Unknown schedule mode '{schedule}'. "
            f"Must be 'sequential' or 'interleaved'."
        )

    experiments: list[tuple[str, list[list[str]], int]] = []
    for exp in manifest["experiments"]:
        name = exp["name"]
        commands = build_commands(exp)
        max_parallel = exp.get("max_parallel", global_max_parallel)
        experiments.append((name, commands, max_parallel))

    cp_path = checkpoint_path(manifest_path)
    current_hash = compute_manifest_hash(experiments)
    checkpoint = load_checkpoint(cp_path)

    if checkpoint is None:
        checkpoint = init_checkpoint(experiments, current_hash)
        save_checkpoint(checkpoint, cp_path)
    elif checkpoint.get("manifest_hash", "") != current_hash:
        if not force:
            expected_keys = set()
            for name, commands, _ in experiments:
                for i in range(len(commands)):
                    expected_keys.add(f"{name}/{i}")
            cp_keys = set(checkpoint.get("runs", {}).keys())
            new_count = len(expected_keys - cp_keys)
            removed_count = len(cp_keys - expected_keys)

            print(
                "ERROR: Manifest has changed since last run "
                "(checkpoint hash mismatch).\n"
                f"  New runs: {new_count}, "
                f"Removed runs: {removed_count}\n\n"
                "Options:\n"
                f"  1. Delete the checkpoint and re-run:\n"
                f"     rm {cp_path}\n"
                "  2. Use --force to merge (keep existing states, "
                "new runs as pending)"
            )
            sys.exit(1)
        else:
            checkpoint, warnings = merge_checkpoint(
                checkpoint, experiments, current_hash
            )
            save_checkpoint(checkpoint, cp_path)
            print(
                "WARNING: Manifest changed, merging checkpoint "
                "(--force):"
            )
            for w in warnings:
                print(w)
            print()
    else:
        for name, commands, _ in experiments:
            for i, cmd in enumerate(commands):
                key = f"{name}/{i}"
                checkpoint["runs"][key]["command"] = cmd
        save_checkpoint(checkpoint, cp_path)

    if dry_run:
        total_runs = sum(len(cmds) for _, cmds, _ in experiments)
        print(
            f"DRY RUN — {total_runs} total run(s) across "
            f"{len(experiments)} experiment(s)"
        )
        print(f"Schedule: {schedule}")

        n_ok = n_fail = n_running = n_pending = 0
        for v in checkpoint["runs"].values():
            s = v.get("status", "pending")
            if s == "ok":
                n_ok += 1
            elif s == "fail":
                n_fail += 1
            elif s == "running":
                n_running += 1
            else:
                n_pending += 1

        if n_ok or n_fail or n_running:
            parts = []
            if n_ok:
                parts.append(f"{n_ok} ok")
            if n_fail:
                parts.append(f"{n_fail} failed")
            if n_running:
                parts.append(f"{n_running} interrupted")
            if n_pending:
                parts.append(f"{n_pending} pending")
            print(f"Checkpoint: {', '.join(parts)}")

        print()
        for name, commands, max_parallel in experiments:
            print(
                f"--- {name} ({len(commands)} runs, "
                f"max_parallel={max_parallel}) ---"
            )
            for i, cmd in enumerate(commands):
                key = f"{name}/{i}"
                entry = checkpoint["runs"].get(key, {})
                status = entry.get("status", "pending")
                cmd_str = " ".join(cmd)
                if status == "ok":
                    dur = entry.get("duration_s")
                    dur_str = f" ({dur}s)" if dur is not None else ""
                    print(f"  [{i+1}] OK{dur_str}  {cmd_str}")
                elif status == "fail":
                    code = entry.get("exit_code", "?")
                    dur = entry.get("duration_s")
                    dur_str = f" ({dur}s)" if dur is not None else ""
                    if restart_failed:
                        print(
                            f"  [{i+1}] RETRY(code={code}){dur_str}  "
                            f"{cmd_str}"
                        )
                    else:
                        print(
                            f"  [{i+1}] SKIP(code={code}){dur_str}  "
                            f"{cmd_str}"
                        )
                elif status == "running":
                    print(
                        f"  [{i+1}] INTERRUPTED  {cmd_str}"
                    )
                else:
                    print(f"  [{i+1}] {cmd_str}")
            print()
        return

    if schedule == "sequential":
        for name, commands, max_parallel in experiments:
            run_experiment(
                name, commands, max_parallel,
                checkpoint, cp_path, restart_failed
            )
        print_summary(experiments, checkpoint)
    elif schedule == "interleaved":
        run_interleaved(
            experiments, global_max_parallel,
            checkpoint, cp_path, restart_failed
        )


if __name__ == "__main__":
    main()