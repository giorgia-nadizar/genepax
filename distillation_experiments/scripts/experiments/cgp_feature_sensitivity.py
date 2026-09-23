"""Run the fixed-budget inverted-double-pendulum feature-count sweep on CUDA.

Runs each setting sequentially to avoid GPU contention. Existing k=8 results
are reused; timing was not recorded for those historical runs.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    experiments = Path(__file__).resolve().parents[2]
    root = experiments.parent
    base = experiments / "artifacts" / "repertoires" / "cgp_feature_inverted_double_pendulum"
    sweep = base / "k_sensitivity_5seeds"
    sweep.mkdir(exist_ok=True)
    settings = []
    for k in (1, 2, 4, 8, 16, 32):
        for teacher in ("cgp", "ann"):
            run_name = (
                f"{teacher}_teacher_bc_k8_5seeds_gpu" if k == 8
                else f"{teacher}_teacher_bc_k{k}_sensitivity_5seeds_gpu"
            )
            dataset = experiments / "artifacts" / "expert_datasets" / (
                "cgp_teacher_inverted_double_pendulum.npz" if teacher == "cgp"
                else "expert_inverted_double_pendulum.npz"
            )
            settings.append({
                "teacher": teacher, "k": k, "run_name": run_name,
                "dataset": str(dataset), "reused": k == 8,
            })
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "feature_counts": [1, 2, 4, 8, 16, 32], "seeds": list(range(5)),
        "n_nodes": 50, "generations": 100, "population_size": 100,
        "expert_samples": 10000, "validation_fraction": 0.2,
        "rollout_steps": 1000, "evaluation_trajectories": 10,
        "target_reward": 9350, "strict_target_reward": 9359,
        "settings": settings,
        "timing_note": "Per-seed fit/evaluation wall time includes JAX compilation; historical k=8 timing unavailable.",
    }
    manifest_path = sweep / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert old["settings"] == settings, "Existing sweep settings differ"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    env = os.environ | {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    statuses = []
    for setting in settings:
        run_dir = base / setting["run_name"]
        summary_path = run_dir / "aggregate_summary.json"
        if summary_path.exists():
            records = json.loads(summary_path.read_text())["variants"]
            assert sorted(r["seed"] for r in records) == list(range(5)), (
                f"Incomplete run at {run_dir}; inspect before resuming"
            )
            assert all(r["k"] == setting["k"] for r in records)
            status = setting | {"status": "existing", "completed_seeds": 5}
        else:
            if setting["reused"]:
                raise FileNotFoundError(f"Missing historical baseline: {run_dir}")
            if run_dir.exists():
                raise RuntimeError(f"Incomplete run at {run_dir}; inspect before resuming")
            command = [
                sys.executable, "-u", "-m",
                "distillation_experiments.scripts.experiments.cgp_feature_imitation",
                "--env", "inverted_double_pendulum", "--dataset-path", setting["dataset"],
                "--k", str(setting["k"]), "--num-seeds", "5", "--n-nodes", "50",
                "--expert-samples", "10000", "--generations", "100", "--population-size", "100",
                "--run-name", setting["run_name"],
            ]
            print(f"Starting {setting['teacher']} teacher, k={setting['k']}", flush=True)
            started = time.perf_counter()
            with (sweep / f"{setting['teacher']}_k{setting['k']}.log").open("w") as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            status = setting | {
                "status": "completed", "completed_seeds": 5,
                "process_seconds": time.perf_counter() - started,
            }
        statuses.append(status)
        (sweep / "status.json").write_text(json.dumps(statuses, indent=2))
        print(f"Finished {setting['teacher']} teacher, k={setting['k']}", flush=True)


if __name__ == "__main__":
    main()
