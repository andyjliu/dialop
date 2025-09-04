import os
import json
from datetime import datetime
from typing import Dict, List, Any
from argparse import ArgumentParser
import wandb
DUMMY_VALUE = -999
EXPECTED_KEYS = [
    "reward", "hh_turns", "hh_words", "hh_score", "hh_score_norm",
    "t", "num_turns", "num_words",
    "info.num_msgs", "info.score", "info.score_norm"
]

def parse_metrics_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a single metrics file, extract JSON at the end, and fill missing keys with DUMMY_VALUE.
    Handles nested keys for 'info'.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Find the last non-empty line that is valid JSON
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                metrics = json.loads(line[1:-1])
                break
            except json.JSONDecodeError:
                continue
    else:
        metrics = {}
    flat_metrics = {}
    for key in EXPECTED_KEYS:
        if key.startswith("info."):
            info_key = key.split(".", 1)[1]
            flat_metrics[key] = metrics.get("info", {}).get(info_key, DUMMY_VALUE)
        else:
            flat_metrics[key] = metrics.get(key, DUMMY_VALUE)
    return flat_metrics

def aggregate_metrics(exp_dir: str, exp_name: str) -> Dict[str, float]:
    """
    Parse all 'n_*.out' files in EXP_DIR/exp_name, extract metrics, compute averages.
    Returns a dict of metric_name -> average_value, and logs count of complete trajectories.
    """
    dir_path = os.path.join(exp_dir, exp_name)
    metric_sums = {key: 0.0 for key in EXPECTED_KEYS}
    metric_counts = {key: 0 for key in EXPECTED_KEYS}
    complete_trajectories = 0
    total_files = 0
    for filename in os.listdir(dir_path):
        if filename.endswith('.out'):
            filepath = os.path.join(dir_path, filename)
            metrics = parse_metrics_file(filepath)
            total_files += 1
            if metrics["info.score_norm"] != DUMMY_VALUE:
                complete_trajectories += 1
            for key in EXPECTED_KEYS:
                if metrics[key] != DUMMY_VALUE:
                    metric_sums[key] += float(metrics[key])
                    metric_counts[key] += 1
    avg_metrics = {}
    for key in EXPECTED_KEYS:
        if metric_counts[key] > 0:
            avg_metrics[key] = metric_sums[key] / metric_counts[key]
        else:
            avg_metrics[key] = DUMMY_VALUE
    avg_metrics["complete_trajectory_rate"] = complete_trajectories/total_files if total_files > 0 else 0.0
    avg_metrics["num_total_files"] = total_files
    return avg_metrics

def make_exp_name(exp_dir: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(os.path.normpath(exp_dir))
    return f"{base}_{date_str}"

def write_to_wandb(metrics: Dict[str, float], args: dict, wandb_project: str, exp_name: str = None):
    args = {'args/'+k : v for k, v in args.items()}
    wandb.init(project=wandb_project, config=args, name=exp_name)
    wandb.log(metrics)
    wandb.finish()