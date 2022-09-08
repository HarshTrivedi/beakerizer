"""
A script to download results of beaker experiments.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__))))

from typing import List
import shutil
import argparse
import subprocess
import json
import _jsonnet
from glob import glob
from collections import defaultdict

from run import make_beaker_experiment_name


def get_experiments_results_dataset_ids(beaker_experiment_name: str) -> List[str]:
    experiment_details = subprocess.check_output(
        [
            "beaker",
            "experiment",
            "inspect",
            "--format",
            "json",
            "harsh-trivedi/" + beaker_experiment_name,
        ]
    ).strip()
    experiment_details = json.loads(experiment_details)
    results_dataset_ids = [
        task_obj["execution"]["result"]["beaker"] for task_obj in experiment_details[0]["jobs"]
    ]
    return results_dataset_ids


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="Experiment name.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If specified, the download commands will only be printed and not executed.",
    )
    args = parser.parse_args()

    experiment_config_path = os.path.join(
        "experiment_configs", args.experiment_name + ".jsonnet"
    )
    if not os.path.exists(experiment_config_path):
        exit("Experiment config found.")

    experiment_config = json.loads(
        _jsonnet.evaluate_file(experiment_config_path)
    )

    output_directory = experiment_config["local_output_directory"]

    beaker_experiment_name = make_beaker_experiment_name(args.experiment_name)
    results_dataset_ids = get_experiments_results_dataset_ids(beaker_experiment_name)

    if len(results_dataset_ids) > 1:
        if "$INDEX" not in output_directory:
            print(f"WARNING: Will download all results in the same output_directory: {output_directory}")
        output_directories = [
            output_directory.replace("$INDEX", str(index))
            for index in range(len(results_dataset_ids))
        ]
    else:
        assert "$INDEX" not in output_directory
        output_directories = [output_directory]

    assert len(output_directories) == len(results_dataset_ids)

    for index, (output_directory, dataset_id) in enumerate(
        zip(output_directories, results_dataset_ids)
    ):
        print(f"Downloading results {index+1}/{len(results_dataset_ids)} in {output_directory}")
        command = [
            "beaker", "dataset", "fetch", "--output", output_directory, dataset_id
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command)

if __name__ == "__main__":
    main()
