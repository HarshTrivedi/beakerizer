"""
A script to run beaker experiments.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__))))

import re
import json
import argparse
import subprocess
from typing import List, Dict, Any
import _jsonnet
import uuid
import base58
import dill
import io
import hashlib

from utils import (
    hash_object,
    prepare_beaker_image,
    safe_create_dataset,
    dataset_name_to_id,
    image_name_to_id,
    make_beaker_experiment_name,
    make_beaker_experiment_description,
    get_experiments_results_dataset_ids,
)


def clean_white_space(text: str) -> str:
    return re.sub(r" +", " ", text).strip()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="Experiment name.")
    parser.add_argument(
        "--cluster",
        type=str,
        choices={"v100", "onperm-aristo", "onperm-ai2", "onperm-mosaic", "cpu"},
        default="v100",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If specified, an experiment will not be created.",
    )
    parser.add_argument(
        "--allow-rollback",
        action="store_true",
        default=False,
        help="Allow rollback / use latest already present image.",
    )
    args = parser.parse_args()

    experiment_config_path = os.path.join(
        "beaker_configs", args.experiment_name + ".jsonnet"
    )
    if not os.path.exists(experiment_config_path):
        exit("Experiment config found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_path))

    command = clean_white_space(experiment_config.pop("command"))
    data_filepaths = experiment_config.pop("data_filepaths")
    local_output_directory = experiment_config.pop(
        "local_output_directory"
    )  # not used here.
    beaker_output_directory = experiment_config.pop("beaker_output_directory")
    docker_filepath = experiment_config.pop("docker_filepath")
    gpu_count = experiment_config.pop("gpu_count", None)
    cpu_count = experiment_config.pop("cpu_count", None)
    memory = experiment_config.pop("memory", None)
    parallel_run_count = experiment_config.pop("parallel_run_count", None)
    cluster = experiment_config.pop("cluster", args.cluster)

    if experiment_config:
        exit(f"Unused experiment_config: {experiment_config}")

    cluster_map = {
        "v100": "ai2/harsh-v100",
        "onperm-aristo": "ai2/aristo-cirrascale",
        "onperm-ai2": "ai2/general-cirrascale",
        "onperm-mosaic": "ai2/mosaic-cirrascale",
        "cpu": "ai2/cpu-p10c16g100n",
    }
    cluster = cluster_map[cluster]

    CONFIGS_FILEPATH = ".project-beaker-config.json"
    with open(CONFIGS_FILEPATH) as file:
        configs = json.load(file)

    working_dir = configs.pop("working_dir")
    beaker_workspace = configs.pop("beaker_workspace")

    dataset_mounts = []
    for data_filepath in data_filepaths:

        if data_filepath.startswith("result_of_"):
            # Mount result dataset of a beaker experiment.
            source_experiment_name = data_filepath.replace("result_of_", "")
            source_experiment_config_path = os.path.join(
                "beaker_configs", source_experiment_name + ".jsonnet"
            )
            if not os.path.exists(source_experiment_config_path):
                exit("Source experiment config not found.")

            source_experiment_config = json.loads(
                _jsonnet.evaluate_file(experiment_config_path)
            )
            source_local_output_directory = source_experiment_config[
                "local_output_directory"
            ]
            source_beaker_experiment_name = make_beaker_experiment_name(
                source_experiment_name
            )
            source_result_ids = get_experiments_results_dataset_ids(
                source_beaker_experiment_name
            )

            source_local_output_directories = [
                source_local_output_directory.replace("$INDEX", str(index))
                for index in range(len(source_result_ids))
            ]

            for local_output_directory, result_id in zip(
                source_local_output_directories, source_result_ids
            ):
                dataset_mounts.append(
                    {
                        "source": {"beaker": result_id},
                        # "subPath": data_file_name,
                        "mountPath": f"{working_dir}/{local_output_directory}",
                    }
                )

        else:
            # Mount local dataset filepath.
            dataset_name = safe_create_dataset(data_filepath)
            data_file_name = os.path.basename(data_filepath)
            dataset_id = dataset_name_to_id(dataset_name)
            dataset_mounts.append(
                {
                    "source": {"beaker": dataset_id},
                    "subPath": data_file_name,
                    "mountPath": f"{working_dir}/{data_filepath}",
                }
            )

    # Prepare Dockerfile
    beaker_image = prepare_beaker_image(
        docker_filepath=docker_filepath, allow_rollback=args.allow_rollback
    )

    beaker_image_id = image_name_to_id(beaker_image)
    results_path = os.path.join(working_dir, beaker_output_directory)

    # Prepare Experiment Config
    beaker_experiment_name = make_beaker_experiment_name(args.experiment_name)
    beaker_experiment_description = make_beaker_experiment_description(
        args.experiment_name
    )

    wandb_run_name = uuid.uuid4().hex
    env = {"WANDB_RUN_NAME": wandb_run_name}

    task_configs = []
    for run_index in range(parallel_run_count):

        beaker_task_name = beaker_experiment_name
        if parallel_run_count:
            beaker_task_name += f"__task_{run_index+1}"

        task_configs.append(
            {
                "image": {"beaker": beaker_image_id},
                "result": {"path": results_path.replace("$INDEX", str(run_index))},
                "arguments": command.replace("$INDEX", str(run_index)).split(" "),
                "envVars": [
                    {"name": key, "value": value} for key, value in env.items()
                ],
                "resources": {"gpuCount": gpu_count},
                "context": {"cluster": cluster, "priority": "normal"},
                "datasets": dataset_mounts,
                "name": beaker_task_name,
            }
        )

    assert len(set([hash_object(task_config) for task_config in task_configs])) == len(
        task_configs
    ), "Looks like some of the task configs are identical. Make sure to use $INDEX correctly."

    experiment_config = {
        "description": beaker_experiment_description,
        "tasks": task_configs,
    }

    # Save full config file.
    experiment_hash_id = hash_object(experiment_config)[:10]
    beaker_experiment_config_path = (
        f".beaker_experiment_specs/{experiment_hash_id}.json"
    )
    with open(beaker_experiment_config_path, "w") as output:
        output.write(json.dumps(experiment_config, indent=4))
    print(f"Beaker spec written to {beaker_experiment_config_path}.")

    # Build beaker command to run.
    experiment_run_command = [
        "beaker",
        "experiment",
        "create",
        beaker_experiment_config_path,
        "--name",
        beaker_experiment_name,
        "--workspace",
        beaker_workspace,
    ]
    print(f"\nRun the experiment with:")
    print(f"    " + " ".join(experiment_run_command))

    # Run beaker command if required.
    if not args.dry_run:
        subprocess.run(experiment_run_command)


if __name__ == "__main__":
    main()
