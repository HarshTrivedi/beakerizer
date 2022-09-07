"""
A script to run beaker experiments.
"""
import json
import argparse
import subprocess
import os
from typing import List, Dict
import _jsonnet
import uuid
import base58
import dill
import io
import hashlib

from common import (
    prepare_beaker_image,
    safe_create_dataset,
    dataset_name_to_id,
    image_name_to_id,
)


def clean_white_space(text: str) -> str:
    return re.sub(r" +", " ", text).strip()


def hash_object(o: Any) -> str:
    # Taken from allennlp
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def make_beaker_experiment_description(experiment_name) -> str:    
    return f"Running {experiment_name}."


def make_beaker_experiment_name(experiment_name: str) -> str:
    command_str = experiment_name[:105]
    return f"{command_str}__{hash_object(experiment_name)[:10]}"


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
        "--dry_run",
        action="store_true",
        default=False,
        help="If specified, an experiment will not be created.",
    )
    parser.add_argument(
        "--allow_rollback",
        action="store_true",
        default=False,
        help="Allow rollback / use latest already present image.",
    )
    args = parser.parse_args()

    experiment_config_jsonnet_path = os.path.join(
        "experiment_configs", "configs", args.experiment_name + ".jsonnet"
    )
    if not os.path.exists(experiment_config_jsonnet_path):
        exit("Neither jsonnet or json experiment config found.")

    experiment_config = json.loads(
        _jsonnet.evaluate_file(experiment_config_jsonnet_path)
    )

    command = clean_white_space(experiment_config["command"])
    data_filepaths = experiment_config["data_filepaths"]
    dockerfile_path = experiment_config["dockerfile_path"]
    beaker_output_directory = experiment_config.get(
        "beaker_output_directory", "output"
    ).rstrip("/")
    docker_filepath = experiment_config.pop("docker_filepath", None)
    gpu_count = experiment_config.pop("gpu_count", False)
    cpu_count = experiment_config.pop("cpu_count", None)
    memory = experiment_config.pop("memory", None)

    cluster_map = {
        "v100": "ai2/harsh-v100",
        "onperm-aristo": "ai2/aristo-cirrascale",
        "onperm-ai2": "ai2/general-cirrascale",
        "onperm-mosaic": "ai2/mosaic-cirrascale",
        "cpu": "ai2/harsh-cpu32",
    }
    cluster = cluster_map[args.cluster]

    CONFIGS_FILEPATH = ".project-beaker-config.json"
    with open(CONFIGS_FILEPATH) as file:
        configs = json.load(file)

    working_dir = configs.pop("working_dir")
    beaker_workspace = configs.pop("beaker_workspace")

    dataset_mounts = []
    for data_filepath in data_filepaths:
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
        dockerfile=docker_filepath,
        allow_rollback=args.allow_rollback,
        beaker_image_prefix=hash_prefix,
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

    experiment_config = {
        "description": beaker_experiment_description,
        "tasks": [
            {
                "image": {"beaker": beaker_image_id},
                "result": {"path": results_path},
                "arguments": command.split(" "),
                "envVars": [
                    {"name": key, "value": value} for key, value in env.items()
                ],
                "resources": {"gpuCount": gpu_count},
                "context": {"cluster": cluster, "priority": "normal"},
                "datasets": dataset_mounts,
                "name": beaker_experiment_name,
            }
        ],
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
