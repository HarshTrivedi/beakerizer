from typing import List, Any
import subprocess
import os
import random
from dateutil import parser
from datetime import datetime
from collections import defaultdict
import json
import re
import base58
import dill
import io
import hashlib
import math
from tqdm import tqdm


CONFIGS_FILEPATH = ".project-beaker-config.json"
with open(CONFIGS_FILEPATH) as file:
    configs = json.load(file)

user_name = configs.pop("user_name")
project_name = configs.pop("project_name")
working_dir = configs.pop("working_dir")
image_critical_paths = configs.pop("image_critical_paths")
beaker_workspace = configs.pop("beaker_workspace")

text2hash = lambda text: str(
    int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8
)
# text2hash helps distinguish between experiments with same title. Beaker experiment names need to be unique.

safe_char_limit = 100


def hash_object(o: Any) -> str:
    # Taken from allennlp
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def get_true_randint(start: int, end: int):
    old_state = random.getstate()
    import time

    random.seed(int(time.time()))
    true_randint = random.randint(start, end)
    random.setstate(old_state)
    return true_randint


def remove_dataset(dataset: str):
    trash_name = (
        dataset[:safe_char_limit] + "__" + text2hash(dataset)
        if len(dataset) > safe_char_limit
        else dataset
    )
    updated_dataset_name = f"trash-{trash_name}-{get_true_randint(0, 1000)}"
    command = f"beaker dataset rename {user_name}/{dataset} {updated_dataset_name}"
    returncode = subprocess.run(command, shell=True).returncode
    success = returncode == 0
    if not success:

        success = (
            subprocess.run(
                f"beaker dataset inspect {user_name}/{updated_dataset_name}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if success:
            print(
                "Ingore above error, beaker renaming succeeded but yet showed misleading error."
            )

    return success


def create_dataset(dataset_name: str, dataset_path: str):
    command = f"beaker dataset create --workspace {beaker_workspace} --name  {dataset_name} {dataset_path}"
    returncode = subprocess.run(command, shell=True).returncode
    return returncode == 0


def dataset_file_path_to_name(file_path: str, skip_limit: bool = False):
    file_path = file_path.replace("/", "__")
    if not skip_limit and len(file_path) > safe_char_limit:
        file_path = file_path[:safe_char_limit] + "__" + text2hash(file_path)
    return file_path.replace("/", "__")


def dataset_name_to_id(dataset_name):
    output = json.loads(
        subprocess.check_output(
            f"beaker dataset inspect --format json {user_name}/{dataset_name}",
            shell=True,
        )
    )
    return output[0]["id"]


def image_name_to_id(image_name):
    output = json.loads(
        subprocess.check_output(
            f"beaker image inspect --format json {user_name}/{image_name}", shell=True
        )
    )
    return output[0]["id"]


def safe_create_dataset(dataset_path: str):
    if not os.path.exists(dataset_path):
        exit(f"{dataset_path} doesn't exist.")
    dataset_name = dataset_file_path_to_name(dataset_path)
    unavailable = subprocess.run(
        f"beaker dataset inspect {user_name}/{dataset_name}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    if unavailable:
        created = create_dataset(dataset_name, dataset_path)
        if not created:
            raise Exception(f"The dataset {dataset_name} at {dataset_path} couldn't be created.")
    else:

        def get_file_or_dir_mtime_ts(path: str):
            if not os.path.isdir(path):
                return os.path.getmtime(path)
            sub_paths = [root for root, _, _ in os.walk(path) if root != path]
            if sub_paths:
                return max(
                    [get_file_or_dir_mtime_ts(sub_path) for sub_path in sub_paths]
                )
            else:
                return os.path.getmtime(path)

        local_dataset_time = datetime.utcfromtimestamp(
            get_file_or_dir_mtime_ts(dataset_path)
        )

        output = json.loads(
            subprocess.check_output(
                f"beaker dataset inspect --format json {user_name}/{dataset_name}",
                shell=True,
            )
        )
        beaker_dataset_time = parser.parse(output[0]["committed"])

        if local_dataset_time > beaker_dataset_time.replace(tzinfo=None):
            # Need to delete (rename) beaker_dataset and create a new one.
            remove_dataset(dataset_name)
            create_dataset(dataset_name, dataset_path)

    return dataset_name


def is_beaker_dataset_id_available(dataset_id: str) -> bool:
    return (
        bool(
            subprocess.run(
                f"beaker dataset inspect {dataset_id}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
        )
        == 0
    )


def is_beaker_image_available(beaker_image: str):
    unavailable = bool(
        subprocess.run(
            f"beaker image inspect {user_name}/{beaker_image}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
    )
    return not unavailable


def remove_beaker_image(image: str):
    command = f"beaker image rename {image} trash-{image}-{get_true_randint(0, 100)}"
    returncode = subprocess.run(command, shell=True).returncode
    return returncode == 0


def create_beaker_image(beaker_image, docker_image):
    command = f"beaker image create --workspace={beaker_workspace} --name={beaker_image} {docker_image}"
    returncode = subprocess.run(command, shell=True).returncode
    return returncode == 0


def is_docker_image_available(docker_image: str):
    assert "/" in docker_image  # It should start with {user_name}/
    unavailable = bool(
        subprocess.run(
            f"docker image inspect {docker_image}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
    )
    return not unavailable


def remove_docker_image(docker_image):
    returncode = subprocess.run(
        f"docker image rm {docker_image}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return returncode == 0


def create_docker_image(docker_image, dockerfile):
    command = f"docker build -f {dockerfile} -t {docker_image} ."
    returncode = subprocess.run(command, shell=True).returncode
    return returncode == 0


def git_hash_to_beaker_image(git_hash: str, beaker_image_prefix: str = ""):
    if git_hash is None:
        return None
    if beaker_image_prefix:
        return beaker_image_prefix + "__" + git_hash
    return git_hash


def git_hash_to_docker_image(git_hash: str):
    return f"harshtrivedi/{project_name}:{git_hash_to_beaker_image(git_hash)}"


def beaker_to_docker_image(beaker_image: str):
    return f"harshtrivedi/{project_name}:{beaker_image}"


def _get_last_git_hash_with_beaker_image(
    num_history_to_try=5, beaker_image_prefix: str = ""
):
    command = "git log | grep commit | cut -c8-14 | head -n 5"
    output, _ = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, universal_newlines=True
    ).communicate()
    git_hashes = map(str, output.strip().split("\n"))
    for history, git_hash in enumerate(git_hashes):
        if is_beaker_image_available(
            git_hash_to_beaker_image(git_hash, beaker_image_prefix)
        ):
            return git_hash, history
    return [None, -1]


def prepare_beaker_image(
    docker_filepath: str,
    allow_rollback: bool = False,
    use_git_hash: str = None,
):

    if not os.path.exists(docker_filepath):
        exit(f"Dockerfile {docker_filepath} not found.")

    with open(docker_filepath, "r") as file:
        dockerfile_content = file.read().strip()

    beaker_image_prefix = text2hash(dockerfile_content)

    beaker_image = None

    if use_git_hash:
        print(f"Trying to use git hash: {use_git_hash}.")
        if not is_beaker_image_available(
            git_hash_to_beaker_image(use_git_hash, beaker_image_prefix)
        ):
            exit(
                f"You passed git-hash {use_git_hash}, but corresponding beaker image is not available."
            )
        beaker_image = git_hash_to_beaker_image(use_git_hash, beaker_image_prefix)

    elif allow_rollback:
        print(f"Trying to find git hash with a beaker image.")
        last_git_hash_with_beaker_image, _ = _get_last_git_hash_with_beaker_image(
            num_history_to_try=5, beaker_image_prefix=beaker_image_prefix
        )
        beaker_image = git_hash_to_beaker_image(
            last_git_hash_with_beaker_image, beaker_image_prefix
        )

    if not beaker_image:
        print(f"Trying to create beaker image from current git hash.")

        # Make sure there's no critical change since last commit.
        command = "git status | grep modified"
        output, _ = subprocess.Popen(
            command, stdout=subprocess.PIPE, shell=True, universal_newlines=True
        ).communicate()
        changed_files = [
            line.replace("modified:", "").strip() for line in output.strip().split("\n")
        ]

        cant_use_current_githash = any(
            [
                image_critical_path in changed_file
                for changed_file in changed_files
                for image_critical_path in image_critical_paths
            ]
        )
        if cant_use_current_githash:
            # There is critical change, warn user and terminate.
            exit(
                "Some file/s code critical files have changed since last commit. "
                "Please commit them before making beaker_image"
            )

        current_git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], universal_newlines=True
        ).strip()
        beaker_image = git_hash_to_beaker_image(current_git_hash, beaker_image_prefix)
        docker_image = beaker_to_docker_image(beaker_image)

        print(f"\nDocker Image Name: {docker_image}")
        print(f"Beaker Image Name: {beaker_image}\n")

        if not is_docker_image_available(docker_image):
            # Create Docker Image
            print(
                f"Docker image for latest git-hash {current_git_hash} doesn't exist. Building one."
            )
            created = create_docker_image(docker_image, docker_filepath)
            if not created:
                exit("Error in creating docker image.")
        created = create_beaker_image(beaker_image, docker_image)
        if not created:
            exit("Error in creating beaker image.")

    return beaker_image


def beaker_name_for_experiment(command: str, experiment_name: str, dataset_filepath: str = ""):
    assert command in ("train", "evaluate", "predict")
    dataset_filepath = dataset_filepath.strip()
    assert (command in ("evaluate", "predict")) == bool(dataset_filepath), \
        "The beaker name can be obtained for train with dataset_filepath and for evaluate/prediction without."
    full_identifier = text2hash(experiment_name + dataset_filepath)
    if "__" not in experiment_name:
        print(f"Warning: No title found in the experiment_name, {experiment_name}.")
    title = experiment_name.split("__")[0]
    if len(title) > 100:
        print("Warning: Experiment name can't more than 115 characters.")
    return f"{command}__{title[:100]}__{full_identifier}"


def get_experiments_result_dataset_id(beaker_experiment_name):
    experiment_details = subprocess.check_output(
        [
            "beaker",
            "experiment",
            "inspect",
            "--format",
            "json",
            f"{user_name}/" + beaker_experiment_name,
        ]
    ).strip()
    experiment_details = json.loads(experiment_details)
    # make sure that if experiment has multiple tasks then last task is the one of interest.
    return experiment_details[0]["jobs"][-1]["execution"]["result"]["beaker"]


def fetch_beaker_dataset_to(dataset_id, target_path, prefix=None):
    beaker_pull_command = f"beaker dataset fetch --output {target_path} {dataset_id}"
    if prefix:
        beaker_pull_command = beaker_pull_command + f" --prefix {prefix}"
    print(beaker_pull_command)
    subprocess.run(beaker_pull_command, shell=True)
    print(f"Pulled at: {target_path}")


def is_experiment_available(experiment_name):
    returncode = subprocess.run(
        f"beaker experiment inspect {user_name}/{experiment_name}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return returncode == 0


def get_experiments_results_dataset_ids(
        beaker_experiment_name: str,
        task_name_regex: str = None
    ) -> List[str]:
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
    name_to_result_dataset_ids = defaultdict(list)

    relevant_jobs = experiment_details[0]["jobs"]
    if task_name_regex is not None:
        relevant_jobs = [
            job for job in relevant_jobs
            if bool(re.compile(task_name_regex).match(job["name"]))
        ]

    name_to_jobs = defaultdict(list)
    for job in relevant_jobs:
        job_name = job["name"]
        if "exitCode" not in job["status"]:
            # This means that the result did not get committed
            assert "failed" in job["status"]
            continue
        name_to_jobs[job_name].append(job)

    relevant_jobs = [
        sorted(_jobs, key=lambda e: parser.parse(e["status"]["finalized"]))[-1]
        for _, _jobs in name_to_jobs.items()
    ]

    for job in relevant_jobs:
        # mount failed experiment as well, it's okay as long as it's committed.
        # if job["status"]["exitCode"] != 0:
        #     continue
        name = job["name"]
        result_dataset_id = job["execution"]["result"]["beaker"]
        name_to_result_dataset_ids[name].append(result_dataset_id)

    for name, result_dataset_ids in name_to_result_dataset_ids.items():
        assert len(result_dataset_ids) == 1, \
            "Found more than one successful result datasets for a job. " \
            "Add code here to pick the last committed one."

    sorted_names = sorted(list(name_to_result_dataset_ids.keys()))
    results_dataset_ids = [
        name_to_result_dataset_ids[name][0] for name in sorted_names
    ]

    num_jobs = len(set(job["name"] for job in relevant_jobs))
    num_results = len(results_dataset_ids)

    if num_results != num_jobs:
        print("WARNING: Not all jobs have finished yet. Skipping the failed ones.")

    if task_name_regex is not None and num_results > 1:
        print(
            "WARINING: Task regex passed, but it matches multiple dataset-committed job names. "
            "Returning all of them. If this is not expected, change the regex."
        )

    if task_name_regex is not None and num_results == 0:
        print(
            "WARINING: Task regex passed, but it matches no dataset-committed job names. "
            "If this is not expected, change the regex."
        )

    return results_dataset_ids


def make_beaker_experiment_description(experiment_name: str) -> str:
    return f"Running {experiment_name}."


def make_beaker_experiment_name(experiment_name: str) -> str:
    command_str = experiment_name[:105]
    return f"{command_str}__{hash_object(experiment_name)[:10]}"
