import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple
from utils.trocr_dataset import PriceTagDataset
from utils.train_config import TransfomerOcrTrainConfig
from model import TrOCRModel
from clearml import Task
from utils.clearml_utils import (
    download_dataset,
    download_pretrained_model,
)


def train_trocr(trocr_config: TransfomerOcrTrainConfig, task: Task,model_dir:str,processor_dir:str):
    model = TrOCRModel(trocr_config,model_dir,processor_dir)
    train_dataset = PriceTagDataset(
        dataset_root_dir=trocr_config.agent_train_dataset,
        path_for_metadata_file=trocr_config.agent_train_dataset
        / trocr_config.train_metadata_file_name,
        processor=model.processor,
    )
    val_dataset = PriceTagDataset(
        dataset_root_dir=trocr_config.agent_val_dataset,
        path_for_metadata_file=trocr_config.agent_val_dataset
        / trocr_config.val_metadata_file_name,
        processor=model.processor
    )
    model.train(train_dataset, val_dataset, task)


def download_dataset_task(trocr_config: TransfomerOcrTrainConfig) -> None:
    download_dataset(
        {
            "dataset_path": trocr_config.agent_train_dataset,
            "dataset_clearml_id": trocr_config.train_dataset_id,
        }
    )
    download_dataset(
        {
            "dataset_path": trocr_config.agent_val_dataset,
            "dataset_clearml_id": trocr_config.val_dataset_id,
        }
    )


def download_model_task(trocr_config: TransfomerOcrTrainConfig) -> Tuple[str,str]:

    model_dir = download_pretrained_model(
        artefact_name=trocr_config.model_artifact,
        save_dir=trocr_config.agent_model_dir,
        task_id=trocr_config.task_id,
    )
    processor_dir = download_pretrained_model(
        artefact_name=trocr_config.processor_artifact,
        save_dir=trocr_config.agent_processor_dir,
        task_id=trocr_config.task_id,
    )
    return model_dir,processor_dir


def main(trocr_config: TransfomerOcrTrainConfig) -> None:
    Task.add_requirements("requirements.txt")
    Task.add_requirements("boto3", "1.9.0")
    train_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name=trocr_config.task_name,
        task_type=Task.TaskTypes.training,
    )
    train_task.set_base_docker(
        docker_image="nvidia/cuda:12.2.2-base-ubuntu20.04",
        docker_setup_bash_script=[
            "apt-get update && apt-get install -y",
            "curl \ git \ build-essential \ python3-dev \ python3-pip \ locales \ && rm -rf /var/lib/apt/lists/*",
            "pip install --upgrade pip setuptools",
            "pip install boto3",
        ],
    )
    train_task.execute_remotely(queue_name=trocr_config.queue_name)
    model_dir,processor_dir =download_model_task(trocr_config)
    download_dataset_task(trocr_config)
    train_trocr(trocr_config, train_task,model_dir,processor_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOcrTrainConfig(args.path_to_yaml_config)
    main(yaml_config)
