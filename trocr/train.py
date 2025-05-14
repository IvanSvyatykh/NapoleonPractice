import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Tuple
from trocr.utils.trocr_dataset import PriceTagDataset
from config import TransfomerOCRConfig
from model import TrOCRModel
from clearml import Task
from trocr.utils.clearml_utils import (
    download_dataset,
    download_pretrained_model,
)


def train_trocr(trocr_config: TransfomerOCRConfig, task: Task):
    model = TrOCRModel(trocr_config)
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
        processor=model.processor,
    )
    model.train(train_dataset, val_dataset, task.get_logger())


def download_dataset_task(
    trocr_config: TransfomerOCRConfig, dataset_meta: Dict[str, str]
) -> None:
    download_dataset(
        {
            "dataset_path": trocr_config.agent_train_dataset,
            "dataset_clearml_id": dataset_meta["train_dataset_id"],
        }
    )
    download_dataset(
        {
            "dataset_path": trocr_config.agent_val_dataset,
            "dataset_clearml_id": dataset_meta["val_dataset_id"],
        }
    )


def download_model_task(trocr_config: TransfomerOCRConfig, task: Task) -> None:

    download_pretrained_model(
        task,
        artefact_name=trocr_config.artifact_model_name,
        save_dir=trocr_config.agent_model_dir,
    )
    download_pretrained_model(
        task,
        artefact_name=trocr_config.artifact_processor_name,
        save_dir=trocr_config.agent_processor_dir,
    )

def main(trocr_config: TransfomerOCRConfig) -> None:
    Task.add_requirements("requirements.txt")
    train_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name=trocr_config.task_name,
        task_type=Task.TaskTypes.training,
    )
    train_task.execute_remotely(queue_name=trocr_config.queue_name)
    download_model_task(trocr_config, train_task)
    download_dataset_task(trocr_config, res)
    train_trocr(trocr_config, train_task)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOCRConfig(args.path_to_yaml_config)
    main(yaml_config)
