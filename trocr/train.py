import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Tuple
from trocr_dataset import PriceTagDataset
from config import TransfomerOCRConfig
from model import TrOCRModel
from clearml import Task
from clearml_utils import (
    upload_dataset,
    download_dataset,
    upload_model_as_artifact,
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


def upload_dataset_task(trocr_config: TransfomerOCRConfig) -> Dict[str, str]:

    upload_dataset_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name="upload_dataset",
        task_type=Task.TaskTypes.data_processing,
    )

    upload_dataset_task.execute_remotely(queue_name=trocr_config.queue_name)
    train_dataset_id = upload_dataset(
        project_name="retail",
        dataset_name="retail-ocr-practice-train",
        dataset_dir=trocr_config.path_to_local_train_dataset,
    )
    val_dataset_id = upload_dataset(
        project_name="retail",
        dataset_name="retail-ocr-practice-val",
        dataset_dir=trocr_config.path_to_local_val_dataset,
    )
    result = {
        "task_id": upload_dataset_task.id,
        "train_dataset_id": train_dataset_id,
        "val_dataset_id": val_dataset_id,
    }
    return result


def download_dataset_task(
    trocr_config: TransfomerOCRConfig, dataset_meta: Dict[str, str]
) -> str:
    download_dataset_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name="download_dataset",
        task_type=Task.TaskTypes.data_processing,
    )
    download_dataset_task.set_parent(parent=dataset_meta["task_id"])
    download_dataset_task.execute_remotely(queue_name=trocr_config.queue_name)
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
    return download_dataset_task.id


def upload_model_taks(trocr_config: TransfomerOCRConfig, prev_task_id: str) -> str:
    upload_model_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name="upload_model",
        task_type=Task.TaskTypes.data_processing,
    )
    upload_model_task.set_parent(parent=prev_task_id)
    upload_model_task.execute_remotely(queue_name=trocr_config.queue_name)
    upload_model_as_artifact(trocr_config, upload_model_task)
    return upload_model_task.id


def download_model_task(trocr_config: TransfomerOCRConfig, prev_task_id: str) -> str:

    download_model_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name="upload_model",
        task_type=Task.TaskTypes.data_processing,
    )

    download_model_task.set_parent(parent=prev_task_id)
    download_model_task.execute_remotely(queue_name=trocr_config.queue_name)
    download_pretrained_model(
        prev_task_id,
        artefact_name=trocr_config.artifact_model_name,
        save_dir=trocr_config.agent_model_dir,
    )
    download_pretrained_model(
        prev_task_id,
        artefact_name=trocr_config.artifact_processor_name,
        save_dir=trocr_config.agent_processor_dir,
    )
    return download_model_task.id


def main(trocr_config: TransfomerOCRConfig) -> None:
    #Task.add_requirements("requirements.txt")
    res = upload_dataset_task(trocr_config)
    download_dataset_id = download_dataset_task(trocr_config, res)
    upload_model_id = upload_model_taks(trocr_config, download_dataset_id)
    download_model_id = download_model_task(trocr_config, upload_model_id)
    train_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name="upload_model",
        task_type=Task.TaskTypes.training,
    )
    download_model_task.set_parent(parent=download_model_id)
    train_task.execute_remotely(queue_name=trocr_config.queue_name)
    train_trocr(trocr_config, train_task)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOCRConfig(args.path_to_yaml_config)
    main(yaml_config)
