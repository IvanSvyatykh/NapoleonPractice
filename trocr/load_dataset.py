import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict
from utils.load_config import TransfomerOcrLoadConfig
from clearml import Task
from trocr.utils.clearml_utils import upload_dataset, upload_model_as_artifact


def upload_dataset_task(trocr_config: TransfomerOcrLoadConfig) -> Dict[str, str]:
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
        "train_dataset_id": train_dataset_id,
        "val_dataset_id": val_dataset_id,
    }
    return result


def main(trocr_config: TransfomerOcrLoadConfig) -> None:
    Task.add_requirements("./requirements.txt")
    load_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name=trocr_config.task_name,
        task_type=Task.TaskTypes.data_processing,
    )
    res = upload_dataset_task(trocr_config)
    upload_model_as_artifact(trocr_config, load_task)
    load_task.upload_artifact(
        name="Train dataset id",
        artifact_object=res["train_dataset_id"],
    )
    load_task.upload_artifact(
        name="Val dataset id",
        artifact_object=res["val_dataset_id"],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOcrLoadConfig(args.path_to_yaml_config)
    main(yaml_config)
