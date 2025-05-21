import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple

import pandas as pd
from tqdm import tqdm
from utils.trocr_dataset import PriceTagDataset
from utils.train_config import TransfomerOcrTrainConfig
from model import TrOCRModel
from clearml import Task
from utils.clearml_utils import (
    download_dataset,
    download_pretrained_model,
)


def __validate_dataset_dir(dataset_root_dir: Path, metadata_filename: str) -> None:
    assert dataset_root_dir.exists()
    assert (dataset_root_dir / "images").exists()
    assert (
        dataset_root_dir / metadata_filename
    ).exists(), f"{dataset_root_dir / metadata_filename} does not exists"


def create_df_from_txt(dataset_root_dit: Path, metadata_filename: str) -> pd.DataFrame:
    __validate_dataset_dir(dataset_root_dit, metadata_filename)
    data = pd.read_table(dataset_root_dit / metadata_filename, sep=" ", header=None)
    data.columns = ["filename", "words"]
    data["filename"] = data["filename"].apply(lambda x: x.split("/")[-1])
    return data


def test_trocr(
    trocr_config: TransfomerOcrTrainConfig,
    task: Task,
    model_dir: str,
):
    model = TrOCRModel(trocr_config, model_dir, None)

    df = create_df_from_txt(
        trocr_config.agent_val_dataset, trocr_config.val_metadata_file_name
    )
    result = df.copy(deep=True)
    all_preds = []
    for _, row in tqdm(df.iterrows(), "Test model", total=len(df)):
        path_to_file = trocr_config.agent_val_dataset / "images" / row["filename"]
        pred, _ = model.inference(path_to_file)
        all_preds.append(pred)
    result["preds"] = all_preds
    path_for_csv = Path(trocr_config.output_dir) / (f"{trocr_config.task_name}.csv")
    result.to_csv(path_for_csv, index=False)
    task.upload_artifact("predications", artifact_object=path_for_csv)


def download_dataset_task(trocr_config: TransfomerOcrTrainConfig) -> None:
    download_dataset(
        {
            "dataset_path": trocr_config.agent_val_dataset,
            "dataset_clearml_id": trocr_config.val_dataset_id,
        }
    )


def download_model_task(trocr_config: TransfomerOcrTrainConfig) -> str:

    model_dir = download_pretrained_model(
        artefact_name=trocr_config.model_artifact,
        save_dir=trocr_config.agent_model_dir,
        task_id=trocr_config.task_id,
    )
    return model_dir


def main(trocr_config: TransfomerOcrTrainConfig) -> None:
    Task.add_requirements("requirements.txt")
    Task.add_requirements("boto3", "1.9.0")
    train_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name=trocr_config.task_name,
        task_type=Task.TaskTypes.testing,
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
    model_dir = download_model_task(trocr_config)
    download_dataset_task(trocr_config)
    test_trocr(trocr_config, train_task, model_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOcrTrainConfig(args.path_to_yaml_config)
    main(yaml_config)
