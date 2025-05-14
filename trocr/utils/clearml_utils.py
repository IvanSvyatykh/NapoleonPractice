import logging
import zipfile
import os
from typing import Dict
from clearml import Dataset, Task
from pathlib import Path
from trocr.utils.load_config import TransfomerOcrLoadConfig


def upload_dataset(
    project_name: str, dataset_name: str, dataset_dir: Path, parent_dataset: str = None
) -> str:
    """
    Загружает датасет в ClearML

    Аргументы:
        project_name (str): Название проекта в ClearML.
        dataset_name (str): Название датасета.
        dataset_dir (str): Путь до локальной папки с данными.
        parent_dataset (str, optional): ID родительского датасета. По умолчанию None.

    Возвращает:
        str: ID загруженного датасета.

    Пример использования:
        dataset_id = upload_dataset("retail", "retail-ocr", "/path/to/dataset")
        dataset_id = upload_dataset("retail", "retail-ocr", "/path/to/dataset", "parent_dataset_id")
    """
    dataset = Dataset.create(
        dataset_project=project_name,
        dataset_name=dataset_name,
        parent_datasets=[parent_dataset] if parent_dataset else None,
    )
    dataset.sync_folder(dataset_dir)
    dataset.finalize(auto_upload=True)
    dataset_id = dataset.id
    logging.info(f"Successfully upload dataset. Dataset id {dataset_id}")
    return dataset_id


def upload_model_as_artifact(trocr_config: TransfomerOcrLoadConfig, task: Task) -> None:

    is_model_artifact = task.upload_artifact(
        name=trocr_config.artifact_model_name,
        artifact_object=trocr_config.path_to_local_model_dir,
    )
    if not is_model_artifact:
        logging.error(f"Could not upload model as artifact.")
        raise AttributeError(f"Stop execution because artifact was't uploaded")

    is_processor_artifact = task.upload_artifact(
        name=trocr_config.artifact_processor_name,
        artifact_object=trocr_config.path_to_local_processor_dir,
    )

    if not is_processor_artifact:
        logging.error(f"Could not upload processor as artifact.")
        raise AttributeError(f"Stop execution because artifact was't uploaded")


def download_dataset(config: Dict[str, str]) -> str:
    dataset_path = config["dataset_path"]

    if os.path.exists(dataset_path) and os.listdir(dataset_path):
        logging.warning(
            f"Датасет уже существует в {dataset_path}. Продолжаем выполнение."
        )
        return dataset_path
    else:
        os.makedirs(dataset_path, exist_ok=True)

        dataset = Dataset.get(dataset_id=config["dataset_clearml_id"])
        dataset_path = dataset.get_mutable_local_copy(
            target_folder=dataset_path, overwrite=False
        )
        logging.info(f"Датасет загружен в {dataset_path}.")
        return dataset_path


def download_pretrained_model(task: Task, save_dir: str, artefact_name: str) -> None:

    if not task:
        return None

    logging.warning(f"Загрузка предобученной модели из задачи ClearML ID: {task.id}")
    os.makedirs(save_dir, exist_ok=True)

    # Получаем задачу ClearML

    if task.artifacts.get(artefact_name):
        # Сначала пробуем найти best_model
        model_path = task.artifacts["best_model"].get_local_copy()
    else:
        logging.warning(f"Не найдены артефакты {artefact_name} в указанной задаче")
        raise AttributeError(f"Can not find artefact with name {artefact_name}")

    if model_path and model_path.endswith(".zip"):
        with zipfile.ZipFile(model_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
    else:
        logging.warning(
            f"Предупреждение: Загруженный артефакт не является zip-архивом или не существует: {model_path}"
        )
        return None
