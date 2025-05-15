import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict
from utils.load_config import TransfomerOcrLoadConfig
from clearml import Task
from utils.clearml_utils import upload_model_as_artifact


def main(trocr_config: TransfomerOcrLoadConfig) -> None:
    Task.add_requirements("./requirements.txt")
    load_task = Task.init(
        project_name="retail/ocr/trocr",
        task_name=trocr_config.task_name,
        task_type=Task.TaskTypes.data_processing,
    )
    upload_model_as_artifact(trocr_config, load_task)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOcrLoadConfig(args.path_to_yaml_config)
    main(yaml_config)
