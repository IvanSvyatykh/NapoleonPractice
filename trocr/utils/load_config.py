import yaml
from pathlib import Path


class TransfomerOcrLoadConfig:
    def __init__(self, path_to_yaml: Path):
        assert path_to_yaml.exists()
        self.__config = yaml.load(open(path_to_yaml), Loader=yaml.SafeLoader)
        # Local paths
        self.__path_to_local_model_dir = Path(self.__config["path_to_local_model_dir"])
        self.__path_to_local_processor_dir = Path(
            self.__config["path_to_local_processor_dir"]
        )
        self.__path_to_local_train_dataset = Path(
            self.__config["path_to_local_train_dataset"]
        )
        self.__path_to_local_val_dataset = Path(
            self.__config["path_to_local_val_dataset"]
        )
        # Artifacts name
        self.__artifact_processor_name = self.__config["artifact_processor_name"]
        self.__artifact_model_name = self.__config["artifact_model_name"]

        # Other info
        self.__task_name = self.__config["task_name"]

    # Local paths
    @property
    def path_to_local_model_dir(self) -> Path:
        return self.__path_to_local_model_dir

    @property
    def path_to_local_processor_dir(self) -> Path:
        return self.__path_to_local_processor_dir

    @property
    def path_to_local_train_dataset(self) -> Path:
        return self.__path_to_local_train_dataset

    @property
    def path_to_local_val_dataset(self) -> Path:
        return self.__path_to_local_val_dataset

    # Artifacts name

    @property
    def artifact_model_name(self) -> str:
        return self.__artifact_model_name

    @property
    def artifact_processor_name(self) -> str:
        return self.__artifact_processor_name

    # Other info

    @property
    def task_name(self) -> str:
        return self.__task_name
