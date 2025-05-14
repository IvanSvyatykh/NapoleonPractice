import yaml
from pathlib import Path


class TransfomerOcrTrainConfig:
    def __init__(self, path_to_yaml: Path):
        assert path_to_yaml.exists()
        self.__config = yaml.load(open(path_to_yaml), Loader=yaml.SafeLoader)
        # Datasets
        self.__train_dataset_id = self.__config["train_dataset_id"]
        self.__val_dataset_id = self.__config["val_dataset_id"]
        # Model
        self.__task_id = self.__config["task_id"]
        self.__model_artifact = self.__config["model_artifact"]
        self.__processor_artifact = self.__config["processor_artifact"]
        # Artifacts name
        self.__agent_model_dir = Path(self.__config["agent_model_dir"])
        self.__agent_processor_dir = Path(self.__config["agent_processor_dir"])
        self.__agent_train_dataset = Path(self.__config["agent_train_dataset"])
        self.__agent_val_dataset = Path(self.__config["agent_val_dataset"])

        # Other info
        self.__train_metadata_file_name = Path(
            self.__config["train_metadata_file_name"]
        )
        self.__val_metadata_file_name = Path(self.__config["val_metadata_file_name"])
        self.__output_dir = Path(self.__config["output_dir"])
        self.__device = self.__config["device"]
        self.__optimizer = self.__config["optimizer"]
        self.__optimizer_step = float(self.__config["optimizer_step"])
        self.__epoch = int(self.__config["epoch"])
        self.__batch_size = int(self.__config["batch_size"])
        self.__task_name = self.__config["task_name"]
        self.__queue_name = self.__config["queue_name"]

    # Datasets
    @property
    def train_dataset_id(self) -> Path:
        return self.__train_dataset_id

    @property
    def val_dataset_id(self) -> Path:
        return self.__val_dataset_id

    # Model
    @property
    def task_id(self) -> Path:
        return self.__task_id

    @property
    def model_artifact(self) -> Path:
        return self.__model_artifact

    @property
    def processor_artifact(self) -> Path:
        return self.__processor_artifact

    # Artifacts name

    @property
    def agent_model_dir(self) -> str:
        return self.__agent_model_dir

    @property
    def agent_processor_dir(self) -> str:
        return self.__agent_processor_dir

    @property
    def agent_train_dataset(self) -> str:
        return self.__agent_train_dataset

    @property
    def agent_val_dataset(self) -> str:
        return self.__agent_val_dataset

    # Other info
    @property
    def train_metadata_file_name(self) -> Path:
        return self.__train_metadata_file_name

    @property
    def val_metadata_file_name(self) -> Path:
        return self.__val_metadata_file_name

    @property
    def output_dir(self) -> Path:
        return self.__output_dir

    @property
    def task_name(self) -> str:
        return self.__task_name

    @property
    def device(self) -> str:
        return self.__device

    @property
    def optimizer(self) -> str:
        return self.__optimizer

    @property
    def optimizer_step(self) -> float:
        return self.__optimizer_step

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def queue_name(self) -> str:
        return self.__queue_name
