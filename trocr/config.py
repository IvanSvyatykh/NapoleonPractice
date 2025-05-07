import yaml
from pathlib import Path


class TransfomerOCRConfig:
    def __init__(self, path_to_yaml: Path):
        assert path_to_yaml.exists()
        self.__config = yaml.load(open(path_to_yaml), Loader=yaml.SafeLoader)
        self.__path_to_model_dir = Path(self.__config["path_to_model_dir"])
        self.__path_to_processor_dir = Path(self.__config["path_to_processor_dir"])
        self.__path_to_test_dataset = Path(self.__config["path_to_test_dataset"])
        self.__path_to_train_dataset = Path(self.__config["path_to_train_dataset"])
        self.__path_to_val_dataset = Path(self.__config["path_to_val_dataset"])
        self.__train_metadata_file_name = Path(self.__config["train_metadata_file_name"])
        self.__val_metadata_file_name = Path(self.__config["val_metadata_file_name"])
        self.__output_dir = Path(self.__config["output_dir"])
        self.__device = self.__config["device"]
        self.__optimizer = self.__config["optimizer"]
        self.__optimizer_step = float(self.__config["optimizer_step"])
        self.__epoch = int(self.__config["epoch"])
        self.__batch_size = int(self.__config["batch_size"])
        self.__task_name = self.__config["task_name"]

    @property
    def task_name(self)->str:
        return self.__task_name

    @property
    def path_to_model_dir(self) -> Path:
        return self.__path_to_model_dir

    @property
    def processor_dir(self) -> Path:
        return self.__path_to_processor_dir

    @property
    def train_metadata_file_name(self) -> Path:
        return self.__train_metadata_file_name
    
    @property
    def val_metadata_file_name(self) -> Path:
        return self.__val_metadata_file_name

    @property
    def path_to_test_dataset(self) -> Path:
        return self.__path_to_test_dataset

    @property
    def path_to_train_dataset(self) -> Path:
        return self.__path_to_train_dataset

    @property
    def path_to_val_dataset(self) -> Path:
        return self.__path_to_val_dataset

    @property
    def output_dir(self) -> Path:
        return self.__output_dir

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
