import yaml
from pathlib import Path


class EasyOCRConfig:
    def __init__(self, path_to_yaml: Path):
        assert path_to_yaml.exists()
        self.__config = yaml.load(open(path_to_yaml), Loader=yaml.SafeLoader)
        self.__path_to_model_dir = Path(self.__config["path_to_model_dir"])
        self.__path_to_dataset = Path(self.__config["dataset_dir"])
        self.__metadata_filename = self.__config["metadata_filename"]
        self.__result_dir = Path(self.__config["results_dir"])
        self.__device = self.__config["device"]

    @property
    def path_to_dataset(self) -> Path:
        return self.__path_to_dataset
    
    @property
    def device(self)->str:
        return self.__device

    @property
    def result_dir(self) -> Path:
        return self.__result_dir

    @property
    def path_to_model_dir(self) -> Path:
        return self.__path_to_model_dir

    @property
    def metadata_filename(self) -> str:
        return self.__metadata_filename
