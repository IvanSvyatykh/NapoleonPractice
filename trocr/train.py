from pathlib import Path
from argparse import ArgumentParser
from trocr_dataset import PriceTagDataset
from config import TransfomerOCRConfig
from model import TrOCRModel
from clearml import Task


def train_trocr(trocr_config: TransfomerOCRConfig ,task:Task):
    model = TrOCRModel(trocr_config)
    train_dataset = PriceTagDataset(
        dataset_root_dir=trocr_config.path_to_train_dataset,
        path_for_metadata_file=trocr_config.path_to_train_dataset
        / trocr_config.train_metadata_file_name,
        processor=model.processor,
    )
    val_dataset = PriceTagDataset(
        dataset_root_dir=trocr_config.path_to_val_dataset,
        path_for_metadata_file=trocr_config.path_to_val_dataset
        / trocr_config.val_metadata_file_name,
        processor=model.processor,
    )
    model.train(train_dataset, val_dataset , None)


def main(trocr_config: TransfomerOCRConfig) -> None:
    task = Task.init(project_name="retail/ocr/trocr", task_name="train_model")
    train_trocr(trocr_config,None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = TransfomerOCRConfig(args.path_to_yaml_config)
    main(yaml_config)
