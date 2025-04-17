import pandas as pd
from pathlib import Path


def __validate_dataset_dir(dataset_root_dir: Path) -> None:
    assert dataset_root_dir.exists()
    assert (dataset_root_dir / "images").exists()
    assert (dataset_root_dir / "annotations.txt").exists()


def add_csv_to_img_dir(dataset_root_dit: Path) -> None:
    __validate_dataset_dir(dataset_root_dit)
    data = pd.read_table(dataset_root_dit / "annotations.txt", sep=" ", header=None)
    data.columns = ["filename", "words"]
    data["filename"] = data["filename"].apply(lambda x: x.split("/")[-1])
    data.to_csv(dataset_root_dit / "images" / "labels.csv", index=False)
