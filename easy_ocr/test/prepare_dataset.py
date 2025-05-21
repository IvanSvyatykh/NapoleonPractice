import pandas as pd
from pathlib import Path


def __validate_dataset_dir(dataset_root_dir: Path, metadata_filename: str) -> None:
    assert dataset_root_dir.exists()
    assert (dataset_root_dir / "images").exists()
    assert (
        dataset_root_dir / metadata_filename
    ).exists(), f"{dataset_root_dir / metadata_filename} does not exists"


def create_df_from_txt(dataset_root_dit: Path, metadata_filename: str)->pd.DataFrame:
    __validate_dataset_dir(dataset_root_dit, metadata_filename)
    data = pd.read_table(dataset_root_dit / metadata_filename, sep=" ", header=None)
    data.columns = ["filename", "words"]
    data["filename"] = data["filename"].apply(lambda x: x.split("/")[-1])
    return data

def add_csv_to_img_dir(dataset_root_dit: Path, metadata_filename: str) -> None:
    data = create_df_from_txt(dataset_root_dit, metadata_filename)
    data.to_csv(dataset_root_dit / "images" / "labels.csv", index=False)



    