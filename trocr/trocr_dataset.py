import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor


class PriceTagDataset(Dataset):
    def __init__(
        self,
        dataset_root_dir: Path,
        path_for_metadata_file: Path,
        processor: TrOCRProcessor,
    ) -> None:
        assert dataset_root_dir.exists(), f"{dataset_root_dir} does not exists"
        assert path_for_metadata_file.exists(), f"{path_for_metadata_file} does not exists"

        self.__dataset_root_dir = dataset_root_dir
        self.__metadata = path_for_metadata_file
        self.__df = self.__read_txt_metadata()
        self.__max_target_length = self.__df["text"].str.len().max()
        self.__processor = processor

    def __read_txt_metadata(self) -> pd.DataFrame:
        df = pd.read_table(self.__metadata, encoding="utf8", header=None, sep=" ")
        df.columns = ["file_name", "text"]
        df["text"] = df["text"].astype(str)
        return df

    def __len__(self) -> int:
        return self.__df.shape[0]

    def __getitem__(self, idx: int):
        file_name = self.__df["file_name"][idx]
        text = self.__df["text"][idx]
        image = Image.open(self.__dataset_root_dir / file_name).convert("RGB")
        pixel_values = self.__processor(image, return_tensors="pt").pixel_values
        labels = self.__processor.tokenizer(
            text, padding="max_length", max_length=self.__max_target_length
        ).input_ids
        labels = [
            label if label != self.__processor.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding
