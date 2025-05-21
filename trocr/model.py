import os
import time
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from pathlib import Path
from tqdm import tqdm
from utils.train_config import TransfomerOcrTrainConfig
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Optimizer, Adadelta
from clearml import Task
from typing import List, Tuple


class TrOCRModel:
    def __init__(
        self, trocr_config: TransfomerOcrTrainConfig, model_dir: str, processor_dir: str
    ):
        self.__config = trocr_config
        self.__model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(
            self.__config.device
        )
        self.__processor = TrOCRProcessor.from_pretrained(processor_dir)

    def inference(self, path_to_photo: Path) -> Tuple[str, float]:
        start_time = time.time()
        image = Image.open(path_to_photo).convert("RGB")
        pixel_values = self.__processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.__config.device)
        generated_ids = self.__model.generate(pixel_values)
        end_time = time.time()
        time_delta = end_time - start_time
        return (
            self.__processor.batch_decode(generated_ids, skip_special_tokens=True)[0],
            time_delta,
        )

    def train(self, train_dataset: Dataset, val_dataset: Dataset, task: Task) -> None:
        if self.__config.optimizer not in optimizers.keys():
            raise ValueError(f"Supports only this optimizers : {optimizers.values()}")
        torch.cuda.empty_cache()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.__config.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.__config.batch_size,
            pin_memory=True,
        )
        optimizer = optimizers[self.__config.optimizer](
            self.__model.parameters(), lr=self.__config.optimizer_step
        )
        self.__set_train_params_for_model()
        prev_val_accuracy = 0
        best_model_path = None
        for epoch in range(self.__config.epoch):
            self.__model.train()
            train_loss = self.__train_loop(train_dataloader, optimizer)
            task.get_logger().report_scalar(
                "Train loss", "train", train_loss, iteration=epoch
            )
            print(f"Train loss after epoch {epoch}:", train_loss)
            self.__model.eval()
            val_accuracy, val_loss = self.__val_loop(val_dataloader)
            task.get_logger().report_scalar(
                "Val accuracy", "accuracy", val_accuracy, iteration=epoch
            )
            task.get_logger().report_scalar(
                "Val loss", "loss", val_loss, iteration=epoch
            )
            print(f"Val loss after epoch {epoch}:", val_loss)
            print(f"Validation Accuracy after epoch {epoch}:", val_accuracy)
            if prev_val_accuracy < val_accuracy:
                prev_val_accuracy = val_accuracy
                best_model_path = (
                    self.__config.output_dir / f"val_accuracy_{val_accuracy}"
                )
                self.save_model(best_model_path)
                task.upload_artifact("best model", artifact_object=best_model_path)

        self.save_model(
            self.__config.output_dir / f"finished_model_accuracy_{val_accuracy}"
        )
        task.upload_artifact("result model", artifact_object=best_model_path)

    def save_model(self, output_dir: Path) -> None:

        output_dir.mkdir(exist_ok=True, parents=True)
        processor_dir = output_dir / "processor"
        processor_dir.mkdir(exist_ok=True)
        model_dir = output_dir / "model"
        model_dir.mkdir(exist_ok=True)

        self.__processor.save_pretrained(processor_dir)
        self.__model.save_pretrained(model_dir)

    def __train_loop(
        self,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
    ) -> float:
        train_loss = 0.0
        for images, labels in tqdm(train_dataloader):
            images = images.to(self.__config.device)
            labels = labels.to(self.__config.device)
            outputs = self.__model(images, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        return train_loss / len(train_dataloader)

    def __val_loop(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        count = 0
        elements = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):
                images = images.to(self.__config.device)
                labels = labels.to(self.__config.device)
                predictions = self.__model.generate(images)
                predictions = self.__processor.batch_decode(
                    predictions, skip_special_tokens=True
                )
                outputs = self.__model(images, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                elements += len(images)
                count += self.__count_correct_preds(
                    model_output=predictions, labels=labels
                )

        return count / elements, val_loss / len(val_dataloader)

    def __count_correct_preds(
        self, model_output: List[str], labels: torch.Tensor
    ) -> float:
        # add pad_token to
        labels[labels == -100] = self.__processor.tokenizer.pad_token_id
        predictions = np.array(model_output)
        label_str = np.array(
            self.__processor.batch_decode(labels, skip_special_tokens=True)
        )
        assert len(predictions) == len(label_str)
        return np.sum(predictions == label_str)

    def __set_train_params_for_model(self) -> None:
        self.__model.config.decoder_start_token_id = (
            self.__processor.tokenizer.cls_token_id
        )
        self.__model.config.pad_token_id = self.__processor.tokenizer.pad_token_id
        self.__model.config.vocab_size = self.__model.config.decoder.vocab_size
        self.__model.config.eos_token_id = self.__processor.tokenizer.sep_token_id
        self.__model.config.max_length = 64
        self.__model.config.early_stopping = True
        self.__model.config.no_repeat_ngram_size = 3
        self.__model.config.length_penalty = 2.0
        self.__model.config.num_beams = 4

    @property
    def processor(self) -> TrOCRProcessor:
        return self.__processor

    @property
    def model(self) -> VisionEncoderDecoderModel:
        return self.__model


optimizers = {"AdamW": AdamW, "Adadelta": Adadelta}
