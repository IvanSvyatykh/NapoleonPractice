import time
import torch
import evaluate
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from pathlib import Path
from tqdm import tqdm
from config import TransfomerOCRConfig
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Optimizer, Adadelta
from clearml import Logger


class TrOCRModel:
    def __init__(self, trocr_config: TransfomerOCRConfig):
        self.__config = trocr_config
        self.__model = VisionEncoderDecoderModel.from_pretrained(
            self.__config.path_to_model_dir
        ).to(self.__config.device)
        self.__processor = TrOCRProcessor.from_pretrained(self.__config.processor_dir)

    def inference(self, path_to_photo: Path) -> tuple[str, float]:
        start_time = time.time()
        image = Image.open(path_to_photo).convert("RGB")
        pixel_values = self.__processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.__config.device)
        generated_ids = self.__model.generate(pixel_values)
        end_time = time.time()
        time_delta = end_time - start_time
        return self.__processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ], time_delta

    def train(
        self, train_dataset: Dataset, val_dataset: Dataset, logger: Logger
    ) -> None:
        if self.__config.optimizer not in optimizers.keys():
            raise ValueError(f"Supports only this optimizers : {optimizers.values()}")
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.__config.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.__config.batch_size)
        optimizer = optimizers[self.__config.optimizer](
            self.__model.parameters(), lr=self.__config.optimizer_step
        )
        self.__set_train_params_for_model()
        prev_val_cer = 11111
        for epoch in range(self.__config.epoch):
            self.__model.train()
            train_loss = self.__train_loop(train_dataloader, optimizer, epoch, logger)
            print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))
            self.__model.eval()
            val_cer = self.__val_loop(val_dataloader)
            print("Validation CER:", val_cer / len(val_dataloader))
            if prev_val_cer > val_cer:
                prev_val_cer = val_cer
                self.save_model(self.__config.output_dir / f"val_cer_{val_cer}")
        self.save_model(self.__config.output_dir / f"finish_model_val_cer_{val_cer}")

    def save_model(self, output_dir: Path) -> None:
        assert output_dir.exists()
        processor_dir = output_dir / "processor"
        model_dir = output_dir / "model"

        self.__processor.save_pretrained(processor_dir)
        self.__model.save_pretrained(model_dir)

    def __train_loop(
        self,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        epoch_num: int,
        logger: Logger,
    ) -> float:
        train_loss = 0.0
        plot_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader, 0)):
            for k, v in batch.items():
                batch[k] = v.to(self.__config.device)

            # forward + backward + optimize
            outputs = self.__model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            plot_loss += loss.iem()
            if i % 2000 == 1999:
                print(f"[{epoch_num + 1}, {i + 1:5d}] loss: {plot_loss / 2000:.3f}")
                logger.report_scalar("loss", "train", plot_loss / 2000, iteration=i)
                plot_loss = 0.0

        return train_loss

    def __val_loop(self, val_dataloader: DataLoader) -> float:
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                outputs = self.__model.generate(
                    batch["pixel_values"].to(self.__config.device)
                )
                cer = self.__compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer
        return valid_cer

    def __compute_cer(pred, processor: TrOCRProcessor):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        cer_metric = evaluate.load("cer")
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

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


optimizers = {"AdamW": AdamW, "Adadelta": Adadelta}
