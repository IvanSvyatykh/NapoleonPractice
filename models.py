import re
import time
import easyocr
import pandas as pd
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR
from PIL import Image


class OCRModel:
    def __init__(self, config):
        self.__config = config

    def inference(self, path_to_photo: Path) -> tuple[str, float]:
        raise NotImplementedError("Not implemented method !")

    def test(self, test_df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
        res = test_df.copy()
        inference_res = {"model_answer": [], "inference_time": []}
        for _, row in tqdm(
            res.iterrows(), total=res.shape[0], desc="Inference on test dataset"
        ):
            answer, inference_time = self.inference(Path(row["image_path"]))
            inference_res["model_answer"].append(answer)
            inference_res["inference_time"].append(inference_time)
        res["model_answer"] = inference_res["model_answer"]
        res["inference_time"] = inference_res["inference_time"]
        res["device"] = self.device
        self.__save_test_res_to_dir(res)
        return res

    def save_model(self, output_dir: Path) -> None:
        raise NotImplementedError("Not implemented method !")

    @property
    def model_name(self) -> str:
        return self.__config.model_dir.name

    @property
    def model_dir(self) -> Path:
        return self.__config.model_dir

    @property
    def device(self) -> str:
        return self.__config.device

    @property
    def path_for_res_file(self) -> Path:
        return self.res_dir_path / f"{self.model_name}_on_{self.device}.csv"

    @property
    def res_dir_path(self) -> Path:
        return self.__config.res_dir_path

    def __save_test_res_to_dir(self, df: pd.DataFrame) -> Path:
        self.path_for_res_file.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(self.path_for_res_file, index=False)
        return self.path_for_res_file


class PaddleOCRModel(OCRModel):
    def __init__(self, paddleocr_config: OCRModelConfig) -> None:
        super().__init__(paddleocr_config)
        self.__model = PaddleOCR(lang="en", use_gpu=paddleocr_config.device == "cuda")

    def inference(self, path_to_photo: Path) -> tuple[str, float]:
        start_time = time.time()
        result = self.__model.ocr(str(path_to_photo), rec=True, det=True, cls=True)
        result_text = ""
        for idx in range(len(result)):
            res = result[idx]
            if res is None:
                break
            for line in res:
                temp = str(line[-1][0])
                if temp.isdigit():
                    result_text += temp
        end_time = time.time()
        return result_text, end_time - start_time


class EasyOCRModel(OCRModel):
    def __init__(self, paddleocr_config: OCRModelConfig) -> None:
        super().__init__(paddleocr_config)
        self.__model = easyocr.Reader(
            ["ru"],
            model_storage_directory=self.model_dir,
            gpu=self.device == "cuda",
        )

    def inference(self, path_to_photo: Path) -> tuple[str, float]:
        start_time = time.time()
        result = self.__model.readtext(str(path_to_photo))
        res_dic = {}
        for res in result:
            coords = res[0]
            text = res[1]
            conf = res[2]
            if not text.isdigit():
                continue

            res_dic[min([point[0] for point in coords])] = text, conf
        answer = sorted(res_dic.values())
        s = ""
        for a in answer:
            s += a[0]
        end_time = time.time()
        return s, end_time - start_time

    def train(self, train_yaml_config: Path):
        pass


class DonutOCRModel(OCRModel):
    def __init__(self, donut_config: TransfomerOCRConfig):
        super().__init__(donut_config)
        self.__model = VisionEncoderDecoderModel.from_pretrained(self.model_dir).to(
            self.device
        )
        self.__processor = DonutProcessor.from_pretrained(donut_config.processor_dir)

    def set_prompt(self, prompt: str) -> None:
        self.__prompt = prompt

    def inference(self, path_to_photo: Path) -> tuple[str, float]:
        start_time = time.time()
        image = Image.open(path_to_photo).convert("RGB")
        decoder_input_ids = self.__processor.tokenizer(
            self.__prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        pixel_values = self.__processor(image, return_tensors="pt").pixel_values

        outputs = self.__model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.__model.decoder.config.max_position_embeddings,
            pad_token_id=self.__processor.tokenizer.pad_token_id,
            eos_token_id=self.__processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.__processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.__processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.__processor.tokenizer.eos_token, "").replace(
            self.__processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        res = self.__processor.token2json(sequence)["answer"]
        end_time = time.time()
        return res, end_time - start_time
