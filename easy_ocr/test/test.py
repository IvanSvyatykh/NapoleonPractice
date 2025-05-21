import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.init as init
import yaml
import pandas as pd
from tqdm import tqdm
from clearml import Task
from pathlib import Path
from utils import CTCLabelConverter, AttnLabelConverter, AttrDict
from model import Model
from typing import Union
from PIL import Image
from argparse import ArgumentParser
from img_preprocessing import prepare_photo
from prepare_dataset import create_df_from_txt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"


def get_config(file_path: Path) -> AttrDict:
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == "None":
        characters = ""
        for data in opt["select_data"].split("-"):
            csv_path = os.path.join(opt["train_data"], data, "labels.csv")
            df = pd.read_csv(
                csv_path,
                sep="^([^,]+),",
                engine="python",
                usecols=["filename", "words"],
                keep_default_na=False,
            )
            all_char = "".join(df["words"].astype(str))
            characters += "".join(set(all_char))
        characters = sorted(set(characters))
        opt.character = "".join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    return opt


def load_model(config: AttrDict) -> Model:
    pretrained_dict = torch.load(config["saved_model"])
    model = Model(config)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(pretrained_dict, strict=False)  
    model.eval()
    
    return model


def create_converter(config: AttrDict) -> Union[CTCLabelConverter, AttnLabelConverter]:
    if "CTC" in config.Prediction:
        converter = CTCLabelConverter(config.character)
    else:
        converter = AttnLabelConverter(config.character)
    return converter


def test(config: AttrDict, task: Task) -> None:
    converter = create_converter(config)
    config.num_class = len(converter.character)
    if config.rgb:
        config.input_channel = 3
    model = load_model(config)
    model.eval()
    path_to_data_dir = Path(config["valid_data"])
    metadata_file_name = config["val_metadata_file_name"]
    df = create_df_from_txt(path_to_data_dir, metadata_file_name)
    result = df.copy(deep=True)
    all_preds = []
    for _, row in tqdm(df.iterrows(),"Test model",total=len(df)):
        path_to_file = path_to_data_dir / config["select_data"] / row["filename"]
        img = Image.open(path_to_file)
        tensor = prepare_photo(img,config["imgH"],config["imgW"]).unsqueeze(0).to(device)
        preds = model(tensor, None)
        preds_size = torch.IntTensor([preds.size(1)] * 1)
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
        all_preds.append(preds_str[0])
    result["preds"] = all_preds
    
    path_for_csv = Path(config["result_dir"])/(f"{config['task_name']}.csv") 
    result.to_csv(path_for_csv,index=False)
    task.upload_artifact("predications", artifact_object=path_for_csv)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-conf", type=Path, required=True)
    args = parser.parse_args()
    config: AttrDict = get_config(args.config_path)
    task = Task.init(project_name="retail/ocr/easyocr", task_name=config["task_name"])
    test(config, task)
