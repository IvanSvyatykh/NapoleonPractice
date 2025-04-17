from pathlib import Path
from argparse import ArgumentParser
from models import (
    EasyOCRModel,
    easyocr_conf_gen,
)
from utils.datasets import txt_dataset_to_df
from utils.configs import YamlConfig


def test_easy_ocr(yaml_config: YamlConfig):
    for easyocr_config in easyocr_conf_gen(yaml_config):
        test_df = txt_dataset_to_df(easyocr_config.txt_path)
        model = EasyOCRModel(easyocr_config)
        model.test(test_df)


def main(yaml_config: YamlConfig) -> None:
    test_easy_ocr(yaml_config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = YamlConfig(args.path_to_yaml_config)
    main(yaml_config)
