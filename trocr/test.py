from pathlib import Path
from argparse import ArgumentParser
from models import (
    TrOCRModel,
    trocr_conf_gen,
)
from utils.datasets import txt_dataset_to_df
from utils.configs import YamlConfig


def test_trocr(yaml_config: YamlConfig):
    for transformer_config in trocr_conf_gen(yaml_config):
        test_df = txt_dataset_to_df(transformer_config.txt_path)
        model = TrOCRModel(transformer_config)
        model.test(test_df)


def main(yaml_config: YamlConfig) -> None:
    test_trocr(yaml_config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    yaml_config = YamlConfig(args.path_to_yaml_config)
    main(yaml_config)
