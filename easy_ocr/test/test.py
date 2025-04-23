from pathlib import Path
from argparse import ArgumentParser
from model import EasyOCRModel
from config import EasyOCRConfig
from sklearn.metrics import accuracy_score
from prepare_dataset import create_df_from_txt
def test_easy_ocr(config: EasyOCRConfig):
    test_df = create_df_from_txt(config.path_to_dataset,config.metadata_filename)
    model = EasyOCRModel(config)
    df = model.test(test_df)
    a = df["words"]
    b=df["preds"]
    #f"{accuracy_score(a,b)
    config.result_dir.mkdir(exist_ok=True,parents=True)
    df.to_csv(config.result_dir/f"test.csv",index=False)
    


def main(config: EasyOCRConfig) -> None:
    test_easy_ocr(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_to_yaml_config", "-config", required=True, type=Path)
    args = parser.parse_args()
    config = EasyOCRConfig(args.path_to_yaml_config)
    main(config = EasyOCRConfig(args.path_to_yaml_config)
)
