import time
import easyocr
import easyocr.model
import pandas as pd
from pathlib import Path
from config import EasyOCRConfig
from typing import Tuple


class EasyOCRModel :
    def __init__(self, config: EasyOCRConfig) -> None:
        self.__model = easyocr.Reader(
            lang_list=["ru"],
            model_storage_directory=config.path_to_model_dir,
            user_network_directory="./trained_model/user_network",
            gpu=config.device == "cuda",
            recog_network="trained_model"
        )
        self.__dataset_dir=config.path_to_dataset
    
    def test(self,df:pd.DataFrame)->pd.DataFrame:
        result = {"filename":[],"words":[],"preds":[]}
        for _,row in df.iterrows():
            result["filename"].append(row["filename"])
            result["words"].append(row["words"])
            result["preds"].append(self.__inference(self.__dataset_dir/row["filename"]))
        return pd.DataFrame(result)


    def __inference(self, path_to_photo: Path) -> Tuple[str, float]:
        start_time = time.time()
        result = self.__model.readtext(str(path_to_photo),detail=1)
        res_dic = {}
        print(result)
        for res in result:
            coords = res[0]
            text = res[1]
            text=text[: text.find("[GO]")]
            
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