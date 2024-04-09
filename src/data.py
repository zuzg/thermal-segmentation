from pathlib import Path

import numpy as np
import pandas as pd


def read_txt(txt_url: str) -> np.ndarray:
    result = pd.read_csv(
        txt_url, sep=" ", names=["label", "x_center", "y_center", "width", "height"]
    ).values
    return result


def read_yolo_dataset(data_dir: str, classes: list[str]) -> pd.DataFrame:
    data_dir = Path(data_dir)
    labels_list = [str(p) for p in data_dir.rglob("labels/*/*.txt")]
    df = pd.DataFrame(columns=["url"], data=labels_list)

    df["annotations"] = df["url"].apply(lambda x: read_txt(x))
    df = df[~df["annotations"].isna()]
    df = df.explode("annotations")
    df = df.reset_index(drop=True)

    df["class_id"] = df["annotations"].apply(lambda x: x[0])
    df["x_center"] = df["annotations"].apply(lambda x: x[1])
    df["y_center"] = df["annotations"].apply(lambda x: x[2])
    df["width"] = df["annotations"].apply(lambda x: x[3])
    df["height"] = df["annotations"].apply(lambda x: x[4])
    df["class_name"] = df["class_id"].apply(lambda x: classes[int(x)])
    df["split"] = df.apply(lambda x: Path(x["url"]).parent.name, axis=1)
    return df
