import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.process_image import add_altitude_rotation_channel, add_altitude_channel, add_rotation_channel


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


def annotations_to_yolo(img_name: str, filepath: Path, annotations: dict, class_dict: dict) -> None:
    image = annotations[img_name]
    w_img = image["size"]["width"]
    h_img = image["size"]["height"]
    with filepath.open("a") as f:
        for image_ann in image["robbox"]:
            cx, cy = (
                image_ann["cx"],
                image_ann["cy"],
            )
            w, h, angle_rad = image_ann["w"], image_ann["h"], image_ann["angle"]
            category = image_ann["category"]
            class_id = class_dict[category]
            half_w = w / 2
            half_h = h / 2

            corners = [
                (
                    cx
                    + (math.cos(angle_rad) * half_w + math.sin(angle_rad) * half_h),
                    cy
                    + (math.sin(angle_rad) * half_w - math.cos(angle_rad) * half_h),
                ),
                (
                    cx
                    + (math.cos(angle_rad) * half_w - math.sin(angle_rad) * half_h),
                    cy
                    + (math.sin(angle_rad) * half_w + math.cos(angle_rad) * half_h),
                ),
                (
                    cx
                    - (math.cos(angle_rad) * half_w + math.sin(angle_rad) * half_h),
                    cy
                    - (math.sin(angle_rad) * half_w - math.cos(angle_rad) * half_h),
                ),
                (
                    cx
                    - (math.cos(angle_rad) * half_w - math.sin(angle_rad) * half_h),
                    cy
                    - (math.sin(angle_rad) * half_w + math.cos(angle_rad) * half_h),
                ),
            ]
            corners = [[int(x) / w_img, int(y) / h_img] for x, y in corners]
            flat_corners = [x for xs in corners for x in xs]
            object_str = f"{class_id}, {*flat_corners,}\n"
            object_str = object_str.replace("(", "").replace(")", "").replace(",", "")
            f.write(object_str)


def save_alt_rot_images_annotations(
    img_dir: Path,
    ann_dir: Path,
    image_names: dict,
    images_path: str,
    annotations: dict,
    class_dict: dict,
    transform: str,
) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for img_set in image_names.keys():
        set_dir = img_dir / img_set
        set_dir.mkdir(parents=True, exist_ok=True)
        set_dir_ann = ann_dir / img_set
        set_dir_ann.mkdir(parents=True, exist_ok=True)
        for img_name in image_names[img_set]:
            image = cv2.imread(f"{images_path}/{img_name}.jpg")
            if transform == "rotation":
                image = add_rotation_channel(image, img_name)
            elif transform == "altitude":
                image = add_altitude_channel(image, img_name)
            elif transform == "both":
                image = add_altitude_rotation_channel(image, img_name)
            filepath = set_dir / f"{img_name}.jpg"
            cv2.imwrite(str(filepath), image)
            filepath_ann = set_dir_ann / f"{img_name}.txt"
            annotations_to_yolo(img_name, filepath_ann, annotations, class_dict)
