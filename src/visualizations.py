import glob
import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.utils.process_image import mask_oriented_annotations

COLORS = {
    "Person": (155, 165, 0),
    "Car": (0, 128, 0),
    "Bicycle": (160, 32, 255),
    "OtherVehicle": (32, 178, 170),
    "DontCare": (255, 0, 0),
}


def yolo2bbox(bboxes: list[int]) -> tuple[int]:
    """
    Helper function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax
    """
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax


def plot_box(
    image: np.ndarray,
    bboxes: list[list],
    labels: list[str],
    classes: list[str],
    colors: list[int],
) -> None:
    height, width, _ = image.shape
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        xmin = int(x1 * width)
        ymin = int(y1 * height)
        xmax = int(x2 * width)
        ymax = int(y2 * height)
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

        class_name = classes[int(labels[box_num])]
        color = colors[classes.index(class_name)]
        cv2.rectangle(image, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)
        w, h = cv2.getTextSize(class_name, 0, fontScale=lw / 3, thickness=tf)[0]

        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color=color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            class_name,
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3.5,
            color=(255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_images(
    image_path: str,
    label_path: str,
    num_samples: int,
    classes: list[str],
    colors: list[int],
) -> None:
    all_training_images = glob.glob(image_path + "/*")
    all_training_labels = glob.glob(label_path + "/*")
    all_training_images.sort()
    all_training_labels.sort()

    temp = list(zip(all_training_images, all_training_labels))
    random.shuffle(temp)
    all_training_images, all_training_labels = zip(*temp)
    all_training_images, all_training_labels = list(all_training_images), list(
        all_training_labels
    )

    num_cols = 2
    num_rows = int(math.ceil(num_samples / num_cols))
    plt.figure(figsize=(5 * num_cols, 3 * num_rows))
    for i in range(num_samples):
        image = cv2.imread(all_training_images[i])
        with open(all_training_labels[i], "r") as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label, x_c, y_c, w, h = label_line.split()
                bboxes.append([float(x_c), float(y_c), float(w), float(h)])
                labels.append(label)
        plot_box(image, bboxes, labels, classes, colors)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image[:, :, ::-1])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_class_frequencies(df: pd.DataFrame) -> None:
    classes_df = (
        df[["class_name", "annotations"]]
        .groupby("class_name")
        .count()
        .reset_index()
        .sort_values(by="annotations", ascending=False)
    )

    plt.figure(figsize=(6, 4))
    bars = plt.bar(classes_df["class_name"], classes_df["annotations"], color="purple")
    plt.title("Frequency of Each Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.5, yval, int(yval), va="bottom")

    plt.show()


def plot_class_frequencies_split(df: pd.DataFrame, classes: list[str]) -> None:
    classes_df = (
        df[["class_name", "annotations", "split"]]
        .groupby(["class_name", "split"])
        .size()
    )
    fig, ax = plt.subplots(layout="constrained")
    splits = ["train", "test", "val"]
    x = np.arange(len(classes))
    width = 0.25
    multiplier = 0

    for split in splits:
        vals = []
        for cls in classes:
            vals.append(classes_df[cls][split])
        offset = width * multiplier
        rects = ax.bar(x + offset, vals, width, label=split)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_xticks(x + width, classes)
    ax.set_title("Number of instances per class")
    ax.legend()
    plt.show()


def plot_instances(df: pd.DataFrame) -> None:
    df_filtered = (
        df[["url", "split"]]
        .pivot_table(index="url", columns=["split"], aggfunc=lambda l: int(len(l)))
        .reset_index()
        .melt("url")
    )
    df_filtered = df_filtered[~df_filtered["value"].isna()]
    plt.figure(figsize=(10, 4))
    plt.hist(df_filtered["value"], bins=20)

    plt.xlabel("Number of instances")
    plt.ylabel("Count")
    plt.title("Number of instances per image")
    plt.show()


def plot_oriented_annotations(
    image: np.ndarray, annotations: dict, show: bool = False
) -> np.ndarray:
    """
    Plot oriented bounding boxes on the image.

    :param image: Image to plot the bounding boxes on.
    :param annotations: Annotations for the image.
    :param show: Whether to show the image.
    :return: Image with bounding boxes.
    """

    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)

    for image_ann in annotations["robbox"]:

        cx, cy = (
            image_ann["cx"],
            image_ann["cy"],
        )
        w, h, angle_rad = image_ann["w"], image_ann["h"], image_ann["angle"]
        category = image_ann["category"]
        color = COLORS[category]
        half_w = w / 2
        half_h = h / 2

        corners = [
            (
                cx + (math.cos(angle_rad) * half_w + math.sin(angle_rad) * half_h),
                cy + (math.sin(angle_rad) * half_w - math.cos(angle_rad) * half_h),
            ),
            (
                cx + (math.cos(angle_rad) * half_w - math.sin(angle_rad) * half_h),
                cy + (math.sin(angle_rad) * half_w + math.cos(angle_rad) * half_h),
            ),
            (
                cx - (math.cos(angle_rad) * half_w + math.sin(angle_rad) * half_h),
                cy - (math.sin(angle_rad) * half_w - math.cos(angle_rad) * half_h),
            ),
            (
                cx - (math.cos(angle_rad) * half_w - math.sin(angle_rad) * half_h),
                cy - (math.sin(angle_rad) * half_w + math.cos(angle_rad) * half_h),
            ),
        ]

        xmin, xmax = min([corner[0] for corner in corners]), max(
            [corner[0] for corner in corners]
        )
        ymin, ymax = min([corner[1] for corner in corners]), max(
            [corner[1] for corner in corners]
        )
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

        corners = [(int(x), int(y)) for x, y in corners]
        cv2.polylines(
            image,
            [np.array(corners, dtype=np.int32)],
            isClosed=True,
            color=color,
            thickness=2,
        )

        w, h = cv2.getTextSize(category, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color=color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            category,
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3.5,
            color=(255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    if show:
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(image)
        plt.show()
    return image


def plot_rotated_images(
    images_path: str,
    images_list: list[str],
    annotations: dict,
    rows: int = None,
    cols: int = None,
    figsize: tuple[int, int] = (15, 20),
) -> None:
    if rows is None:
        rows = len(images_list)
        columns = 1
    elif cols is None:
        cols = len(images_list)
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    if len(images_list) > rows * cols:
        raise ValueError("Number of images to display exceeds the number of subplots")

    for idx, image_name in enumerate(images_list):
        img_path = f"{images_path}/{image_name}.jpg"
        img = cv2.imread(img_path)

        img_with_annotations = plot_oriented_annotations(img, annotations[image_name])
        axes[idx].imshow(cv2.cvtColor(img_with_annotations, cv2.COLOR_BGR2RGB))
        axes[idx].set_axis_off()

    for ax in axes[len(images_list) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def mask_rotated_images(
    images_path: str,
    images_list: list[str],
    annotations: dict,
    colored_masks: bool = False,
    rows: int = None,
    cols: int = None,
    figsize: tuple[int, int] = (15, 20),
) -> None:
    if rows is None:
        rows = len(images_list)
        cols = 1
    elif cols is None:
        cols = len(images_list)
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    if len(images_list) > rows * cols:
        raise ValueError("Number of images to display exceeds the number of subplots")

    for idx, image_name in enumerate(images_list):
        img_path = f"{images_path}/{image_name}.jpg"
        img = cv2.imread(img_path)

        img_with_annotations = mask_oriented_annotations(
            img, annotations[image_name], colored_masks=colored_masks
        )
        axes[idx].imshow(cv2.cvtColor(img_with_annotations, cv2.COLOR_BGR2RGB))
        axes[idx].set_axis_off()

    for ax in axes[len(images_list) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_model_input_output(img_path: str, output: torch.Tensor) -> None:

    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(img_rgb)
    ax[0].set_title("Input image")
    ax[0].axis("off")

    masked_img = np.argmax(output.detach().numpy(), axis=1).squeeze(0)
    print(masked_img.shape)

    colors = np.array(
        [
            (0, 0, 0),  # Background
            (155, 165, 0),  # Person
            (0, 128, 0),  # Car
            (160, 32, 255),  # Bicycle
            (32, 178, 170),  # OtherVehicle
            (255, 0, 0),  # DontCare
        ]
    )

    color_mask = colors[masked_img]

    ax[1].imshow(color_mask)
    ax[1].set_title("Segmentation output")
    ax[1].axis("off")
    plt.show()
