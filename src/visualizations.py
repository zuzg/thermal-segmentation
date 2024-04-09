import glob
import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes_df["class_name"], classes_df["annotations"], color="purple")
    plt.title("Frequency of Each Class")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.5, yval, int(yval), va="bottom")

    plt.show()
