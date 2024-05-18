import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

COLORS = {
    "Person": (155, 165, 0),
    "Car": (0, 128, 0),
    "Bicycle": (160, 32, 255),
    "OtherVehicle": (32, 178, 170),
    "DontCare": (255, 0, 0),
}

MASK_COLORS = {
    "Person": 1,
    "Car": 2,
    "Bicycle": 3,
    "OtherVehicle": 4,
    "DontCare": 5,
}


def add_altitude_rotation_channel(image: np.ndarray, image_name: str) -> np.ndarray:
    h, w = image.shape[:2]
    altitude, angle = image_name.split("_")[1:3]
    # TODO: normalize the channels!
    altitude_channel = np.full((h, w, 1), int(altitude))
    angle_channel = np.full((h, w, 1), int(angle))
    return np.concatenate(
        (image[:, :, 0][:, :, np.newaxis], altitude_channel, angle_channel), axis=-1
    )


def mask_oriented_annotations(
    image: np.ndarray, annotations: dict, colored_masks: bool = False
) -> np.ndarray:
    """
    Mask oriented bounding boxes on the image.

    :param image: Image to mask the bounding boxes on.
    :param annotations: Annotations for the image.
    :return: Image with masked bounding boxes.
    """
    # zero the image backgournd
    image = np.zeros_like(image)

    for image_ann in annotations["robbox"]:
        cx, cy = image_ann["cx"], image_ann["cy"]
        w, h, angle_rad = image_ann["w"], image_ann["h"], image_ann["angle"]
        category = image_ann["category"]

        if colored_masks:
            color = COLORS[category]
        else:
            color = MASK_COLORS[category]

        half_w = w / 2
        half_h = h / 2

        # Compute the four corners of the rotated rectangle
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

        corners = [(int(x), int(y)) for x, y in corners]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(corners, dtype=np.int32)], 1)
        image[mask == 1] = color

    return image
