import numpy as np


def add_altitude_rotation_channel(image: np.ndarray, image_name: str) -> np.ndarray:
    h, w = image.shape[:2]
    altitude, angle = image_name.split("_")[1:3]
    # TODO: normalize the channels!
    altitude_channel = np.full((h, w, 1), int(altitude))
    angle_channel = np.full((h, w, 1), int(angle))
    return np.concatenate(
        (image[:, :, 0][:, :, np.newaxis], altitude_channel, angle_channel), axis=-1
    )
