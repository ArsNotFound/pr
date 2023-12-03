from cv2.typing import MatLike
import numpy as np


def one_hot_encode(label: MatLike, label_values: list[list[int]]):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image: MatLike):
    return np.argmax(image, axis=-1)


def color_code_segmentation(image: MatLike, label_values: list[list[int]]):
    color_codes = np.array(label_values)
    return color_codes[image.astype(int)]
