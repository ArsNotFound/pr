import cv2
import imutils
import numpy as np
import torch

from azulvision.board_extraction.config import BOARD_IMAGE_SIZE
from azulvision.board_extraction.one_hot import reverse_one_hot


def ratio(a: float, b: float):
    if a == 0 or b == 0:
        return -1
    return min(a, b) / float(max(a, b))


def fix_mask(mask: np.ndarray, threshold: float = 0.5):
    mask *= 255
    mask = mask.astype(np.uint8)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask


def ignore_contours(img_shape: tuple[int, ...],
                    contours,
                    min_ratio_bounding: float = 0.3,
                    min_area_percentage: float = 0.2,
                    max_area_percentage: float = 1.0):
    ret = []
    mask_area = img_shape[0] * img_shape[1]

    for i in range(len(contours)):
        ca = cv2.contourArea(contours[i])
        ca /= mask_area
        if ca < min_area_percentage or ca > max_area_percentage:
            continue
        _, _, h, w = cv2.boundingRect(contours[i])
        if ratio(h, w) < min_ratio_bounding:
            continue
        ret.append(contours[i])

    return ret


def rotate_quadrangle(approx: np.ndarray):
    if approx[0, 0, 0] < approx[2, 0, 0]:
        approx = approx[[3, 0, 1, 2], :, :]
    return approx


def extract_perspective(image: np.ndarray, approx: np.ndarray, out_size: tuple[int, int]):
    w, h = out_size[0], out_size[1]

    dest = ((0, 0), (w, 0), (w, h), (0, h))

    approx = np.array(approx, np.float32).squeeze()
    dest = np.array(dest, np.float32)

    coeffs = cv2.getPerspectiveTransform(approx, dest)
    return cv2.warpPerspective(image, coeffs, out_size)


def find_quadrangle(mask: np.ndarray) -> np.ndarray | None:
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 1:
        contours = ignore_contours(mask.shape, contours)

    if len(contours) == 0:
        return None

    approx = None

    for i in range(len(contours)):
        cnt = contours[i]

        arclen = cv2.arcLength(cnt, True)
        candidate = cv2.approxPolyDP(cnt, 0.1 * arclen, True)

        if len(candidate) != 4:
            continue

        approx = rotate_quadrangle(candidate)
        break

    return approx


def scale_approx(approx, orig_size):
    sf = orig_size[0] / 256.0
    scaled = np.array(approx * sf, dtype=np.uint32)
    return scaled


@torch.no_grad()
def extract_board(image: np.ndarray, orig: np.ndarray, model, threshold: float = 0.5, device="cpu"):
    image = image.transpose(2, 0, 1).astype(np.float32)
    image_batch = torch.from_numpy(image).to(device).unsqueeze(0)

    predicted_mask_batch = model(image_batch)

    predicted_mask: np.ndarray = predicted_mask_batch.detach().squeeze().cpu().numpy()
    predicted_mask = np.transpose(predicted_mask, (1, 2, 0))
    mask = reverse_one_hot(predicted_mask)
    mask = fix_mask(mask, threshold=threshold)

    approx = find_quadrangle(mask)
    if approx is None:
        print("Contour approximation failed!")
        raise Exception("Failed to find a board.")

    approx = scale_approx(approx, (orig.shape[0], orig.shape[1]))

    board = extract_perspective(orig, approx, BOARD_IMAGE_SIZE)

    return board, predicted_mask
