import logging

import cv2
import numpy as np
import torch

from azulvision.board_extraction import config as be_config
from azulvision.board_extraction.extract import extract_board

logger = logging.getLogger("azulvision.board_extraction")
logger.setLevel(logging.DEBUG)


def load_models():
    logger.debug("Loading models...")
    if not be_config.MODEL_WEIGHTS.is_file():
        raise Exception("Board model file not found")
    board_extractor = torch.load(be_config.MODEL_WEIGHTS)

    logger.debug("Models loaded")
    return board_extractor


def classify_board(idx: str, img: np.ndarray, board_model=None, threshold: float = 0.5, device: str = "cpu"):
    logger.debug(f"Processing image {idx}")

    if not board_model:
        board_model = load_models()

    comp_img = cv2.resize(img, be_config.INPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype('float32')

    logger.debug("Extracting board from image")
    try:
        board_img, mask = extract_board(comp_img, img, board_model, threshold, device)
    except Exception as e:
        raise e

    return board_img, mask


if __name__ == '__main__':
    if not be_config.BASE_OUTPUT.exists():
        be_config.BASE_OUTPUT.mkdir(parents=True)

    paths = []
    paths.extend(be_config.X_TRAIN_DIR.glob('*.jpg'))
    paths.extend(be_config.X_VALID_DIR.glob('*.jpg'))
    paths.extend(be_config.X_TEST_DIR.glob('*.jpg'))

    board_model = load_models()

    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        try:
            board, mask = classify_board(p.name, img, board_model, device=be_config.DEVICE)
        except Exception as e:
            logger.warning(f"Skipping {p.name} image: {e}")
            continue
        board = cv2.cvtColor(board, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(be_config.BASE_OUTPUT / p.name), board)
