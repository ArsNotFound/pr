import argparse
import logging
from pathlib import Path
from pprint import pprint

import cv2
import torch

from azulvision.board_extraction import config as be_config
from azulvision.board_extraction.extract import extract_board
from azulvision.square_classifier import config as sc_config
from azulvision.square_classifier.classifier import classify_board

logger = logging.getLogger("azulvision")
logger.setLevel(logging.DEBUG)


def load_models():
    logger.debug("Loading models...")
    if not be_config.MODEL_WEIGHTS.is_file():
        raise Exception("Board extractor weights not found!")
    if not sc_config.MODEL_WEIGHTS.is_file():
        raise Exception("Square classifier weights not found!")

    be = torch.load(be_config.MODEL_WEIGHTS)
    sc = torch.load(sc_config.MODEL_WEIGHTS)
    logger.debug("Models loaded")

    return be, sc


def classify_raw(file: Path, board_model=None, square_model=None, flip: bool = True, threshold=0.5,
                 device: str = "cpu"):
    logger.debug(f"Processing image: {file}")

    if not board_model or not square_model:
        board_model, square_model = load_models()

    img = cv2.imread(str(file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    comp_image = cv2.resize(img, be_config.INPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    logger.debug("Extracting board from image")
    try:
        board_img, mask = extract_board(comp_image, img, board_model, threshold=threshold, device=device)
    except Exception as e:
        logger.error(f"Failed to extract board from image: {e}")
        raise e

    if flip:
        board_img = cv2.flip(board_img, 1)

    logger.debug("Classifying squares")
    board = classify_board(board_img, square_model, device=device)

    logger.debug(f"Processing image {file} done.")

    return board_img, mask, board


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("i", help="Input image", type=Path)
    parser.add_argument("-o", help="Board output image", type=Path)
    parser.add_argument("--no-flip", help="Don't flip extracted image", type=bool)

    args = parser.parse_args()

    input_file: Path = args.i
    output_file: Path | None = args.o
    no_flip: bool = args.no_flip

    be_model, sc_model = load_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    board_img, mask, board = classify_raw(input_file, be_model, sc_model, flip=not no_flip, threshold=0.5,
                                          device=device)

    pprint(board)

    if output_file:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), board_img)


if __name__ == '__main__':
    main()
