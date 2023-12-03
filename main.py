import argparse
from pathlib import Path

import cv2
import torch

from azulvision.board import Board
from azulvision.classify import classify_raw, load_models


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

    board_img, mask, raw_board = classify_raw(input_file, be_model, sc_model, flip=not no_flip, threshold=0.5,
                                              device=device)

    board = Board.from_classifier(raw_board)
    print(board)

    if output_file:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), board_img)


if __name__ == '__main__':
    main()
