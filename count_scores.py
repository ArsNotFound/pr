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
    parser.add_argument("input_files", nargs=2, help="Input images", type=Path)
    args = parser.parse_args()

    input_files: list[Path] = args.input_files

    be_model, sc_model = load_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    board_img1, _, raw_b1 = classify_raw(input_files[0], be_model, sc_model, flip=True, threshold=0.5, device=device)
    board_img2, _, raw_b2 = classify_raw(input_files[1], be_model, sc_model, flip=True, threshold=0.5, device=device)

    board1 = Board.from_classifier(raw_b1)
    board2 = Board.from_classifier(raw_b2)

    board_img1 = cv2.cvtColor(board_img1, cv2.COLOR_RGB2BGR)
    board_img2 = cv2.cvtColor(board_img2, cv2.COLOR_RGB2BGR)

    print("Board 1:")
    print(board1)
    print("\nBoard 2:")
    print(board2)

    score, ended = board2.count_score(board1)

    print(f"\nScore: {score}")
    print(f"Ended: {ended}")

    cv2.imshow("Board 1", board_img1)
    cv2.imshow("Board 2", board_img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
