import argparse
from pathlib import Path

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

    _, _, raw_b1 = classify_raw(input_files[0], be_model, sc_model, flip=True, threshold=0.5, device=device)
    _, _, raw_b2 = classify_raw(input_files[1], be_model, sc_model, flip=True, threshold=0.5, device=device)

    board1 = Board.from_classifier(raw_b1)
    board2 = Board.from_classifier(raw_b2)

    print("Board 1:")
    print(board1)
    print("\nBoard 2:")
    print(board2)

    score, ended = board2.count_score(board1)

    print(f"\nScore: {score}")
    print(f"Ended: {ended}")


if __name__ == '__main__':
    main()
