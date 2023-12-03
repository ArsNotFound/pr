import argparse
from pathlib import Path

import cv2
import numpy as np


def extract_squares(board: np.ndarray):
    squares = []
    ww, hh = board.shape[0], board.shape[1]
    w = int(ww / 5)
    h = int(hh / 5)

    for i in range(5):
        for j in range(5):
            squares.append(board[i * w:(i + 1) * w, j * h:(j + 1) * h])

    squares = np.array(squares)
    return squares


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Input dir", type=Path)
    parser.add_argument("output_dir", help="Output dir", type=Path)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Input directory not exists: {args.input_dir}")
        exit(1)

    if output_dir.exists():
        print(f"Output directory already exists: {args.output_dir}")
        exit(1)

    output_dir.mkdir(parents=True)

    for f in input_dir.glob('*.jpg'):
        print(f"Processing board: {f.name}")
        board = cv2.imread(str(f))
        squares = extract_squares(board)
        for (i, s) in enumerate(squares):
            cv2.imwrite(str(output_dir / f"{f.stem}-{i}{f.suffix}"), s)


if __name__ == "__main__":
    main()
