import itertools

import numpy as np
import torch

from . import config as sc_config


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


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


@torch.no_grad()
def classify_board(board_img, model, device: str = "cpu"):
    squares = extract_squares(board_img)

    board = []

    model.eval()
    for c in squares:
        x = c.transpose((2, 0, 1)).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

        out = model(x_tensor)

        _, preds = torch.max(out, 1)
        preds = preds.detach().cpu().numpy()

        cat = sc_config.LABEL_ENCODER.inverse_transform(preds)

        board.append(cat[0])

    return list(batched(board, 5))
