from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIN_MEMORY = True if DEVICE == "cuda" else False

DATA_DIR = Path("data")
