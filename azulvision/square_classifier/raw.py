import logging
import shutil

import cv2
import numpy as np
import torch

import azulvision.square_classifier.config as sc_config

logger = logging.getLogger("azulvision.square_classifier")
logger.setLevel(logging.DEBUG)


def load_models():
    logger.debug("Loading models...")
    if not sc_config.MODEL_WEIGHTS.is_file():
        raise Exception("Square classifier file not found")
    square_classifier = torch.load(sc_config.MODEL_WEIGHTS)

    logger.debug("Models loaded")
    return square_classifier


@torch.no_grad()
def classify_square(idx: str, img: np.ndarray, square_model, device: str = "cpu"):
    logger.debug(f"Processing image: {idx}")

    if not square_model:
        square_model = load_models()

    square_model.eval()

    x = img.transpose((2, 0, 1)).astype(np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

    out = square_model(x_tensor)

    _, preds = torch.max(out, 1)
    preds = preds.detach().cpu().numpy()

    return preds


if __name__ == '__main__':
    if not sc_config.BASE_OUTPUT.exists():
        sc_config.BASE_OUTPUT.mkdir(parents=True)

    path = sc_config.SC_DATA_DIR / "!raw"

    paths = path.rglob('*.jpg')

    square_model = load_models()

    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        try:
            cat = classify_square(p.name, img, square_model, device=sc_config.DEVICE)
            cat = sc_config.LABEL_ENCODER.inverse_transform(cat)
        except Exception as e:
            logger.warning(f"Skipping {p.name} image: {e}")
            continue

        cat_dir = sc_config.BASE_OUTPUT / cat[0]
        cat_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, cat_dir / p.name)
