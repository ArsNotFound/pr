import pandas as pd

from azulvision.config import *

BE_DATA_DIR = DATA_DIR / "board_extraction"

X_TRAIN_DIR = BE_DATA_DIR / "train" / "imgs"
Y_TRAIN_DIR = BE_DATA_DIR / "train" / "masks"

X_VALID_DIR = BE_DATA_DIR / "valid" / "imgs"
Y_VALID_DIR = BE_DATA_DIR / "valid" / "masks"

X_TEST_DIR = BE_DATA_DIR / "test" / "imgs"
Y_TEST_DIR = BE_DATA_DIR / "test" / "masks"

MODEL_WEIGHTS = BE_DATA_DIR / "best_model_be.pth"

class_dict = pd.read_csv(BE_DATA_DIR / "label_class_dict.csv")
CLASS_NAMES = class_dict["name"].tolist()
CLASS_RGB_VALUES = class_dict[['r', 'g', 'b']].values.tolist()

NUM_CHANNELS = 3
NUM_CLASSES = len(CLASS_NAMES)
NUM_LEVELS = 5

INIT_LR = 1e-3
NUM_EPOCHS = 100
NUM_WORKERS = 8
BATCH_SIZE = 6

INPUT_IMAGE_SIZE = (256, 256)
BOARD_IMAGE_SIZE = (500, 500)

THRESHOLD = 0.5

BASE_OUTPUT = BE_DATA_DIR / "output"
