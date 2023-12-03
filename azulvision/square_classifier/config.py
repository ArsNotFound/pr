import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from azulvision.config import *

SC_DATA_DIR = DATA_DIR / "square_classifier"

class_dict = pd.read_csv(SC_DATA_DIR / "class_names.csv")
CLASS_NAMES = class_dict["name"].tolist()
CLASS_DIRS = pd.Series(class_dict["folder"].values, index=class_dict["name"]).to_dict()

LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.fit(np.array(list(sorted(CLASS_NAMES))))

MODEL_WEIGHTS = SC_DATA_DIR / "best_model_sc.pth"

NUM_CHANNELS = 3
NUM_CLASSES = len(CLASS_NAMES)

INIT_LR = 1e-3
NUM_EPOCHS = 100
NUM_WORKERS = 8
BATCH_SIZE = 8

BASE_OUTPUT = SC_DATA_DIR / "output"
