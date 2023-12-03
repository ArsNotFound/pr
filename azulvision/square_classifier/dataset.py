import os
from pathlib import Path

import albumentations as album
import cv2
import numpy as np
from torch.utils.data import Dataset

from azulvision.square_classifier.config import LABEL_ENCODER


class SCDataset(Dataset):
    def __init__(self,
                 base_dir: str | os.PathLike[str],
                 class_dict: dict[str, str],
                 augmentation: album.Compose | None = None,
                 preprocessing: album.Compose | None = None) -> None:
        super().__init__()

        self.base_dir = Path(base_dir)
        self.class_dirs: dict[str, Path] = {
            cls_name: self.base_dir / cls_dir
            for cls_name, cls_dir in class_dict.items()
        }
        self.images: list[tuple[str, Path]] = [
            (cls_name, path)
            for cls_name, cls_dir in self.class_dirs.items()
            for path in cls_dir.glob('*.jpg')
        ]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        cls: np.ndarray = LABEL_ENCODER.transform([self.images[idx][0]])
        image = cv2.cvtColor(cv2.imread(str(self.images[idx][1])), cv2.COLOR_BGR2RGB)

        if self.augmentation:
            image = self.augmentation(image=image)['image']

        if self.preprocessing:
            image = self.preprocessing(image=image)['image']

        return image, cls

    @staticmethod
    def decode_cls(cls: np.ndarray) -> str:
        return LABEL_ENCODER.inverse_transform(cls)
