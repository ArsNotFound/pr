import os

import albumentations as album
import cv2
from torch.utils.data import Dataset

from .one_hot import one_hot_encode


class BEDataset(Dataset):
    def __init__(self,
                 images_dir: str | os.PathLike[str],
                 masks_dir: str | os.PathLike[str],
                 class_rgb_values: list[list[int]] | None = None,
                 augmentation: album.Compose | None = None,
                 preprocessing: album.Compose | None = None):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, mask_id) for mask_id in sorted(os.listdir(masks_dir))]

        assert len(self.image_paths) == len(self.mask_paths), (f"Expected {len(self.image_paths)} masks. "
                                                               f"Got {len(self.mask_paths)}")

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx: int):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)

        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return min(len(self.image_paths), len(self.mask_paths))
