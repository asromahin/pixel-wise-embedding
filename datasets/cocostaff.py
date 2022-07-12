import random
import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor
import glob
import os


class CocoStaffDataset:
    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        a = self.transforms(image=image, mask=mask)
        image = a['image']
        mask = a['mask']

        if mask.max() == 0:
            return self[idx]

        image = ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_cocostaff(train_transforms, val_transforms, dataset_path):
    train_dataset_cocostaff = CocoStaffDataset(
        images=sorted(glob.glob(os.path.join(dataset_path, 'train2017/*.jpg'))),
        masks=sorted(glob.glob(os.path.join(dataset_path, 'train2017/*.png'))),
        transforms=train_transforms,
    )
    val_dataset_cocostaff = CocoStaffDataset(
        images=sorted(glob.glob(os.path.join(dataset_path, 'val2017/*.jpg'))),
        masks=sorted(glob.glob(os.path.join(dataset_path, 'val2017/*.png'))),
        transforms=train_transforms,
    )
    return train_dataset_cocostaff, val_dataset_cocostaff







