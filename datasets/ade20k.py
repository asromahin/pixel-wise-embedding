import cv2
from torchvision.transforms import ToTensor
import torch
import glob
import os
import numpy as np


from datasets.ade20k_utils import loadAde20K
from datasets.ade20k_classes import ade20k_class_to_code


class ADE20KDataset:
    def __init__(self, images, transforms, mask_key='class_mask'):
        self.images = images
        self.transforms = transforms
        self.mask_key =  mask_key

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = cv2.imread(image_name)
        try:
            res = loadAde20K(image_name)
        except:
            return self[idx + 1]
        mask = res[self.mask_key]

        a = self.transforms(image=image, mask=mask)
        image = a['image']
        mask = a['mask']

        image = ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


class ADE20KObjectsDataset:
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image = cv2.imread(image_name)
        try:
            res = loadAde20K(image_name)
        except:
            return self[idx + 1]
        data = res['objects']

        mask = np.zeros((image.shape[0], image.shape[1]))

        for i in range(len(data['class'])):
            cur_class = data['class'][i]
            cur_class_idx = ade20k_class_to_code[cur_class]
            cur_poly = data['polygon'][i]
            x = cur_poly['x']
            y = cur_poly['y']
            p = np.stack([x, y], axis=-1)
            p = np.expand_dims(p, 1)
            cv2.drawContours(mask, [p], -1, cur_class_idx, -1)

        a = self.transforms(image=image, mask=mask)
        image = a['image']
        mask = a['mask']

        image = ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


def get_ade20k(train_transforms, val_transforms, dataset_path='data/ADE20K_2021_17_01', mask_key='class_mask'):
    train_images = glob.glob(os.path.join(dataset_path, 'images/ADE/training/**/**/*.jpg'))
    val_images = glob.glob(os.path.join(dataset_path, 'images/ADE/validation/**/**/*.jpg'))
    train_dataset = ADE20KDataset(train_images, transforms=train_transforms, mask_key=mask_key)
    val_dataset = ADE20KDataset(val_images, transforms=val_transforms, mask_key=mask_key)
    return train_dataset, val_dataset



