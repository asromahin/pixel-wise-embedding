import cv2
from torchvision.transforms import ToTensor
import torch
import glob
import os


from datasets.ade20k_utils import loadAde20K


class ADE20KDataset:
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
        mask = res['class_mask']

        a = self.transforms(image=image, mask=mask)
        image = a['image']
        mask = a['mask']

        image = ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


def get_ade20k(train_transforms, val_transforms, dataset_path='data/ADE20K_2021_17_01'):
    train_images = glob.glob(os.path.join(dataset_path, 'images/ADE/training/**/**/*.jpg'))
    val_images = glob.glob(os.path.join(dataset_path, 'images/ADE/validation/**/**/*.jpg'))
    train_dataset = ADE20KDataset(train_images, transforms=train_transforms)
    val_dataset = ADE20KDataset(val_images, transforms=val_transforms)
    return train_dataset, val_dataset



