import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader
import typing as tp
import segmentation_models_pytorch as smp
import albumentations as A
import glob

from train_old import TrainStep, TrainStepLossTrain, ValStep
from losses import PixelWiseLossWithMeanVector, PixelWiseLossWithVectors
from datasets.ade20k import get_ade20k
from datasets.cocostaff import get_cocostaff
from tester import Tester


if __name__ == '__main__':
    FEATURES_SIZE = 256
    DEVICE = 'cuda'
    AMP = False
    SHAPE = (256, 256)
    BATCH_SIZE = 2
    EPOCHS = 100
    EXP_NAME = 'exp1'
    BACKBONE = 'resnet101'

    RESULT_PATH = os.path.join('results', EXP_NAME)
    RESULT_PATH_MODEL = os.path.join(RESULT_PATH, 'models')
    os.makedirs(RESULT_PATH_MODEL, exist_ok=True)

    RESULT_TESTER = os.path.join(RESULT_PATH, 'tester')
    BEST_MODEL = os.path.join(RESULT_PATH_MODEL, 'best.pth')
    LAST_MODEL = os.path.join(RESULT_PATH_MODEL, 'last.pth')

    geometric_transform = A.OneOf([
            A.Resize(SHAPE[0], SHAPE[1]),
            A.RandomResizedCrop(SHAPE[0], SHAPE[1]),
        ])


    aug_transform = A.Compose([
        A.Flip(),
        A.ShiftScaleRotate(),
        A.OneOf([
            A.HueSaturationValue(),
            A.ColorJitter(),
        ]),
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.MotionBlur(),
    ])

    train_dataset_ade20k, val_dataset_ade20k = get_ade20k(
        train_transforms=A.Compose([geometric_transform, aug_transform]),
        val_transforms=A.Resize(SHAPE[0], SHAPE[1]),
        dataset_path='~/data/ADE20K_2021_17_01',
    )
    # train_dataset_cocostaff, val_dataset_cocostaff = get_cocostaff(
    #     train_transforms=A.Compose([geometric_transform, aug_transform]),
    #     val_transforms=A.Resize(SHAPE[0], SHAPE[1]),
    #     dataset_path='data/coco',
    # )

    train_loader_ade20k = DataLoader(train_dataset_ade20k, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader_ade20k = DataLoader(val_dataset_ade20k, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # train_loader_cocostaff = DataLoader(train_dataset_cocostaff, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader_cocostaff = DataLoader(val_dataset_cocostaff, batch_size=BATCH_SIZE, shuffle=True)

    # loss = PixelWiseLossWithMeanVector()
    loss = PixelWiseLossWithVectors(3688, FEATURES_SIZE, is_full=False, ignore_classes=[0]).to(DEVICE)
    loss.load_state_dict(torch.load(LAST_MODEL.replace('.pth', '_loss.pth')))

    loss_optim = torch.optim.Adam(loss.parameters(), eps=1e-4)

    model = smp.FPN(
        BACKBONE,
        classes=FEATURES_SIZE,
        activation=None,
        decoder_segmentation_channels=FEATURES_SIZE*2,
        decoder_pyramid_channels=FEATURES_SIZE*2,
    )
    # model = smp.Unet(
    #     BACKBONE,
    #     classes=FEATURES_SIZE,
    #     activation=None,
    #     decoder_channels=[FEATURES_SIZE*6, FEATURES_SIZE*5, FEATURES_SIZE*4, FEATURES_SIZE*3, FEATURES_SIZE*2],
    # )
    # model.load_state_dict(torch.load(LAST_MODEL))
    optim = torch.optim.Adam(model.parameters(), eps=1e-4)
    train_step = TrainStepLossTrain(
        model=model,
        optim=optim,
        loss=loss,
        loss_optim=loss_optim,
        device=DEVICE,
        amp=AMP,
    )
    val_step = ValStep(
        model=model,
        loss=loss,
        device=DEVICE,
        amp=AMP,
    )
    os.makedirs(RESULT_PATH, exist_ok=True)
    last_metric = None

    tester_cars = Tester(
        model,
        images_paths=sorted(glob.glob('data/test_images/cars/*')),
        x=0.2, y=0.4,
        target_b=0,
        save_folder=os.path.join(RESULT_TESTER, 'cars'),
        transforms=A.Resize(SHAPE[0], SHAPE[1]),
        threshold=0.9,
        gif_duration=500,
        run_every=100,
        device=DEVICE,
    )

    tester_cats = Tester(
        model,
        images_paths=sorted(glob.glob('data/test_images/cats/*')),
        x=0.4, y=0.6,
        target_b=1,
        save_folder=os.path.join(RESULT_TESTER, 'cats'),
        transforms=A.Resize(SHAPE[0], SHAPE[1]),
        threshold=0.9,
        gif_duration=500,
        run_every=100,
        device=DEVICE,
    )

    tester_fish = Tester(
        model,
        images_paths=sorted(glob.glob('data/test_images/fish/*')),
        x=0.7, y=0.7,
        target_b=0,
        save_folder=os.path.join(RESULT_TESTER, 'fish'),
        transforms=A.Resize(SHAPE[0], SHAPE[1]),
        threshold=0.9,
        gif_duration=500,
        run_every=100,
        device=DEVICE,
    )

    model.to(DEVICE)
    loss.to(DEVICE)

    for epoch in range(EPOCHS):

        train_logs_ade20k = train_step.run(train_loader_ade20k, [tester_cars.test, tester_cats.test, tester_fish.test])
        # train_logs_cocostaff = train_step.run(train_loader_cocostaff, [tester_cars.test, tester_cats.test, tester_fish.test])

        val_logs_ade20k = val_step.run(val_loader_ade20k)
        # val_logs_cocostaff = val_step.run(train_loader_cocostaff)

        cur_metric = val_logs_ade20k['loss']

        torch.save(model.state_dict(), LAST_MODEL)
        torch.save(loss.state_dict(), LAST_MODEL.replace('.pth', '_loss.pth'))
        if last_metric is not None:
            if cur_metric < last_metric:
                torch.save(model.state_dict(), BEST_MODEL)
                torch.save(loss.state_dict(), BEST_MODEL.replace('.pth', '_loss.pth'))
        else:
            last_metric = cur_metric

        print('train_logs_ade20k =', train_logs_ade20k)
        # print('train_logs_cocostaff =', train_logs_cocostaff)
        print('val_logs_ade20k =', val_logs_ade20k)
        # print('val_logs_cocostaff =', val_logs_cocostaff)

