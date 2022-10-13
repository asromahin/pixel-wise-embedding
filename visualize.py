import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import glob
import albumentations as A
import numpy as np
import argparse


from tester import Tester, DBSCANTester


class Visualizer:
    def __init__(self, model, images_paths, image_size=256, threshold=0.9, device='cuda'):
        self.image_size = image_size
        self.tester = Tester(
            model=model,
            images_paths=images_paths,
            x=0.5,
            y=0.5,
            target_b=0,
            save_folder=None,
            threshold=threshold,
            transforms=A.Resize(self.image_size, self.image_size),
            device=device,
            plot_index=False,
        )

        self.fig, self.axs = plt.subplots(figsize=(16, 16))
        # self.hist_fig, self.hist_axs = plt.subplots(figsize=(8, 8))

        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.outs = self.tester.predict(self.tester.imgs)
        pims = self.tester.plot_predicts(self.tester.imgs, self.outs)
        pim = np.concatenate(pims, axis=1)
        self.axs.imshow(pim)

        # self.hist_axs.hist(pim[pim.shape[0]//2:].flatten(), bins=256, log=True)

        plt.show()

    def onclick(self, event):
        if event.ydata is not None and event.xdata is not None:
            if event.ydata < self.image_size:
                target_b = int(event.xdata // self.image_size)
                self.tester.target_b = target_b
                self.tester.x = event.xdata / self.image_size - target_b
                self.tester.y = event.ydata / self.image_size
                pims = self.tester.plot_predicts(self.tester.imgs, self.outs)
                pim = np.concatenate(pims, axis=1)
                self.axs.imshow(pim)
                self.fig.canvas.draw()
                # self.hist_axs.hist(pim[pim.shape[0] // 2:].flatten(), bins=256, log=True, alpha=0.5)
                # self.hist_fig.canvas.draw()


parser = argparse.ArgumentParser(description='Vizualizer for pixel-wise-embeddings')
parser.add_argument('--model_path', type=str, default='weights/pixel_wise_encoder_v3.pt')
parser.add_argument('--images_path', type=str, default='data/test_images/cars')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--device', type=str, default='cuda')


if __name__ == '__main__':
    args = parser.parse_args()

    # FEATURES_SIZE = 64
    # model = smp.FPN(
    #     'efficientnet-b0',
    #     classes=FEATURES_SIZE,
    #     activation=None,
    #     decoder_segmentation_channels=FEATURES_SIZE * 2,
    #     decoder_pyramid_channels=FEATURES_SIZE * 2,
    #     encoder_weights=None,ss
    # )
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = torch.load(args.model_path, map_location=args.device)
    model.eval()
    model.to(args.device)
    viz = Visualizer(
        model=model,
        images_paths=sorted(glob.glob(args.images_path+'/*')),
        device=args.device,
        image_size=args.image_size,
        threshold=args.threshold,
    )
