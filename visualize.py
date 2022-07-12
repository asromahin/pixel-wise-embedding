import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import glob
import albumentations as A
import numpy as np

from tester import Tester


class Visualizer:
    def __init__(self, model, images_paths, image_size=256, device='cuda'):
        self.image_size = image_size
        self.tester = Tester(
            model=model,
            images_paths=images_paths,
            x=0.5,
            y=0.5,
            target_b=0,
            save_folder=None,
            transforms=A.Resize(self.image_size, self.image_size),
            device=device,
            plot_index=False,
        )

        self.fig, self.axs = plt.subplots(figsize=(16, 16))

        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        pims = self.tester.plot_predicts(self.tester.imgs)
        pim = np.concatenate(pims, axis=1)
        self.axs.imshow(pim)
        plt.show()

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.ydata < self.image_size:
            target_b = int(event.xdata // self.image_size)
            print(target_b)
            self.tester.target_b = target_b
            self.tester.x = event.xdata / self.image_size - target_b
            self.tester.y = event.ydata / self.image_size
            pims = self.tester.plot_predicts(self.tester.imgs)
            pim = np.concatenate(pims, axis=1)
            self.axs.imshow(pim)
            # self.fig.clf()
            self.axs.imshow(pim)
            self.fig.canvas.draw()


if __name__ == '__main__':
    model = torch.load('weights/pixel_wise_encoder.pt', map_location='cpu')
    model.eval()
    viz = Visualizer(
        model=model,
        images_paths=sorted(glob.glob('data/test_images/plates_v2/*')),
        device='cpu',
        image_size=256+128,
    )