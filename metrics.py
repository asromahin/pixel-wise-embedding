import torch
import segmentation_models_pytorch as smp


class MulticlassAccuracy(torch.nn.Module):
    def __init__(self):
        super(MulticlassAccuracy, self).__init__()

    def forward(self, x, y):
        return ((x.argmax(dim=1) == y)/1).mean()

#
# class BinaryAccuracy(torch.nn.Module):
#     def __init__(self):
#         super(BinaryAccuracy, self).__init__()
#
#     def forward(self, x, y):
#         return (x.argmax(dim=1) == y).mean()