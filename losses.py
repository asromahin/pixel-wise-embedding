import torch
from pytorch_metric_learning import distances
from segmentation_models_pytorch import losses


def multiclass_out_and_mask(out, mask):
  out = torch.softmax(out, dim=1)
  mask = mask.argmax(dim=1)
  return out, mask


class BasePixelWiseLoss(torch.nn.Module):
  def __init__(
          self,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          loss=losses.DiceLoss('multiclass', from_logits=False),
          loss_prepare_callback=multiclass_out_and_mask,
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):

    super(BasePixelWiseLoss, self).__init__()

    self.is_full = is_full
    self.ignore_classes = [] if ignore_classes is None else ignore_classes
    self.batch_isolated = batch_isolated

    self.distance = distance
    self.distance_callback = distance_callback
    self.loss = loss
    self.loss_prepare_callback = loss_prepare_callback

  def prepare_input(self, x):
    full_out = x.permute(0, 2, 3, 1)
    out_flat = full_out.reshape(-1, full_out.shape[-1])
    return full_out, out_flat

  def generate_masks(self, collect_out_list, collect_target_mask_list, is_full=False):
    collect_target_mask = torch.cat(collect_target_mask_list, dim=1).to(dtype=torch.float32)
    collect_out = torch.cat(collect_out_list, dim=1)
    if not is_full:
        zero_out = -torch.sum(collect_out, dim=1)/len(collect_out_list)
        zero_mask = torch.min(collect_target_mask, dim=1).values == 0
        collect_out_list.append(zero_out.unsqueeze(1))
        collect_target_mask_list.append(zero_mask.unsqueeze(1))

        collect_target_mask = torch.cat(collect_target_mask_list, dim=1).to(dtype=torch.float32)
        collect_out = torch.cat(collect_out_list, dim=1)
    return collect_out, collect_target_mask

  def calc_loss(self, outs, masks):
    if self.batch_isolated:
      res_loss = None
      for b in range(outs.shape[0]):
        cur_loss = self.loss(outs[b:b+1], masks[b:b+1])
        if res_loss is None:
          res_loss = cur_loss
        else:
          res_loss += cur_loss
        res_loss = res_loss / outs.shape[0]
    else:
      res_loss = self.loss(outs, masks)
    return res_loss

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    raise NotImplementedError()

  def forward(self, x, y):
    full_out, out_flat = self.prepare_input(x)

    collect_target_mask_list, collect_out_list = self.extract_mask_for_each_target(out_flat, full_out, y)

    collect_out, collect_target_mask = self.generate_masks(collect_out_list, collect_target_mask_list, self.is_full)

    collect_out, collect_target_mask = self.loss_prepare_callback(collect_out, collect_target_mask)

    res_loss = self.calc_loss(collect_out, collect_target_mask)
    return res_loss


class PixelWiseLossWithMeanVector(BasePixelWiseLoss):
  def __init__(
          self,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          loss=losses.DiceLoss('multiclass', from_logits=False),
          loss_prepare_callback=multiclass_out_and_mask,
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):
    super(PixelWiseLossWithMeanVector, self).__init__(
      distance=distance,
      distance_callback=distance_callback,
      loss=loss,
      loss_prepare_callback=loss_prepare_callback,
      is_full=is_full,
      ignore_classes=ignore_classes,
      batch_isolated=batch_isolated,
    )

  def extract_vector(self, cmask, full_out):
    cout = full_out[cmask]
    return torch.mean(cout, dim=0)

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    collect_target_mask_list = []
    collect_out_list = []

    utarget = torch.unique(target)

    for i, u in enumerate(utarget):
      if int(u) in self.ignore_classes:
        continue
      cmask = (target == u)
      vector = self.extract_vector(cmask, full_out)
      distance_mask = self.distance(out_flat, vector.unsqueeze(0))
      distance_mask = distance_mask.reshape(full_out.shape[0], full_out.shape[1], full_out.shape[2])
      distance_mask = self.distance_callback(distance_mask)
      collect_target_mask_list.append(cmask.unsqueeze(1))
      collect_out_list.append(distance_mask.unsqueeze(1))
    return collect_target_mask_list, collect_out_list


class PixelWiseLossWithVectors(BasePixelWiseLoss):
  def __init__(
          self,
          n_classes,
          features_size,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          loss=losses.DiceLoss('multiclass', from_logits=False),
          loss_prepare_callback=multiclass_out_and_mask,
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):
    super(PixelWiseLossWithVectors, self).__init__(
      distance=distance,
      distance_callback=distance_callback,
      loss=loss,
      loss_prepare_callback=loss_prepare_callback,
      is_full=is_full,
      ignore_classes=ignore_classes,
      batch_isolated=batch_isolated,
    )
    self.n_classes = n_classes
    self.features_size = features_size

    self.vectors = torch.nn.Parameter(torch.zeros((self.n_classes, self.features_size)), requires_grad=True)

  def extract_vector(self, uniq_target):
    return self.vectors[uniq_target]

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    collect_target_mask_list = []
    collect_out_list = []

    utarget = torch.unique(target)

    for i, u in enumerate(utarget):
      if int(u) in self.ignore_classes:
        continue
      cmask = (target == u)
      vector = self.extract_vector(u)
      distance_mask = self.distance(out_flat, vector.unsqueeze(0))
      distance_mask = distance_mask.reshape(full_out.shape[0], full_out.shape[1], full_out.shape[2])
      distance_mask = self.distance_callback(distance_mask)
      collect_target_mask_list.append(cmask.unsqueeze(1))
      collect_out_list.append(distance_mask.unsqueeze(1))
    return collect_target_mask_list, collect_out_list


class PixelWiseLossWithVectorsConvFit(PixelWiseLossWithVectors):
  def __init__(
          self,
          n_classes,
          features_size,
          distance=distances.CosineSimilarity(),
          loss=losses.DiceLoss('multiclass', from_logits=False),
          loss_prepare_callback=multiclass_out_and_mask,
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):
    super(PixelWiseLossWithVectorsConvFit, self).__init__(
      n_classes=n_classes,
      features_size=features_size,
      distance=distance,
      distance_callback=self.fit_distance,
      loss=loss,
      loss_prepare_callback=loss_prepare_callback,
      is_full=is_full,
      ignore_classes=ignore_classes,
      batch_isolated=batch_isolated,
    )

    self.fit_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

  def fit_distance(self, distance_mask):
    distance_mask = self.fit_conv(distance_mask.unsqueeze(1))[:, 0]
    return distance_mask





