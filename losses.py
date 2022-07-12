import torch
from pytorch_metric_learning import distances
from segmentation_models_pytorch import losses


class BasePixelWiseLoss(torch.nn.Module):
  def __init__(
          self,
          distance=distances.CosineSimilarity(),
          loss=losses.DiceLoss('multiclass', from_logits=False),
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):

    super(BasePixelWiseLoss, self).__init__()

    self.is_full = is_full
    self.ignore_classes = [] if ignore_classes is None else ignore_classes
    self.batch_isolated = batch_isolated

    self.distance = distance
    self.loss = loss

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

    collect_out, collect_target_mask = self.generate_masks(collect_target_mask_list, collect_out_list, self.is_full)

    collect_out = torch.softmax(collect_out, dim=1)
    collect_target_mask = collect_target_mask.argmax(dim=1)

    res_loss = self.calc_loss(collect_out, collect_target_mask)
    return res_loss


class PixelWiseLossWithMeanVector(BasePixelWiseLoss):
  def __init__(
          self,
          distance=distances.CosineSimilarity(),
          loss=losses.DiceLoss('multiclass', from_logits=False),
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):
    super(PixelWiseLossWithMeanVector, self).__init__(
      distance=distance,
      loss=loss,
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
      mean_mask = self.distance(out_flat, vector.unsqueeze(0))
      mean_mask = mean_mask.reshape(full_out.shape[0], full_out.shape[1], full_out.shape[2])
      collect_target_mask_list.append(cmask.unsqueeze(1))
      collect_out_list.append(mean_mask.unsqueeze(1))
    return collect_target_mask_list, collect_out_list


class PixelWiseLossWithVectors(BasePixelWiseLoss):
  def __init__(
          self,
          n_classes,
          features_size,
          distance=distances.CosineSimilarity(),
          loss=losses.DiceLoss('multiclass', from_logits=False),
          is_full=True,
          ignore_classes=None,
          batch_isolated=False,
  ):
    super(PixelWiseLossWithVectors, self).__init__(
      distance=distance,
      loss=loss,
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
      mean_mask = self.distance(out_flat, vector.unsqueeze(0))
      mean_mask = mean_mask.reshape(full_out.shape[0], full_out.shape[1], full_out.shape[2])
      collect_target_mask_list.append(cmask.unsqueeze(1))
      collect_out_list.append(mean_mask.unsqueeze(1))
    return collect_target_mask_list, collect_out_list




