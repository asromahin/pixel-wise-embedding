import torch
from pytorch_metric_learning import distances


def multiclass_out(out, mask):
  out = torch.softmax(out, dim=1)
  mask = mask.argmax(dim=1)
  return out, mask


def binary_out(out, mask):
  out = torch.sigmoid(out)
  return out, mask


class BaseOutCollector(torch.nn.Module):
  def __init__(
          self,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          # filter_out=multiclass_out,
          ignore_classes=None,
  ):

    super(BaseOutCollector, self).__init__()

    self.ignore_classes = [] if ignore_classes is None else ignore_classes

    self.distance = distance
    self.distance_callback = distance_callback
    # self.filter_out = filter_out

  def prepare_input(self, x):
    full_out = x.permute(0, 2, 3, 1)
    out_flat = full_out.reshape(-1, full_out.shape[-1])
    return full_out, out_flat

  def generate_masks(self, collect_out_list, collect_target_mask_list):
    collect_target_mask = torch.cat(collect_target_mask_list, dim=1).to(dtype=torch.float32)
    collect_out = torch.cat(collect_out_list, dim=1)
    return collect_out, collect_target_mask

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    raise NotImplementedError()

  def forward(self, x, y):
    full_out, out_flat = self.prepare_input(x)

    collect_target_mask_list, collect_out_list, collect_target_cls = self.extract_mask_for_each_target(out_flat, full_out, y)

    collect_out, collect_target_mask = self.generate_masks(collect_out_list, collect_target_mask_list)

    # collect_out, collect_target_mask = self.filter_out(collect_out, collect_target_mask)

    return collect_out, collect_target_mask, collect_target_cls


class OutCollectorWithMeanVector(BaseOutCollector):
  def __init__(
          self,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          # filter_out=multiclass_out,
          ignore_classes=None,
  ):
    super(OutCollectorWithMeanVector, self).__init__(
      distance=distance,
      distance_callback=distance_callback,
      ignore_classes=ignore_classes,
      # filter_out=filter_out,
    )

  def extract_vector(self, cmask, full_out):
    cout = full_out[cmask]
    return torch.mean(cout, dim=0)

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    collect_target_mask_list = []
    collect_out_list = []
    collect_target_cls = []

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
      collect_target_cls.append(u)
    return collect_target_mask_list, collect_out_list, collect_target_cls


class OutCollectorWithLearningVectors(BaseOutCollector):
  def __init__(
          self,
          n_classes,
          features_size,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          # filter_out=multiclass_out,
          ignore_classes=None,
  ):
    super(OutCollectorWithLearningVectors, self).__init__(
      distance=distance,
      distance_callback=distance_callback,
      ignore_classes=ignore_classes,
      # filter_out=filter_out,
    )
    self.n_classes = n_classes
    self.features_size = features_size

    self.vectors = torch.nn.Parameter(torch.zeros((self.n_classes, self.features_size)), requires_grad=True)

  def extract_vector(self, uniq_target):
    return self.vectors[uniq_target]

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    collect_target_mask_list = []
    collect_out_list = []
    collect_target_cls = []

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
      collect_target_cls.append(u)
    return collect_target_mask_list, collect_out_list, collect_target_cls


class OutCollectorWithLearningVectorsWithConv(OutCollectorWithLearningVectors):
  def __init__(
          self,
          n_classes,
          features_size,
          kernel_size=1,
          distance=distances.CosineSimilarity(),
          # filter_out=multiclass_out,
          ignore_classes=None,
  ):
    self.kernel_size = kernel_size
    conv = torch.nn.Conv2d(1, 1, kernel_size=(self.kernel_size, self.kernel_size), stride=(1, 1), padding=(self.kernel_size//2, self.kernel_size//2))

    super(OutCollectorWithLearningVectorsWithConv, self).__init__(
      n_classes=n_classes,
      features_size=features_size,
      distance=distance,
      distance_callback=self.distance_callback,
      ignore_classes=ignore_classes,
      # filter_out=filter_out,
    )

    self.conv = conv

  def distance_callback(self, x):
    return self.conv(x.unsqueeze(1))[:, 0]


class OutCollectorWithSingleConv(BaseOutCollector):
  def __init__(
          self,
          n_classes,
          features_size,
          distance=distances.CosineSimilarity(),
          distance_callback=lambda x: x,
          # filter_out=multiclass_out,
          ignore_classes=None,
  ):
    super(OutCollectorWithSingleConv, self).__init__(
      distance=distance,
      distance_callback=distance_callback,
      ignore_classes=ignore_classes,
      # filter_out=filter_out,
    )
    self.n_classes = n_classes
    self.features_size = features_size
    self.conv = torch.nn.Conv2d(self.features_size, self.n_classes, kernel_size=(1, 1), padding=0)

  def extract_vector(self, cmask, full_out):
    cout = full_out[cmask]
    return torch.mean(cout, dim=0)

  def extract_mask_for_each_target(self, out_flat, full_out, target):
    collect_target_mask_list = []
    collect_out_list = []
    collect_target_cls = []

    utarget = torch.unique(target)

    outs = self.conv(full_out.permute(0, 3, 1, 2))

    for i, u in enumerate(utarget):
      if int(u) in self.ignore_classes:
        continue
      cmask = (target == u)
      out_mask = outs[:, u]
      collect_target_mask_list.append(cmask.unsqueeze(1))
      collect_out_list.append(out_mask.unsqueeze(1))
      collect_target_cls.append(u)
    return collect_target_mask_list, collect_out_list, collect_target_cls



