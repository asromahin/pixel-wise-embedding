import torch
import cv2
import numpy as np
import os
from pytorch_metric_learning import distances
from PIL import Image
from torchvision.transforms import ToTensor


def gif_from_folder(folder, save_path='result.gif', duration=1000):
  files = os.listdir(folder)
  files = sorted(files, key=lambda x: int(x.split('.')[0]))
  # Список для хранения кадров.
  frames = []

  for file in files:
    # Открываем изображение каждого кадра.
    frame = Image.open(os.path.join(folder, file))
    # Добавляем кадр в список с кадрами.
    frames.append(frame)

  # Берем первый кадр и в него добавляем оставшееся кадры.
  frames[0].save(
    save_path,
    save_all=True,
    append_images=frames[1:],  # Срез который игнорирует первый кадр.
    optimize=True,
    duration=duration,
    loop=0
  )


def get_ref(out1, out2, x, y):
  # print(out1.shape, out2.shape)
  ref_vector = out1[y,x]
  # print(ref_vector)
  reshape_target = out2.reshape(-1, out2.shape[-1])
  dist = distances.CosineSimilarity()(torch.tensor(reshape_target), torch.tensor(ref_vector).unsqueeze(0)).detach().cpu().numpy()
  # print(reshape_target.shape)
  # dist2 = KMeans(n_clusters=5).fit_predict(dist2)
  # print(dist2.shape)
  dist = dist.reshape((out2.shape[0], out2.shape[1]))
  return dist


def plot_text(image, text):
  font = cv2.FONT_HERSHEY_SIMPLEX
  org = (50, 50)
  fontScale = 1
  color = (0, 0, 255)
  thickness = 2
  image = cv2.putText(image, text, org, font, fontScale,
                      color, thickness, cv2.LINE_AA, False)
  return image


class Tester:
  def __init__(
          self,
          model,
          images_paths,
          x,
          y,
          target_b,
          save_folder,
          transforms,
          threshold=0.9,
          radius=5,
          gif_duration=500,
          run_every=100,
          device='cuda',
          plot_index=True,
  ):
    self.model = model
    self.images_paths = images_paths
    self.x = x
    self.y = y
    self.target_b = target_b
    self.transforms = transforms
    self.threshold = threshold
    self.gif_duration = gif_duration
    self.radius = radius
    self.plot_index = plot_index

    self.save_folder = save_folder
    self.device = device

    self.run_every = run_every
    self.iter = 0
    self._real_iter = 0
    self.imgs = self.read_images(self.images_paths)

  def read_images(self, images_paths):
    images = []
    for image_path in images_paths:
      im = cv2.imread(image_path)
      im = self.transforms(image=im)['image']
      im = ToTensor()(im)
      images.append(im.unsqueeze(0))
    images = torch.cat(images, dim=0)
    return images

  def plot_predicts(self, imgs):
    with torch.no_grad():
      outs = self.model(imgs.to(self.device)).detach().cpu().numpy()
    out_np = np.moveaxis(outs, 1, -1)
    x = int(self.x * imgs.shape[3])
    y = int(self.y * imgs.shape[2])
    pims = []
    for b in range(len(imgs)):
      pim = (imgs[b].permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8').copy()
      if b == self.target_b:
        pim = cv2.circle(pim.copy(), (x, y), self.radius, (255, 0, 0), 3)
      dist = get_ref(out_np[self.target_b], out_np[b], x, y)
      mask = (dist > self.threshold).astype('uint8')
      cntrs, _ = cv2.findContours(mask, 0, 1)
      cv2.drawContours(pim, cntrs, -1, (0, 0, 255), 3)
      dist = np.clip(dist, -1, 1)
      dist = (dist + 1) / 2
      dist = (dist * 255).astype('uint8')
      dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
      pim = cv2.cvtColor(pim, cv2.COLOR_BGR2RGB)
      pim = np.concatenate([pim, dist], axis=0)
      if self.plot_index:
        pim = plot_text(pim, str(self.iter))
      pims.append(pim)
    return pims

  def save_results(self, pims):
    for b in range(len(self.imgs)):
      b_name = self.images_paths[b].split('.')[0]
      folder_path = os.path.join(self.save_folder, b_name)
      os.makedirs(folder_path, exist_ok=True)
      cv2.imwrite(os.path.join(folder_path, str(self.iter) + '.jpg'), pims[b])
      gif_from_folder(
        folder_path,
        save_path=os.path.join(self.save_folder, b_name + '.gif'),
        duration=self.gif_duration,
      )

  def test(self):
    if self._real_iter % self.run_every == 0:
      pims = self.plot_predicts(self.imgs)
      self.save_results(pims)
      self.iter += 1
    self._real_iter += 1
