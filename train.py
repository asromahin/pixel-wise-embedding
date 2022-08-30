import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np


MODEL_KEY = 'model'
COLLECTOR_KEY = 'collector'
LOSS_KEY = 'loss'
METRIC_KEY = 'metric'


class BaseStep:
    def __init__(self, model, out_collector, loss, metric, device, amp):
        self.model = model
        self.out_collector = out_collector
        self.loss = loss
        self.metric = metric
        self._objects = {
            MODEL_KEY: self.model,
            COLLECTOR_KEY: self.out_collector,
            LOSS_KEY: self.loss,
            METRIC_KEY: self.metric,
        }
        self.device = device
        self.amp = amp
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def run(self, dataloader, callbacks=None):
        raise NotImplementedError


class TrainStep(BaseStep):
    def __init__(
        self,
        model,
        out_collector,
        # optim,
        loss,
        metric=None,
        trainable_objects=(MODEL_KEY, ),
        is_log_per_cls=False,
        device='cuda',
        amp=True,
    ):
        super(TrainStep, self).__init__(
            model=model,
            out_collector=out_collector,
            loss=loss,
            metric=metric,
            device=device,
            amp=amp)

        for obj in trainable_objects:
            assert obj in [MODEL_KEY, COLLECTOR_KEY, LOSS_KEY, METRIC_KEY]
        assert MODEL_KEY in trainable_objects

        self.trainable_objets = trainable_objects
        self.optims = {key: torch.optim.Adam(self._objects[key].parameters(), eps=1e-4) for key in self.trainable_objets}
        self.is_log_per_cls = is_log_per_cls

    def run(self, dataloader, callbacks=None):
        pbar = tqdm(dataloader)
        log_data = defaultdict(list)
        log_data_per_cls = None
        if self.is_log_per_cls:
            log_data_per_cls = defaultdict(list)
        for i, (im, mask) in enumerate(pbar):
            for key in self.trainable_objets:
                self._objects[key].zero_grad()
                self.optims[key].zero_grad()

            im = im.to(self.device)
            mask = mask.to(self.device)

            with torch.cuda.amp.autocast(self.amp):
                out = self.model(im)
                out, mask, target_cls = self.out_collector(out, mask)
                l = self.loss(out, mask)
                if self.metric is not None:
                    m = self.metric(out, mask)
                    if self.is_log_per_cls:
                        for i in range(len(m)):
                            log_data_per_cls[self.metric.__name__+str(int(target_cls[i]))].append(m[i].item())
                    if m.shape:
                        m = m.mean()
                    log_data[self.metric.__name__].append(m.item())

            if self.amp:
                self.scaler.scale(l).backward()
            else:
                l.backward()

            if self.amp:
                for key in self.trainable_objets:
                    self.scaler.step(self.optims[key])
                    self.scaler.update()
            else:
                for key in self.trainable_objets:
                    self.optims[key].step()

            log_data['loss'].append(l.item())

            if callbacks is not None:
                for callback in callbacks:
                    callback()
            pbar.set_postfix({k: np.mean(v) for k, v in log_data.items()})
        log_data = {k: np.mean(v) for k, v in log_data.items()}
        if self.is_log_per_cls:
            log_data_per_cls = {k: np.mean(v) for k, v in log_data_per_cls.items()}
        return log_data, log_data_per_cls


class ValStep(BaseStep):
    def __init__(
            self,
            model,
            out_collector,
            loss,
            metric=None,
            is_log_per_cls=False,
            device='cuda',
            amp=True,
    ):
        super(ValStep, self).__init__(model=model, out_collector=out_collector, loss=loss, metric=metric, device=device, amp=amp)
        self.is_log_per_cls = is_log_per_cls

    def run(self, dataloader, callbacks=None):
        pbar = tqdm(dataloader)
        log_data = defaultdict(list)
        log_data_per_cls = None
        if self.is_log_per_cls:
            log_data_per_cls = defaultdict(list)
        for i, (im, mask) in enumerate(pbar):
            with torch.no_grad():
                out = self.model(im.to(self.device))
                out, mask, target_cls = self.out_collector(out, mask)
                l = self.loss(out, mask.to(self.device))
                log_data['loss'].append(l.item())
                if self.metric is not None:
                    m = self.metric(out, mask)
                    if self.is_log_per_cls:
                        for i in range(len(m)):
                            log_data_per_cls[self.metric.__name__+str(int(target_cls[i]))].append(m[i].item())
                    if m.shape:
                        m = m.mean()
                    log_data[self.metric.__name__].append(m.item())
            if callbacks is not None:
                for callback in callbacks:
                    callback()
            pbar.set_postfix({k: np.mean(v) for k, v in log_data.items()})
        log_data = {k: np.mean(v) for k, v in log_data.items()}
        if self.is_log_per_cls:
            log_data_per_cls = {k: np.mean(v) for k, v in log_data_per_cls.items()}
        return log_data, log_data_per_cls




