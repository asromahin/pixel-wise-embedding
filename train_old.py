import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np


class BaseStep:
    def __init__(self, model, out_collector, loss, metric, device, amp):
        self.model = model
        self.out_collector = out_collector
        self.loss = loss
        self.metric = metric
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
        optim,
        loss,
        metric=None,
        is_log_per_cls=False,
        device='cuda',
        amp=True,
    ):
        super(TrainStep, self).__init__(model=model, out_collector=out_collector, loss=loss, metric=metric, device=device, amp=amp)
        self.optim = optim
        self.is_log_per_cls = is_log_per_cls

    def run(self, dataloader, callbacks=None):
        pbar = tqdm(dataloader)
        log_data = defaultdict(list)
        log_data_per_cls = None
        if self.is_log_per_cls:
            log_data_per_cls = defaultdict(list)
        for i, (im, mask) in enumerate(pbar):
            self.model.zero_grad()
            self.optim.zero_grad()

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
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.step()

            log_data['loss'].append(l.item())

            if callbacks is not None:
                for callback in callbacks:
                    callback()
            pbar.set_postfix({k: np.mean(v) for k, v in log_data.items()})
        log_data = {k: np.mean(v) for k, v in log_data.items()}
        log_data_per_cls = {k: np.mean(v) for k, v in log_data_per_cls.items()}
        return log_data, log_data_per_cls


class TrainStepLossTrain(BaseStep):
    def __init__(
        self,
        model,
        optim,
        loss,
        loss_optim,
        device='cuda',
        amp=True,
    ):
        super(TrainStepLossTrain, self).__init__(model=model, loss=loss, device=device, amp=amp)
        self.optim = optim
        self.loss_optim = loss_optim

    def run(self, dataloader, callbacks=None):
        pbar = tqdm(dataloader)
        log_data = defaultdict(list)
        for i, (im, mask) in enumerate(pbar):
            self.model.zero_grad()
            self.optim.zero_grad()
            self.loss_optim.zero_grad()

            im = im.to(self.device)
            mask = mask.to(self.device)
            with torch.cuda.amp.autocast(self.amp):
                out = self.model(im)
                l = self.loss(out, mask)
            if self.amp:
                l = self.scaler.scale(l)
            l.backward()
            log_data['loss'].append(l.item())
            if self.amp:
                self.scaler.step(self.optim)
                self.scaler.step(self.loss_optim)
                self.scaler.update()
            else:
                self.optim.step()
                self.loss_optim.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()
            pbar.set_postfix({k: np.mean(v) for k, v in log_data.items()})
        log_data = {k: np.mean(v) for k, v in log_data.items()}
        return log_data


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
        log_data_per_cls = {k: np.mean(v) for k, v in log_data_per_cls.items()}
        return log_data, log_data_per_cls




