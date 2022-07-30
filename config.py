from dataclasses import dataclass
import typing as tp


@dataclass
class BaseConfig:
    model: tp.Callable
    model_kwargs: tp.Dict

    loss = tp.Callable
    loss_kwargs = tp.Dict

    ade20k_dataset_path: str
    device: str
