import os.path
import typing
import typing
import warnings
from pprint import pprint

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.distributed import fsdp as fsdp_

from core import LightningSystem, data
from core.data.augmentation import StemAugmentor
from utils.config import dict_to_trainer_kwargs, read_nested_yaml

torch.set_float32_matmul_precision("medium")
from torch._dynamo import config

config.verbose = True
config.cache_size_limit = 1024

from typing import Any
import copy

import torchmetrics as tm

torch.backends.cudnn.benchmark = True


def test(
        config_path: str,
        ckpt_path: typing.Optional[str] = None,
        **kwargs: Any
) -> None:

    # if torch.cuda.device_count() > 1:
    #     raise RuntimeError("Testing should only be done on a single GPU for reproducibility.")

    config = read_nested_yaml(config_path)

    config_ = copy.deepcopy(config)

    pprint(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    assert isinstance(config["data"], dict)
    assert isinstance(config["data"]["data"], dict)
    dmcls = config["data"]["data"].pop("datamodule")
    assert isinstance(dmcls, str)

    datamodule = data.__dict__[dmcls](**config["data"]["data"])

    assert isinstance(config["trainer"], dict)
    trainer_kwargs = dict_to_trainer_kwargs(config["trainer"])
    loss_adjustment = 1.0

    strategy = "auto"

    trainer_kwargs['callbacks'].append(RichModelSummary(max_depth=3))

    trainer = pl.Trainer(
            # devices=,
            **trainer_kwargs,  # type: ignore[arg-type]
            strategy=strategy,
            **kwargs,
    )

    assert isinstance(config["system"], dict)
    if "augmentation" in config["system"]:
        pass
    elif "augmentation" in config["data"]:
        warnings.warn(
                "Augmentation should now be put under system.augmentation "
                "instead of data.augmentation.",
                DeprecationWarning,
        )
        config["system"]["augmentation"] = config["data"]["augmentation"]
    else:
        config["system"]["augmentation"] = None

    model = LightningSystem(config["system"], loss_adjustment=loss_adjustment)

    if model.fader is not None:
        model.fader = None
    model.attach_fader(force_reattach=True)
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    import fire

    fire.Fire(test)
