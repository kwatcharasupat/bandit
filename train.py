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
# torch.autograd.set_detect_anomaly(True)

def train(
        config_path: str,
        ckpt_path: typing.Optional[str] = None,
        adjust_loss_in_lieu_of_accumulate_grad_batches: bool = False,
        strategy: typing.Optional[str] = "ddp",
        just_validate: bool = False,
        val_batch_size: typing.Optional[int] = None,
        finetune: bool = False,
        **kwargs: Any
) -> None:
    config = read_nested_yaml(config_path)

    config_ = copy.deepcopy(config)

    pprint(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    if just_validate:
        config["trainer"]["logger"]["kwargs"]["name"] = "validation"

    assert isinstance(config["data"], dict)
    assert isinstance(config["data"]["data"], dict)
    dmcls = config["data"]["data"].pop("datamodule")
    assert isinstance(dmcls, str)

    if val_batch_size is not None:
        assert just_validate
        config["data"]["data"]["batch_size"] = val_batch_size

    datamodule = data.__dict__[dmcls](**config["data"]["data"])

    assert isinstance(config["trainer"], dict)
    trainer_kwargs = dict_to_trainer_kwargs(config["trainer"])
    loss_adjustment = 1.0

    if "effective_batch_size" in trainer_kwargs:
        assert ("accumulate_grad_batches" not in trainer_kwargs) or (
                trainer_kwargs["accumulate_grad_batches"] is None
        )
        val_batch_size = config["data"]["data"]["batch_size"]
        gpu_count = torch.cuda.device_count()
        assert isinstance(val_batch_size, int)
        effective_batch_size = trainer_kwargs.pop("effective_batch_size")
        assert isinstance(effective_batch_size, int)
        assert effective_batch_size % (gpu_count * val_batch_size) == 0
        accumulate_grad_batches = effective_batch_size // (
                gpu_count * val_batch_size)

        if adjust_loss_in_lieu_of_accumulate_grad_batches:
            loss_adjustment = 1.0 / accumulate_grad_batches
            use_static_graph = True
            trainer_kwargs["accumulate_grad_batches"] = 1
        else:
            trainer_kwargs["accumulate_grad_batches"] = accumulate_grad_batches
            print(
                    f"Batch size: {val_batch_size}. Requesting effective batch size: {effective_batch_size}."
            )
            print(
                    f"Accumulating gradients from {accumulate_grad_batches} batches."
            )
            use_static_graph = False
    else:
        use_static_graph = "accumulate_grad_batches" not in trainer_kwargs

    if torch.cuda.device_count() == 1:
        strategy = "auto"
    else:
        if strategy == "ddp":
            strategy = DDPStrategy(
                    static_graph=use_static_graph, gradient_as_bucket_view=True
            )

    trainer_kwargs['callbacks'].append(RichModelSummary(max_depth=3))

    trainer = pl.Trainer(
            **trainer_kwargs,  # type: ignore[arg-type]
            strategy=strategy,
            # profiler=AdvancedProfiler(filename="profiler.txt"),
            **kwargs,
            # profiler=PyTorchProfiler(with_modules=True, filename="profiler.txt")
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

    if finetune in ["dnr->musdb", "dnr->mne"]:
        assert ckpt_path is not None

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False
            )

        missing = set([m.split(".")[2] for m in missing])
        unexpected = set([u.split(".")[2] for u in unexpected])

        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        ckpt_path = None
        # raise NotImplementedError
        del ckpt


    assert trainer.logger is not None

    trainer.logger.log_hyperparams(config_)
    trainer.logger.save()

    if just_validate:
        if ckpt_path is None:
            ckpt_path = os.path.join(os.path.dirname(config_path), "checkpoints", "last.ckpt")
            assert os.path.exists(ckpt_path)
        trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    model.attach_fader()
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    import fire

    fire.Fire(train)
