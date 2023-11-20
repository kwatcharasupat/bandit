import glob
import json
from fvcore.nn import FlopCountAnalysis, flop_count_table
import os.path
import typing
import typing
import warnings
from pprint import pprint

import numpy as np

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

from pytorch_benchmark import benchmark

import pandas as pd

torch.backends.cudnn.benchmark = False

def fvcore(
        config_path: str,
        fs: int = 44100,
        chunk_size_seconds: float = 6.0,
) -> None:

    config = read_nested_yaml(config_path)

    config_ = copy.deepcopy(config)

    pprint(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    model = LightningSystem(config["system"]).to("cuda")

    dummy_inputs = {
        "audio": {
            "mixture": torch.randn((1, 1, int(fs * chunk_size_seconds))).to("cuda"),
        }
    }

    flops = FlopCountAnalysis(model, dummy_inputs)

    print(np.round(flops.total() / (1024**3), 1))

    print(flop_count_table(flops))


def bench(
        config_path: str,
        fs: int = 44100,
        chunk_size_seconds: float = 6.0,
        device: str = "cpu",
) -> None:

    config = read_nested_yaml(config_path)

    config_ = copy.deepcopy(config)

    pprint(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.to(device)

        def forward(self, x):
            return self.model({
                "audio": {
                    "mixture": x.to(device)
                }
            })

    model = LightningSystem(config["system"]).to(device)
    model = Wrapper(model).to(device)

    dummy_inputs = torch.randn((1, 1, int(fs * chunk_size_seconds))).to(device)

    def transfer_to_device_fn(x, *args, **kwargs):
        return x

    # flops = model(dummy_inputs)
    flops = benchmark(model, dummy_inputs, transfer_to_device_fn=transfer_to_device_fn)

    basename = os.path.basename(config_path).split(".")[0]

    if device != "cuda":
        basename += f"-{device}"


    with open(f"{os.environ['PROJECT_ROOT']}/benchmarks/{basename}.json", "w") as f:
        json.dump(flops, f, indent=4)

    # print(np.round(torch.cuda.max_memory_allocated() / 1024**3, 1))

def summary(benchmark_path: str = f"{os.environ['PROJECT_ROOT']}/benchmarks") -> None:
    
    benches = glob.glob(f"{benchmark_path}/*.json")

    dfs = []

    for bench in benches:
        with open(bench, "r") as f:
            data = json.load(f)

        if "cpu" in bench:
            continue

        model = bench.split("/")[-1].split(".")[0]

        if "umx" in model:
            model = "umx"
            band = ""
            nband = None
        elif "demucs" in model:
            model = "demucs"
            band = ""
            nband = None
        elif "3s" in model:
            band = model.split("3s-")[-1].split("-")[0]
            nband = int(band[-2:]) if band != "vox7" else None
            model = "bandit"
        else: 
            model = "bsrnn" + ("-large" if "large" in model else "") 
            band = "vox7"
            nband = None

        # use flops from fvcore, not pytorch_benchmark
        # pytorch_benchmark flops use ptops and might not be accurate for RNN

        df = {
            "model": model,
            "band": band,
            "nband": nband,
            # "flops": data["flops"] / (1024 ** 3),
            "params": data["params"] / (10**6),
            "peak_memory": data["memory"]["batch_size_1"]["max_inference_bytes"] / (1024*1024),
            f"batch_per_second_cuda": data["timing"]["batch_size_1"]["total"]["metrics"]["batches_per_second_mean"],
        }

        bench_cpu = bench.replace(".json", "-cpu.json")

        if os.path.exists(bench_cpu):
            with open(bench_cpu, "r") as f:
                data = json.load(f)

            df[f"batch_per_second_cpu"] = data["timing"]["batch_size_1"]["on_device_inference"]["metrics"]["batches_per_second_mean"]

        dfs.append(df)

    df = pd.DataFrame.from_records(dfs).sort_values(by=["model", "nband", "band"])

    df[df["model"].isin(["bsrnn", "bsrnn-large"])][["params"]] *= 3

    df["cuda_speedup"] = df["batch_per_second_cuda"] / df["batch_per_second_cpu"]

    df.to_csv(f"{benchmark_path}/summary.csv", index=False)

    df = df.drop(columns=["nband", "params"])

    print(df.to_latex(index=False, float_format="%.2f"))


if __name__ == "__main__":
    import fire

    fire.Fire()