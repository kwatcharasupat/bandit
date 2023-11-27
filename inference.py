import glob
import os.path
from pprint import pprint
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from core import LightningSystem
from utils.config import read_nested_yaml

torch.set_float32_matmul_precision("medium")
import torchaudio as ta
from typing import List, Optional, Dict, Union
from tqdm import tqdm

def inference(
        ckpt_path: str,
        file_path: str,
        model_name: str,
        output_dir: Optional[str] = None,
        include_track_name: Optional[bool] = None,
        get_residual: bool = True,
        get_no_vox_combinations: bool = True,
        channel_filter: Optional[Union[int, List[int]]] = None,
) -> None:

    if os.path.isdir(ckpt_path):
        ckpt_dir = ckpt_path
        if os.path.exists(os.path.join(ckpt_path, "checkpoints", "last.ckpt")):
            ckpt = os.path.join(ckpt_path, "checkpoints", "last.ckpt")

        else:
            ckpts = glob.glob(os.path.join(ckpt_path, "checkpoints", "*.ckpt"))
            epochs = [int(os.path.basename(c).split("-")[0].split("=")[-1]) for
                      c in ckpts]
            max_epochs = np.argmax(epochs)
            ckpt = ckpts[max_epochs]
    else:
        ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
        ckpt = ckpt_path

    config: Dict = read_nested_yaml(os.path.join(ckpt_dir, "hparams.yaml"))

    pprint(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    if output_dir is None:
        output_dir = os.path.join(
                os.path.dirname(file_path),
                "separated",
                model_name
        )

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        map_location = None
    else:
        map_location = torch.device('cpu')

    model: LightningSystem = LightningSystem.load_from_checkpoint(
            ckpt,
            config=config["system"],
            map_location=map_location
    )
    model.set_predict_output_path(output_dir)

    # restore the fader to intended value
    model.fader.__init__(  # type: ignore[misc]
            **config["system"]["inference"]["fader"]["kwargs"]
    )
    model.fader.to(model.device)  # type: ignore[arg-type]
    model.eval()

    audio, fs = ta.load(file_path)

    if fs != config["system"]["model"]["kwargs"]["fs"]:
        audio = ta.functional.resample(
                audio,
                fs,
                config["system"]["model"]["kwargs"]["fs"]
        )
        fs = config["system"]["model"]["kwargs"]["fs"]

    track_name = os.path.basename(file_path).split(".")[0]

    track = [track_name]

    treat_batch_as_channels = False

    if channel_filter is not None:
        if isinstance(channel_filter, int):
            channel_filter = [channel_filter]
        audio = audio[channel_filter, :]

    in_channel_audio = audio.shape[0]
    in_channel_model = config["system"]["model"]["kwargs"]["in_channel"]

    if in_channel_audio != in_channel_model:
        if in_channel_audio == 1 and in_channel_model > 1:
            audio = audio.repeat(in_channel_model, 1)
        elif in_channel_audio > 1 and in_channel_model == 1:
            audio = audio[:, None, :]
            treat_batch_as_channels = True
            track = [track_name + f"_{i}" for i in range(in_channel_audio)]
        else:
            raise ValueError(
                    f"Cannot handle in_channel_audio={in_channel_audio} "
                    f"and in_channel_model={in_channel_model}"
            )

    if in_channel_audio == 1 and in_channel_model == 1:
        audio = audio[None, ...]

    audio = audio.to(model.device)

    with torch.inference_mode():
        model.predict_step(
                {
                        "audio": {
                                "mixture": audio,
                        },
                        "track": track,
                },
                get_residual=get_residual,
                get_no_vox_combinations=get_no_vox_combinations,
                include_track_name=include_track_name,
                treat_batch_as_channels=treat_batch_as_channels,
                fs=fs,
        )


def inference_multiple(
    ckpt_path: str,
    model_name: str = None,
    file_glob: str,
    output_dir: Optional[str],
    include_track_name: Optional[bool] = False,
    get_residual: bool = False,
    get_no_vox_combinations: bool = False,
    channel_filter: Optional[Union[int, List[int]]] = None,
) -> None:  
    

    if os.path.isdir(ckpt_path):
        ckpt_dir = ckpt_path
        if os.path.exists(os.path.join(ckpt_path, "checkpoints", "last.ckpt")):
            ckpt = os.path.join(ckpt_path, "checkpoints", "last.ckpt")

        else:
            ckpts = glob.glob(os.path.join(ckpt_path, "checkpoints", "*.ckpt"))
            epochs = [int(os.path.basename(c).split("-")[0].split("=")[-1]) for
                      c in ckpts]
            max_epochs = np.argmax(epochs)
            ckpt = ckpts[max_epochs]
    else:
        ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
        ckpt = ckpt_path

    if model_name is None:
        model_name = os.path.basename(ckpt_dir)
        print(model_name)

    config: Dict = read_nested_yaml(os.path.join(ckpt_dir, "hparams.yaml"))

    pprint(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    files = sorted(glob.glob(file_glob, recursive=True))

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        map_location = None
    else:
        map_location = torch.device('cpu')

    model: LightningSystem = LightningSystem.load_from_checkpoint(
            ckpt,
            config=config["system"],
            map_location=map_location
    )
    model.set_predict_output_path(output_dir)

    # restore the fader to intended value
    model.fader.__init__(  # type: ignore[misc]
            **config["system"]["inference"]["fader"]["kwargs"]
    )
    model.fader.to(model.device)  # type: ignore[arg-type]
    model.eval()

    output_dir_ = output_dir

    for file_path in tqdm(files):

        if output_dir_ is None:
            output_dir = os.path.join(
                    os.path.dirname(file_path),
                    "separated",
                    model_name
            )
        else:
            output_dir = os.path.join(output_dir_, model_name, os.path.basename(os.path.dirname(file_path)))

        os.makedirs(output_dir, exist_ok=True)

        audio, fs = ta.load(file_path)

        if fs != config["system"]["model"]["kwargs"]["fs"]:
            audio = ta.functional.resample(
                    audio,
                    fs,
                    config["system"]["model"]["kwargs"]["fs"]
            )
            fs = config["system"]["model"]["kwargs"]["fs"]

        track_name = os.path.basename(file_path).split(".")[0]

        track = [track_name]

        treat_batch_as_channels = False

        if channel_filter is not None:
            if isinstance(channel_filter, int):
                channel_filter = [channel_filter]
            audio = audio[channel_filter, :]

        in_channel_audio = audio.shape[0]
        kwargs = config["system"]["model"]["kwargs"]
        in_channel_model = kwargs.get("in_channel", None)
        if in_channel_model is None:
            # for demucs
            in_channel_model = kwargs.get("audio_channels", None)
        if in_channel_model is None:
            warnings.warn("in_channel_model is not specified, using 1")
            in_channel_model = 1

        if in_channel_audio != in_channel_model:
            if in_channel_audio == 1 and in_channel_model > 1:
                audio = audio.repeat(in_channel_model, 1)
            elif in_channel_audio > 1 and in_channel_model == 1:
                audio = audio[:, None, :]
                treat_batch_as_channels = True
                track = [track_name + f"_{i}" for i in range(in_channel_audio)]
            else:
                raise ValueError(
                        f"Cannot handle in_channel_audio={in_channel_audio} "
                        f"and in_channel_model={in_channel_model}"
                )

        if in_channel_audio == 1 and in_channel_model == 1:
            audio = audio[None, ...]

        audio = audio.to(model.device)

        model.set_predict_output_path(output_dir)

        with torch.inference_mode():
            model.predict_step(
                    {
                            "audio": {
                                    "mixture": audio,
                            },
                            "track": track,
                    },
                    get_residual=get_residual,
                    get_no_vox_combinations=get_no_vox_combinations,
                    include_track_name=include_track_name,
                    treat_batch_as_channels=treat_batch_as_channels,
                    fs=fs,
            )


if __name__ == "__main__":
    import fire

    fire.Fire()
