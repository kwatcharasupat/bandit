# BandIt: Cinematic Audio Source Separation
Code for "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation" by Karn N. Watcharasupat, Chih-Wei Wu, Yiwei Ding, Iroro Orife, Aaron J. Hipple, Phillip A. Williams, Scott Kramer, Alexander Lerch, William Wolcott. [[open-access paper]](https://ieeexplore.ieee.org/document/10342812)

> Cinematic audio source separation is a relatively new subtask of audio source separation, with the aim of extracting the dialogue, music, and effects stems from their mixture. In this work, we developed a model generalizing the Bandsplit RNN for any complete or overcomplete partitions of the frequency axis. Psychoacoustically motivated frequency scales were used to inform the band definitions which are now defined with redundancy for more reliable feature extraction. A loss function motivated by the signal-to-noise ratio and the sparsity-promoting property of the 1-norm was proposed. We additionally exploit the information-sharing property of a common-encoder setup to reduce computational complexity during both training and inference, improve separation performance for hard-to-generalize classes of sounds, and allow flexibility during inference time with detachable decoders. Our best model sets the state of the art on the Divide and Remaster dataset with performance above the ideal ratio mask for the dialogue stem.

For the query-based music source separation model, Banquet, go [here](https://github.com/kwatcharasupat/query-bandit).

## For Demo
Go [here](https://kwatcharasupat.github.io/bandit-demo/) for demo of selected models using the first 10 files from DnR test set. Go [here](https://zenodo.org/records/10119822) for exhaustive inference on the entire DnR test set for selected models. 

## For Replication

- Install required dependencies from `environment.yaml`.
- Obtain DnR dataset from [here](https://zenodo.org/records/5574713) and MUSDB18-HQ from [here](https://sigsep.github.io/datasets/musdb.html).
- Run each dataset's respective `proprocess.py`.
- `python train.py expt/path-to-the-desired-experiment.yaml`.
- `python test.py expt/path-to-the-desired-experiment.yaml --ckpt_path=path/to/checkpoint-from-training.ckpt`.

## For Inference

- Get the checkpoints from [Zenodo](https://zenodo.org/records/10160698). 
- Get the corresponding yaml config file from `expt`.
- Put the checkpoint and the yaml config file into the same subfolder. Rename the config file `hparams.yaml`.
- If you run into CUDA OOM, try reducing the batch size in the inference config. Another way without changing the config itself is by setting the `system.inference` parameter to `"file:$PROJECT_ROOT/configs/inference/default16.yaml"`, or `default8.yaml`.
- If you run into a CPU OOM, this is probably due to the resampler. You might want to get your audio file to 44.1 kHz beforehand, especially if it's big. A fix is coming (soon??).
- Please do not hesitate to report other OOM cases.

```bash
python inference.py inference \
  --ckpt_path=path/to/checkpoint.ckpt \
  --file_path=path/to/file.wav \
  --model_name=model_id
```
or

```bash
python inference.py inference_multiple \
  --ckpt_path=path/to/checkpoint.ckpt \
  --file_glob=path/to/glob/*.wav \
  --model_name=model_id
```

## Complexity Benchmark
- Intel Core i9-11900K CPU + NVIDIA GeForce RTX 3090 GPU.
- Note that this is benchmarked on _one_ 6-second chunk. The memory usage in practice will scale according to your inference batch size plus some OLA overhead.

| Model         | Band       |   GFlops |   Params (M) |   Peak Memory (MB) |   Batch per second (GPU) |   Batch per second (CPU) |   GPU speedup |
|:--------------|:-----------|--------:|---------:|--------------:|------------------------:|-----------------------:|---------------:|
| BSRNN-GRU8 (per stem)   | Vocals V7  |   238.2 |     15.8 |         416.2 |                   12.35 |                   0.61 |           20.2 |
| BSRNN-LSTM12 (per stem) | Vocals V7  |   462.2 |     25.8 |         505.4 |                    7.99 |                   0.6  |           13.4 |
| BandIt        | Bark 48    |   290.6 |     64.5 |         643   |                   10.22 |                   0.39 |           26.1 |
| BandIt        | ERB 48     |   274.2 |     32.6 |         519.5 |                   10.31 |                   0.41 |           25   |
| BandIt        | Mel 48     |   274.3 |     32.8 |         519.3 |                   10.15 |                   0.38 |           26.6 |
| BandIt        | Music 48   |   274.7 |     33.5 |         524.2 |                   10.22 |                   0.43 |           23.5 |
| BandIt        | TriBark 48 |   274.2 |     32.7 |         519.9 |                   10.3  |                   0.4  |           25.5 |
| BandIt        | Bark 64    |   387.6 |     82.6 |         828.5 |                    8.64 |                   0.4  |           21.9 |
| BandIt        | ERB 64     |   363.5 |     36   |         649.1 |                    8.68 |                   0.42 |           20.7 |
| BandIt        | Mel 64     |   363.6 |     36.1 |         648.9 |                    8.71 |                   0.32 |           27.2 |
| BandIt        | Music 64   |   364.1 |     37   |         653   |                    8.69 |                   0.31 |           27.7 |
| BandIt        | TriBark 64 |   363.5 |     36   |         648.7 |                    8.68 |                   0.42 |           20.6 |
| BandIt        | Vocals V7  |   243.2 |     25.7 |         454   |                   11.34 |                   0.6  |           18.8 |
| Hybrid Demucs |            |    85   |     83.6 |         552.5 |                   17.04 |                   1.1  |           15.5 |
| Open-Unmix    |            |     5.7 |     22.1 |         187.8 |                   52.5  |                  20.77 |            2.5 |

## Citation

```
@article{Watcharasupat2023Bandit
  author={Watcharasupat, Karn N. and Wu, Chih-Wei and Ding, Yiwei and Orife, Iroro and Hipple, Aaron J. and Williams, Phillip A. and Kramer, Scott and Lerch, Alexander and Wolcott, William},
  journal={IEEE Open Journal of Signal Processing}, 
  title={A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation}, 
  year={2024},
  volume={5},
  number={},
  pages={73-81},
  doi={10.1109/OJSP.2023.3339428}}
```
