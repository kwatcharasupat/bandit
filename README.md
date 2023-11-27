# BandIt: Cinematic Audio Source Separation



Code for "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation" by Karn N. Watcharasupat, Chih-Wei Wu, Yiwei Ding, Iroro Orife, Aaron J. Hipple, Phillip A. Williams, Scott Kramer, Alexander Lerch, William Wolcott. [arXiv](https://arxiv.org/abs/2309.02539)

## For Demo
Go [here](https://karnwatcharasupat.github.io/bandit-demo/) for demo of selected models using the first 10 files from DnR test set. Go [here](https://zenodo.org/records/10119822) for exhaustive inference on the entire DnR test set for selected models. 

## For Replication

- Install required dependencies from `environment.yaml`.
- Obtain DnR dataset from [here](https://zenodo.org/records/5574713) and MUSDB18-HQ from [here](https://sigsep.github.io/datasets/musdb.html).
- Run each dataset's respective `proprocess.py`.
- `python train.py expt/path-to-the-desired-experiment.yaml`.
- `python test.py expt/path-to-the-desired-experiment.yaml --ckpt_path=path/to/checkpoint-from-training.ckpt`.

## For Inference

- Get the checkpoints from [Zenodo](https://zenodo.org/records/10160698)
- `python inference.py expt/path-to-the-desired-model-config.yaml --ckpt_path=path/to/checkpoint.ckpt`

## Complexity Benchmark
Intel Core i9-11900K CPU + NVIDIA GeForce RTX 3090 GPU

| Model         | Band       |   GFlops |   Params (M) |   Peak Memory (MB) |   Batch per second (GPU) |   Batch per second (CPU) |   GPU speedup |
|:--------------|:-----------|--------:|---------:|--------------:|------------------------:|-----------------------:|---------------:|
| BSRNN-GRU8*   | Vocals V7  |   238.2 |     15.8 |         416.2 |                   12.35 |                   0.61 |           20.2 |
| BSRNN-LSTM12* | Vocals V7  |   462.2 |     25.8 |         505.4 |                    7.99 |                   0.6  |           13.4 |
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
@misc{watcharasupat2023generalized,
      title={A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation}, 
      author={Karn N. Watcharasupat and Chih-Wei Wu and Yiwei Ding and Iroro Orife and Aaron J. Hipple and Phillip A. Williams and Scott Kramer and Alexander Lerch and William Wolcott},
      year={2023},
      eprint={2309.02539},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
