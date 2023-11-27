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
