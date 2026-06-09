# FSTL-SA: Few-Shot Transfer Learning for Sentiment Analysis from Facial Expressions

[![MTAP](https://img.shields.io/badge/Multimedia%20Tools%20and%20Applications-2024-blue.svg)](https://doi.org/10.1007/s11042-024-20518-y) [![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11042--024--20518--y-green.svg)](https://doi.org/10.1007/s11042-024-20518-y) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Official PyTorch implementation of **"FSTL-SA: Few-Shot Transfer Learning for Sentiment Analysis from Facial Expressions"**, published in *Multimedia Tools and Applications* (Springer, 2024).



## Installation

```bash
git clone https://github.com/katikapallilokesh/fewshot_vsa.git
cd fewshot_vsa
pip install -r requirements.txt
```



## Data Preparation

```bash
python -m src.data.download --kaggle_json /path/to/kaggle.json --data_root data
python -m src.data.prepare --data_root data
```


## Running Experiments

### Pretrain the backbone

```bash
python pretrain.py --data_root data --epochs 100 --batch_size 32
```

### Run few-shot experiments

```bash
python run_experiment.py --backbone checkpoints/backbone.pt --data_root data
```

To run a quick subset of k values:

```bash
python run_experiment.py --k_shots 1 5 10 20 --epochs 20
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--k_shots` | `1 5 10 20 ... 100` | k values to evaluate |
| `--val_size` | `15` | Validation samples per class |
| `--test_size` | `15` | Test samples per class |
| `--epochs` | `20` | Fine-tuning epochs per condition |
| `--threshold` | `0.99` | Pseudo-label confidence threshold |
| `--single_mult` | `0.25` | Pseudo-label multiplier, single SS round |
| `--iterative_mult` | `0.30` | Pseudo-label multiplier, iterative SS round |


## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{meena2025fstl,
  title={FSTL-SA: Few-shot transfer learning for sentiment analysis from facial expressions},
  author={Meena, Gaurav and Mohbey, Krishna Kumar and Lokesh, K},
  journal={Multimedia Tools and Applications},
  volume={84},
  number={21},
  pages={24457--24485},
  year={2025},
  publisher={Springer}
}
```