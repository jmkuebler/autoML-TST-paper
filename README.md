# AutoML Two-Sample Test
This is the code to repoduce the experiments of our work "AutoML Two-Sample Test".
The image shift pipeline builds upon [the repository](https://github.com/steverab/failing-loudly)
from the paper "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift".

## Installation

Use conda to install the environment.
```
conda create -n automl_tst python=3.7
conda activate automl_tst
pip install -r requirements.txt
pip install git+https://github.com/josipd/torch-two-sample.git
pip install autogluon
pip install mxnet --upgrade
```

## Datasets & Models

MNIST and CIFAR10, the corresponding adversarial example as well as the pretrained datasets
need to be downloaded from the [Failing Loudly repository](https://github.com/steverab/failing-loudly).
The Higgs dataset is provided by the authors of the article
"Learning Deep Kernels for Non-Parametric Two-Sample Tests" and it can be downloaded 
[here](https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc).


## Running experiments

To repoduce the experiments, use the scripts `blob.py`, `higgs.py`, and `img_shift.py`.
The parameters to be used are given in documentation of each script.
