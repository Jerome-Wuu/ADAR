# Alternate Data Augmentation for Generalization in Reinforcement Learning

**[07/18/2024] Under review as a conference paper at AAAI 2025**

Official implementations of 
**Alternate Data Augmentation for Generalization in Reinforcement Learning** (ADAR)


## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:
```
conda env create -f setup/conda.yaml
conda activate adar
sh setup/install_envs.sh
```

Benchmarks for generalization in continuous control from pixels are based on [DMControl Generalization Benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark).
Note: To run the program you need to add dependency files from [data](https://github.com/gemcollector/TLDA/tree/master/src/env/data) to local . /src/env/data

## Training 
The `scripts` directory contains training and evaluation bash scripts for all the included algorithms. Alternatively, you can call the python scripts directly, e.g. for training call

```
python3 src/train.py \
  --algorithm adar \
  --seed 0
```
