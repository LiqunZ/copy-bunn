# Bundle Neural Networks for message diffusion on graphs


This is the official repository for the ICLR 2025 submission of the paper "Bundle Neural Networks for message diffusion on graphs". The repository contains all the code necessary to reproduce the experiments in the paper. 

There are three main folders in this repository:
- [heterophilousgraphs](heterophilousgraphs), forked from the original repository of [A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?](https://github.com/yandex-research/heterophilous-graphs/tree/main), and containing all the code necessary to run the heterophilous graph benhcmark
- [LRGB-tuned](LRGB-tuned), forked from the original repository of [Reassessing the Long-Range Graph Benchmark](https://github.com/toenshoff/LRGB), itself based on the [LRGB codebase](https://github.com/rampasek/GraphGPS), and containing all the code neccasry to run the LRGB experiments
- [synthexp](synthexp), which contains all the synthetic experiments 

We also forked [Neural Sheaf Diffusion](https://github.com/twitter-research/neural-sheaf-diffusion) to evaluate Neural Sheaf Diffusion on the heterophilous graphs

The two first folders were minimally modified from their source. We simply defined BuNNs in their framework to evaluate them. Each of the three main folder corresponds to a main experiment: 

![big_main.png](big_main.png)

### 1. Heterophilous graph baselines
First, cd into the heterophilousgraphs folder: <code> cd heterophilousgraphs </code>

Then, follow the guidelines in the README of the heterophilousgraphs folder to setup the environment.
Once that is done, all the best hyper-parameter configuration, and commands are in the [scripts/best_runs.sh](heterophilousgraphs/scripts/all_runs.sh).
For example, again from the same folder you can run the best hyperparameter for BuNN on minesweeper (make sure to change the device to <code>cuda:0</code> if available):

<code> python -m train  --name BUNN --dataset minesweeper --model BUNN --num_layers 8 --num_gnn 8 --device cpu --num_bundles 256 --hidden_dim 512 --max_degree 8 --tau 1
</code>

### 2. LRGB baselines
First, cd into the LRGB-tuned folder: <code> cd LRGB-tuned </code>

Then, follow the guidelines in the README to setup the correct environment.
All best hyper-parameter configurations are in the [configs/bundle](LRGB-tuned/configs/bundle) folder and each script is in the [scripts](LRGB-tuned/scripts) folder.

For example, to run BuNN on peptides-func, run:
<code>python  main.py --cfg configs/bundle/peptides-func-bunn.yaml seed 0</code>

### 3. Synthetic experiments

First, cd into the  [synthexp](synthexp) folder, where all the synthetic experiment code is: <code>cd synthexp</code>
Then, follow the guidelines in the README to setup the correct environment.
All hyperparameter configurations are in the [configs](synthexp/configs) folder, and all scripts in [all_synth.sh](synthexp/scripts/all_synth.sh).

Sor run one specific experiment, for example the uniform expressiveness one, run:

<code> python -m synthexp.run_multiple --yaml_name barbell-gcn </code>


Thank you for reading!
