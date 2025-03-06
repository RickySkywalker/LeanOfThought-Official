# LeanOfThought-Official

This is the official repository for all the code of MA-LoT. 
    * Paper link: (https://arxiv.org/abs/2503.03205)[https://arxiv.org/abs/2503.03205]
    * LoT-Solver Model: (https://huggingface.co/RickyDeSkywalker/LoT-Solver)[https://huggingface.co/RickyDeSkywalker/LoT-Solver]

Current version of code only supports MA-LoT inference of a single theorem, the step-by-step usage can be found in `quick_start.ipynb`.

## Clone Repo
```
git clone --recurse-submodules https://github.com/RickySkywalker/LeanOfThought-Official.git
```

## Setup Env

### Lean4 Env

#### Install Lean4
Following Lean4 installation page (https://lean-lang.org/lean4/doc/quickstart.html)[https://lean-lang.org/lean4/doc/quickstart.html] to install Lean4 env

#### Build Mathlib4
```bash
cd mathlib4
lake build
```

### Python env
```bash
conda create -n LoT_env python=3.10
pip install -r requirements.txt
```

## Run quick start:

See `quick_start.ipynb`.