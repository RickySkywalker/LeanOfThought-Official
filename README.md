# LeanOfThought-Official

This is the official repository for all the MA-LoT codes. In this repo, we provide an interactive way of using our proposed pipeline to solve Lean4 theorems.
* Paper link: [https://arxiv.org/abs/2503.03205](https://arxiv.org/abs/2503.03205)
* LoT-Solver (Based on DeepSeek-Prover-v1.5-SFT): [https://huggingface.co/RickyDeSkywalker/LoT-Solver](https://huggingface.co/RickyDeSkywalker/LoT-Solver)
* LoT-Solver (Based on Godel-Prover-SFT): [https://huggingface.co/RickyDeSkywalker/LoT-Solver-Godel](https://huggingface.co/RickyDeSkywalker/LoT-Solver-Godel)
* Website: [https://ma-lot.github.io/](https://ma-lot.github.io/)

The current version of the code only supports MA-LoT inference of a single theorem; the step-by-step usage can be found in `quick_start.ipynb`.


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

## Reference
```bib
@misc{wang2025malot,
      title={MA-LoT: Model-Collaboration Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving}, 
      author={Ruida Wang and Rui Pan and Yuxin Li and Jipeng Zhang and Yizhen Jia and Shizhe Diao and Renjie Pi and Junjie Hu and Tong Zhang},
      year={2025},
      eprint={2503.03205},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.03205}, 
}
```

## Acknowledgement
We gratefully acknowledge the computational resources provided under project codes CIS240529 and CIS240076, which were essential to the completion of this work.
This project would not have been possible without the support of [DeltaAI](https://delta.ncsa.illinois.edu/delta-delta-ai/) and [Delta](https://delta.ncsa.illinois.edu/).
