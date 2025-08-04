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
This research used both the DeltaAI advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2320345) and the State of Illinois, and the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois.. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.
