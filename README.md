# Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling
Code for the paper:

> Greg Ver Steeg and Aram Galstyan. "Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling", NeurIPS 2021.
> [[preprint]](https://github.com/gregversteeg/esh_dynamics/blob/master/assets/draft.pdf) [[bibtex]](#bibtex)

<p align="center">
<img align="middle" src="./assets/esh.jpg" width="500" />
</p>

Non-Newtonian Momentum Animation:
<p align="center">
<img align="middle" src="./assets/loop.gif" width="500" />
</p>

This repo contains code for implementing **E**nergy **S**ampling **H**amiltonian Dynamics, 
so-called because the Hamiltonian dynamics with this special form of Non-Newtonian momentum 
ergodically samples from a target un-normalized density specified by an energy function. 

## Requirements

The main sampler code for ESH dynamics is esh_leap.py and uses only PyTorch. 
```
pip install git+https://github.com/gregversteeg/esh_dynamics
```
Use ``pip install -r requirements.txt`` for requirements for all evaluation code. 

## Usage
TODO: add a simple energy model, show Langevin vs ESH

## Generating figures

See the README in the ``generate_figures`` for scripts to generate each figure in the paper, 
and to see more example usage. 


## BibTeX

```markdown
@inproceedings{esh,
  title={Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling},
  author={Greg {Ver Steeg} and Aram Galstyan},
  Booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

