# RSMI-NE

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

`RSMI-NE` is a Python package, implemented using Tensorflow, for optimising coarse-graining rules for real-space renormalisation group by maximising real-space mutual information. 

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation](#installation-guide)
- [Citation](#citation)

## Overview

`RSMI-NE` employs state-of-the-art results for estimating mutual information (MI) by maximising its lower-bounds parametrised by deep neural networks [Poole et al. (2019), [arXiv:1905.06922v1](https://arxiv.org/abs/1905.06922)]. This allows it to overcome the severe limitations of the initial proposals for constructing real-space RG transformations by MI-maximization in [M. Koch-Janusz and Z. Ringel, [Nature Phys. 14, 578-582 (2018)](https://doi.org/10.1038/s41567-018-0081-4), P.M. Lenggenhager et al., [Phys.Rev. X 10, 011037 (2020)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011037)], and to reconstruct the relevant operators of the theory, as detailed in the manuscripts accompanying this code [D.E. Gökmen, Z. Ringel, S.D. Huber and M. Koch-Janusz, [Phys. Rev. Lett. **127**, 240603 (2021)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.240603) and [Phys. Rev. E **104**, 064106 (2021)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.064106)].

## System requirements

### Hardware requirements

`RSMI-NE`  can be run on a standard personal computer. It has been tested on the following setup (without GPU):

+ CPU: 2.3 GHz Quad-Core Intel Core i5, Memory: 8 GB 2133 MHz LPDDR3

### Software requirements

This package has been tested on the following systems with Python 3.8.5:

+ macOS:
  + Catalina (10.15)
  + Big Sur (11.1)

`RSMI-NE` mainly depends on the following Python packages:

* matplotlib
* numpy
* pandas
* scipy
* sklearn
* tensorflow 2.0
* tensorflow_probability

## Installation

Clone `RSMI-NE` from Github and install its dependencies into a virtual environment:

```bash
git clone https://github.com/RSMI-NE/RSMI-NE
cd RSMI-NE
./install.sh
```

## Getting started

Jupyter notebooks demonstrating the basic usage in simple examples are provided in <https://github.com/RSMI-NE/RSMI-NE/examples>.

## Citation

If you use RSMI-NE in your work, please cite our publications [Phys. Rev. Lett. **127**, 240603 (2021)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.240603) and [Phys. Rev. E **104**, 064106 (2021)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.064106):

```latex
@article{PhysRevLett.127.240603,
  title = {Statistical Physics through the Lens of Real-Space Mutual Information},
  author = {G\"okmen, Doruk Efe and Ringel, Zohar and Huber, Sebastian D. and Koch-Janusz, Maciej},
  journal = {Phys. Rev. Lett.},
  volume = {127},
  issue = {24},
  pages = {240603},
  numpages = {7},
  year = {2021},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.127.240603},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.127.240603}
}

@article{PhysRevE.104.064106,
  title = {Symmetries and phase diagrams with real-space mutual information neural estimation},
  author = {G\"okmen, Doruk Efe and Ringel, Zohar and Huber, Sebastian D. and Koch-Janusz, Maciej},
  journal = {Phys. Rev. E},
  volume = {104},
  issue = {6},
  pages = {064106},
  numpages = {17},
  year = {2021},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.104.064106},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.104.064106}
}
```
