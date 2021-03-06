# RSMI-NE

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`RSMI-NE` is a Python package, implemented using Tensorflow, for optimising coarse-graining rules for real-space renormalisation group by maximising real-space mutual information. 

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation](#installation-guide)
- [Citation](#citation)

## Overview

`RSMI-NE` employs state-of-the-art results for estimating mutual information (MI) by maximising its lower-bounds parametrised by deep neural networks [Poole et al. (2019), [arXiv:1905.06922v1](https://arxiv.org/abs/1905.06922)]. This allows it to overcome the severe limitations of the initial proposals for constructing real-space RG transformations by MI-maximization in [M. Koch-Janusz and Z. Ringel, [Nature Phys. 14, 578-582 (2018)](https://doi.org/10.1038/s41567-018-0081-4), P.M. Lenggenhager et al., [Phys.Rev. X 10, 011037 (2020)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011037)], and to reconstruct the relevant operators of the theory, as detailed in the manuscripts accompanying this code [D.E. Gökmen, Z. Ringel, S.D. Huber and M. Koch-Janusz (2021), [arXiv:2101.11633](https://arxiv.org/abs/2101.11633) and [arXiv:2103.16887](https://arxiv.org/abs/2103.16887)].

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
./install/install.sh
```

## Getting started

Jupyter notebooks demonstrating the basic usage in simple examples are provided in <https://github.com/RSMI-NE/RSMI-NE/tree/main/coarsegrainer/examples>.

## Citation

If you use RSMI-NE in your work, for the moment please cite our arxiv preprints [arXiv:2101.11633](https://arxiv.org/abs/2101.11633) and [arXiv:2103.16887](https://arxiv.org/abs/2103.16887):

```latex
@misc{gökmen2021statistical,
      title={Statistical physics through the lens of real-space mutual information}, 
      author={Doruk Efe Gökmen and Zohar Ringel and Sebastian D. Huber and Maciej Koch-Janusz},
      year={2021},
      eprint={2101.11633},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech}
}

@misc{gökmen2021phase,
      title={Phase diagrams with real-space mutual information neural estimation}, 
      author={Doruk Efe Gökmen and Zohar Ringel and Sebastian D. Huber and Maciej Koch-Janusz},
      year={2021},
      eprint={2103.16887},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech}
}
```

