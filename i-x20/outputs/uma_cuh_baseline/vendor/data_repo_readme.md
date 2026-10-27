<p align="center">
  <img src="https://github.com/wgst/ml-gas-surface/blob/main/docs/figures/ml_to_gas_surf.png?raw=true" width="300">
</p>

<h1 align="center">ML for Gas-Surface Dynamics</h1>

<p align="center">
  <b>Fast and accurate molecular dynamics simulations using ML-based interatomic potentials and electronic friction models.</b>
</p>

---
## 🧠 Overview

This repository accompanies the research on nonadiabatic reactive scattering of hydrogen on copper surfaces. It provides:

- Trained machine learning interatomic potentials (MLIPs)
- Scripts for high-throughput molecular dynamics (MD)
- Electronic friction models, including local density friction approximation (LDFA), and orbital-dependent friction (ODF) models.
- Tools for state-to-state scattering dynamics.
- Databases for model training and validation


## ⚙️ Key Features

- **MLIP construction** through adaptive sampling
  - High-error structure search with [NQCDynamics.jl](https://github.com/NQCD/NQCDynamics.jl)
  - Clustering-based structure selection
- **Full MD pipelines** for:
  - Dissociative chemisorption dynamics for evaluation of sticking probabilities
  - State-to-state scattering dynamics

## 🚀 Getting Started

📚 Full documentation and instructions:  
👉 [**Project Website**](https://wgst.github.io/ml-gas-surface/)


---

## 📖 References

If this code or data helped your work, please cite:

1. **H<sub>2</sub>/Cu Nonadiabatic dynamics, construction of electronic friction models:**  
   * W. G. Stark, C. L. Box, M. Sachs, N. Hertl, R. J. Maurer, Nonadiabatic reactive scattering of hydrogen on different surface facets of copper, Phys. Rev. B, 112, 3 (2025) [[arXiv]](http://arxiv.org/abs/2505.18147) [[journal]](https://link.aps.org/doi/10.1103/h7vd-94pk)

2. **H<sub>2</sub>/Cu dissociative chemisorption, construction of MLIPs**  
   * W. G. Stark, J. Westermayr, O. A. Douglas-Gallardo, J. Gardner, S. Habershon, R. J. Maurer, Machine learning interatomic potentials for reactive hydrogen dynamics at metal surfaces based on iterative refinement of reaction probabilities, J. Phys. Chem. C, 127, 50, 24168–24182, (2023) [[arXiv]](https://arxiv.org/abs/2305.10873) [[journal]](https://pubs.acs.org/doi/10.1021/acs.jpcc.3c06648)

3. **NQCDynamics code:**  
   * J. Gardner, O. A. Douglas-Gallardo, W. G. Stark, J. Westermayr, S. M. Janke, S. Habershon, R. J. Maurer, NQCDynamics.jl: A Julia package for nonadiabatic quantum classical molecular dynamics in the condensed phase, J. Chem. Phys. 156, 174801 (2022) [[arXiv]](https://arxiv.org/abs/2202.12925) [[journal]](https://doi.org/10.1063/5.0089436)


<details>
<summary>BibTeX</summary>


```bibtex
@article{stark_nonadiabatic_2025,
  title = {Nonadiabatic reactive scattering of hydrogen on different surface facets of copper},
  author = {Stark, Wojciech G. and Box, Connor L. and Sachs, Matthias and Hertl, Nils and Maurer, Reinhard J.},
  publisher = {Phys. Rev. B},
  volume = {112},
  number = {3},
  doi = {10.1103/h7vd-94pk},
  year = {2025},
  url = {https://link.aps.org/doi/10.1103/h7vd-94pk},
}

@article{stark_machine_2023,
  title = {Machine learning interatomic potentials for reactive hydrogen dynamics at metal surfaces},
  author = {Stark, Wojciech G. et al.},
  journal = {J. Phys. Chem. C},
  year = {2023},
  volume = {127},
  number = {50},
  pages = {24168–24182},
  doi = {10.1021/acs.jpcc.3c06648}
}

@article{gardner_nqcdynamicsjl_2022,
  title = {{NQCDynamics}.jl: {A} Julia package for nonadiabatic quantum classical molecular dynamics},
  author = {Gardner, James et al.},
  journal = {J. Chem. Phys.},
  volume = {156},
  number = {17},
  pages = {174801},
  year = {2022},
  doi = {10.1063/5.0089436}
}
```

</details>
