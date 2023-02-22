# `causalimages`: `R` Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images 

*This software is under beta release. The first stable release occur  sometime in spring of 2023. Email `connor.jerzak@gmail.com` for feature requests.*

# Repository Structure
This repository contains two main components. The first component is code of the `causalimages` software package. At the moment, the package includes code for performing the image-based heterogeneity decomposition described in Jerzak, Johansson, and Daoud (2023). In future releases, we will add functionality for using images in observational causal inference (i.e., as a proxy for confounding variables). 

The second component is replication data. The replication data for the Youth Opportunities Experiment is found in `data/Replication Data/YOP Experiment`. This includes both outcome data and geo-referenced satellite images. We will keep adding to this replication repository with new research being performed over time. 


# Package Download Instructions 
From within `R`, you may download via the `devtools` package. In particular, use 

```
devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")
```

Then, to load the software in, use 
```
library(   causalimages  ) 
```

# Tutorial
Under construction.

# Development Plan
We currently have in beta release code for interpretably decomposing treatment effect heterogeneity by image. In the next stage, we will implement two more functionalities: (1) confounder adjustment via image and (2) causal image system simulation. Core machine learning modules are written in `tensorflow+tensorflow_probability`; subsequent versions may be transfered over to `equinox+oryx+jax`. 

# Acknowledgements
We thank James Bailie, Cindy Conlin, Devdatt Dubhashi, Felipe Jordan, Mohammad Kakooei, Eagon Meng, Xiao-Li Meng, and Markus Pettersson for valuable feedback on this project and software. We also thank Xiaolong Yang for excellent research assistance.

# References
Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Image-based Treatment Effect Heterogeneity. *Forthcoming in Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)*, 2023. [`arxiv.org/pdf/2206.06417.pdf`](https://arxiv.org/pdf/2206.06417.pdf)

Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities. *ArXiv Preprint*, 2023. [`arxiv.org/pdf/2301.12985.pdf`](https://arxiv.org/pdf/2301.12985.pdf)
