# `causalimages`: An R Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images

<!--
Tests not ready 
[![Tutorial Tests](https://github.com/cjerzak/causalimages-software/actions/workflows/tutorial-tests.yml/badge.svg)](https://github.com/cjerzak/causalimages-software/actions/workflows/tutorial-tests.yml)
-->

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-View%20Dataset-orange?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/theaidevlab)

[**What is `causalimages`?**](#description)
| [**Installation**](#installation)
| [**Pipeline**](#pipeline)
| [**Image Heterogeneity Tutorial**](#tutorial)
| [**Other Package Functions**](#otherfunctions)
| [**Data**](https://connorjerzak.com/data/)
| [**References**](#references)
| [**Documentation**](https://github.com/cjerzak/causalimages-software/blob/main/causalimages.pdf)

Additional tutorials: 
[**Image-based De-confounding**](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageConfounding_Tutorial.R)
| [**Image/Video Representations**](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/ExtractImageRepresentations_Tutorial.R) | [**Building tfrecord corpus**](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/UsingTfRecords_Tutorial.R)

Replication data: 
[**Heterogeneity Paper**](https://doi.org/10.7910/DVN/O8XOSF) | 
[**De-confounding Paper**](https://doi.org/10.7910/DVN/QLCSVR)

_Beta package version:_ [`GitHub.com/cjerzak/causalimages-software`](https://github.com/cjerzak/causalimages-software)

_Stable package version:_ [`GitHub.com/AIandGlobalDevelopmentLab/causalimages-software`](https://github.com/AIandGlobalDevelopmentLab/causalimages-software)

# What is causalimages?<a id="description"></a>

Causal inference has entered a new stage where novel data sources are being integrated into the study of cause and effect. Image information is a particularly promising data stream in this context: it is widely available and richly informative in social science and bio-medical contexts.

This package, `causalimages`, enables causal analysis with images. For example, the function, `AnalyzeImageHeterogeneity`, performs the image-based treatment effect heterogeneity decomposition described in [Jerzak, Johansson, and Daoud (2023)](https://proceedings.mlr.press/v213/jerzak23a/jerzak23a.pdf). This function can be used, for example, to determine which neighborhoods are most responsive to an anti-poverty intervention using earth observation data from, e.g., satellites. In the bio-medical domain, this function could be used to model the kinds of patients who would be most responsive to interventions on the basis of pre-treatment diagnostic imaging. See [References](#references) for a link to replication data for the image heterogeneity paper; see [this tutorial](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_Tutorial.R) for a walkthrough using the replication data. 

The function, `AnalyzeImageConfounding`, performs the image-based deconfounding analysis described in [Jerzak, Johansson, and Daoud (2023+)](https://arxiv.org/pdf/2301.12985.pdf). This function can be used, for example, to control for confounding factors correlated with both neighborhood wealth and aid decisions in observational studies of development. In the bio-medical context, this function could be used to control for confounding variables captured via diagnostic imaging in order to improve observational inference.

# Package Installation<a id="installation"></a>
From within `R`, you may download via the `devtools` package. In particular, use 

```
devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")
```

Then, to load the software, use 
```
library(   causalimages  ) 
```

# Pipeline<a id="pipeline"></a>
Use of `causalimages` generally follows the following pipeline. Steps 1 and 2 will be necessary for all downstream tasks. 

*1. Build package backend.* This establishes the necessary modules, including JAX and Equinox, used in the causal image modeling. We attempt to establish GPU acceleration where that hardware is available. For tutorial, see [`tutorials/BuildBackend_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/BuildBackend_Tutorial.R) for more information. You can try using `conda="auto"` or finding the correct path to the conda executable by typing ``where conda`` in the terminal: 
```
causalimages::BuildBackend(conda = "/Users/cjerzak/miniforge3/bin/python")
```

If you prefer to manually install the backend, create a conda environment and
install the Python packages used by `causalimages`. The commands below replicate
what `BuildBackend()` performs under the hood (Python 3.10 or newer is
recommended):

```bash
conda create -n CausalImagesEnv python=3.11
conda activate CausalImagesEnv
python3 -m pip install tensorflow tensorflow-metal optax equinox jmp tensorflow_probability
python3 -m pip install jax-metal
```

*2. Write TfRecord.* Next, you will need to write a TfRecord representation of your image or image sequence corpus. This function converts your image corpus into efficient float16 representations for fast reading of the images into memory for model training and output generation. For a tutorial, see [`tutorials/CausalImage_TfRecordFxns.R`](https://github.com/cjerzak/causalimages-software/blob/main/causalimages/R/CausalImage_TfRecordFxns.R)
```
# see: 
?causalimages::WriteTfRecord
```

*3. Generate image representations for downstream tasks.* You sometimes will only want to extract representations of your image or image sequence corpus. In that case, you'll use `GetImageRepresentations()`. For tutorial, see [`tutorials/ExtractImageRepresentations_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/ExtractImageRepresentations_Tutorial.R).
```
# for help, see:
?causalimages::GetImageRepresentations
```

### Pretrained Models

`GetImageRepresentations()` supports several pretrained vision models via the `pretrainedModel` parameter:

**Built-in Models:**
- `"vit-base"` - Google's Vision Transformer (ViT-Base, 768-dim embeddings)
- `"clip-rsicd"` - CLIP fine-tuned on remote sensing data (512-dim embeddings)
- `"clip-rsicd-v0"` - Legacy CLIP-RSICD implementation

**Generic Transformers Models:**

You can use any HuggingFace vision model by using the `transformers-` prefix:

```r
# DINOv2 (self-supervised vision transformer)
pretrainedModel = "transformers-facebook/dinov2-base"

# ResNet-50
pretrainedModel = "transformers-microsoft/resnet-50"

# ConvNeXt
pretrainedModel = "transformers-facebook/convnext-base-224"

# Any other HuggingFace vision model
pretrainedModel = "transformers-google/vit-large-patch16-224"
```

The generic handler uses `AutoModel.from_pretrained()` for maximum flexibility and automatically:
- Detects the model's hidden dimension from its config
- Uses the model's expected input size (defaults to 224×224)
- Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Handles different output formats (pooler_output, global average pooling, etc.)

Example usage:
```r
ImageReps <- causalimages::GetImageRepresentations(
  file = "~/Downloads/CausalIm.tfrecord",
  imageKeysOfUnits = KeysOfObservations,
  pretrainedModel = "transformers-facebook/dinov2-base",
  NORM_MEAN = c(0, 0, 0),  # dataset-specific normalization
  NORM_SD = c(1, 1, 1)
)
``` 

*4. Perform causal image analysis.* Finally, you may also want to perform a causal analysis using the image or image sequence data. For a tutorial on image-based treatment effect heterogeneity, see [`tutorials/AnalyzeImageHeterogeneity_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_Tutorial.R). For a tutorial on image-based confounding analysis, see [`tutorials/AnalyzeImageConfounding_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageConfounding_Tutorial.R). 
```
# for help, see also: 
?causalimages::AnalyzeImageHeterogeneity
?causalimages::AnalyzeImageConfounding
``` 

# Image Heterogeneity Tutorial<a id="tutorial"></a>
The most up-to-date tutorials are found in the [`tutorials`](https://github.com/cjerzak/causalimages-software/tree/main/tutorials) folder of this GitHub. Here, we also provide an abbreviated tutorial using an image heterogeneity analysis. 

## Load in Tutorial Data
After we've loaded in the package, we can get started running an analysis. Let's read in the tutorial data so we can explore its structure: 
```
# outcome, treatment, and covariate information: 
summary( obsW <- causalimages::obsW ) # treatment vector 
summary( obsY <- causalimages::obsY ) # outcome vector 
summary( LongLat <- causalimages::LongLat ) # long-lat coordinates for each unit
summary( X <- causalimages::X ) # other covariates 

# image information: 
dim( FullImageArray <- causalimages::FullImageArray ) # dimensions of the full image array in memory 
head( KeysOfImages <- causalimages::KeysOfImages ) # unique image keys associated with the images in FullImageArray
head( KeysOfObservations <- causalimages::KeysOfObservations ) # image keys of observations to be associated to images via KeysOfImages
```
We can also analyze the images that we'll use in this analysis. 
```
# plot the second band of the third image
causalimages::image2(FullImageArray[3,,,2])

# plot the first band of the first image
causalimages::image2(FullImageArray[1,,,1])
```
We're using rather small image bricks around each long/lat coordinate so that this tutorial code is memory efficient. In practice, your images will be larger and you'll usually have to read them in from disk (with those instructions outlined in the `acquireImageFxn` function that you'll specify). We have an example of that approach later in the tutorial. 

## Writing image corpus to `tfrecord`
One important part of the image analysis pipeline is writing the image corpus `tfrecord` file for efficient model training. You will use the `causalimages::WriteTfRecord` function, which takes as an input another function, `acquireImageFxn`, as an argument which we use for extracting all the images and writing them to the `tfrecord`. There are two ways that you can approach this: (1) you may store all images in `R`'s memory, or you may (2) save images on your hard drive and read them in when needed while generating the `tfrecord`. The second option will be more common for large images. 

You must write your `acquireImageFxn` to take in a single argument: `keys`.
- `keys` (a positional argument) is a character or numeric vector. Each value of `keys` refers to a unique image object that will be read in. If each observation has a unique image associated with it, perhaps `imageKeysOfUnits = 1:nObs`. In the example we'll use, multiple observations map to the same image. 


### When Loading All Images in Memory 
In this tutorial, we have all the images in memory in the `FullImageArray` array. We can write an `acquireImageFxn` function like so: 
```
acquireImageFromMemory <- function(keys){
  # here, the function input keys
  # refers to the unit-associated image keys
  m_ <- FullImageArray[match(keys, KeysOfImages),,,]

  # if keys == 1, add the batch dimension so output dims are always consistent
  # (here in image case, dims are batch by height by width by channel)
  if(length(keys) == 1){
    m_ <- array(m_,dim = c(1L,dim(m_)[1],dim(m_)[2],dim(m_)[3]))
  }
  return( m_ )
}

OneImage <- acquireImageFromMemory(sample(KeysOfObservations,1))
dim( OneImage )

ImageSample <- acquireImageFromMemory(sample(KeysOfObservations,10))
dim( ImageSample )

# plot image: it's always a good idea 
# to check the images through extensive sanity checks
# such as your comparing satellite image representation
# against those from OpenStreetMaps or Google Earth. 
causalimages::image2( ImageSample[3,,,1] )
```

Now, let's write the `tfrecord`:
```
causalimages::WriteTfRecord(  file = "~/Downloads/CausalIm.tfrecord",
                  uniqueImageKeys = unique( KeysOfObservations ),
                  acquireImageFxn = acquireImageFromMemory )
# Note: You first may need to call causalimages::BuildBackend() to build the backend (done only once)
```

### When Reading in Images from Disk 
For most applications of large-scale causal image analysis, we won't be able to read whole set of images into `R`'s memory. Instead, we will specify a function that will read images from somewhere on your hard drive. You can also experiment with other methods---as long as you can specify a function that returns an image when given the appropriate `imageKeysOfUnits` value, you should be fine. See [`tutorials/AnalyzeImageHeterogeneity_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_Tutorial.R) for a full example. 

## Analyzing the Sample Data 
Now that we've established some understanding of the data and written the `acquireImageFxn`, we are ready to proceed with the initial use of the causal image decomposition. 

*Note: The images used here are heavily clipped to keep this tutorial fast; the model parameters chosen here are selected to make training rapid too. The function output here should therefore not be interpreted too seriously.* 

```
ImageHeterogeneityResults <- causalimages::AnalyzeImageHeterogeneity(
          # data inputs
          obsW =  obsW,
          obsY = obsY,
          imageKeysOfUnits =  KeysOfObservations,
          file = "~/Downloads/CausalIm.tfrecord", # this points to the tfrecord
          X = X, 
          
          # inputs to control where visual results are saved as PDF or PNGs 
          # (these image grids are large and difficult to display in RStudio's interactive mode)
          plotResults = T,
          figuresPath = "~/Downloads/CausalImagesTutorial",

          # other modeling options
          kClust_est = 2,
          nSGD = 400L, # make this larger for full applications
          batchSize = 16L)
```
## Visual Results 
Upon completion, `AnalyzeImageHeterogeneity` will save several images from the analysis to the location `figuresPath`. The `figuresTag` will be appended to these images to keep track of results from different analyses. Currently, these images include the following: 
- The image results with .pdf name starting, `VisualizeHeteroReal_variational_minimal_uncertainty`, which plots the images having great uncertainty in the cluster probabilities. 
- The image results with .pdf name starting, `VisualizeHeteroReal_variational_minimal_mean`; these plots display the images having the highest probabilities for each associated cluster. 
- Finally, one output .pdf has name starting  `HeteroSimTauDensityRealDataFig`, and plots the estimated distributions over image-level treatment effects for the various clusters. Overlap of these distributions is to be expected, since the quantity is computed at the image (not some aggregate) level.

## Numerical Results
We can also examine some of the numerical results contained in the `ImageHeterogeneityResults` output. 
```
# image type treatment effect cluster means 
ImageHeterogeneityResults$clusterTaus_mean

# image type treatment effect cluster standard deviations
ImageHeterogeneityResults$clusterTaus_sd

# per image treatment effect cluster probability means 
ImageHeterogeneityResults$clusterProbs_mean

# per image treatment effect cluster probability standard deviations
ImageHeterogeneityResults$clusterProbs_sd
```

## Pointers 
Here are a few tips for using the `AnalyzeImageHeterogeneity` function: 
- If the cluster probabilities are very extreme (all 0 or 1), try increasing `nSGD`, simplifying the model structure (e.g., making `nFilters`, `nDepthHidden_conv`, or `nDepthHidden_dense` smaller), or increasing the number of Monte Carlo interations in the Variational Inference training (increase `nMonte_variational`).
- For satellite data, images that show up as pure dark blue are centered around a body of water.

# Acknowledgements
We thank [James Bailie](https://jameshbailie.github.io/), [Devdatt Dubhashi](https://sites.google.com/view/devdattdubhashi/home), [Felipe Jordan](http://www.felipejordanc.com/), [Mohammad Kakooei](https://www.chalmers.se/en/persons/kakooei/), [Eagon Meng](https://independent.academia.edu/EagonMeng), [Xiao-Li Meng](https://statistics.fas.harvard.edu/people/xiao-li-meng), and [Markus Pettersson](https://www.chalmers.se/en/persons/markpett/) for valuable feedback on this project. We thank [Xiaolong Yang](https://xiaolong-yang.com/) for excellent research assistance. Special thanks to [Cindy Conlin](https://www.linkedin.com/in/cindy-conlin-540197/) for being our intrepid first package user and for many invaluable suggestions for improvement. 

# References<a id="references"></a>
[1.] [Connor T. Jerzak](https://github.com/cjerzak/), [Fredrik Johansson](https://github.com/fredjoha), [Adel Daoud](https://github.com/adeldaoud). Image-based Treatment Effect Heterogeneity. *Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)*, 213: 531-552, 2023. [\[Article PDF\]](https://proceedings.mlr.press/v213/jerzak23a/jerzak23a.pdf) [\[Summary PDF\]](https://connorjerzak.com/wp-content/uploads/2023/04/ImageHeterogeneitySummary.pdf)  [\[Replication Data\]](https://doi.org/10.7910/DVN/O8XOSF) [\[Replication Data Tutorial\]](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_Tutorial.R) 

```
@article{JJD-Heterogeneity,
  title={Image-based Treatment Effect Heterogeneity},
  author={Jerzak, Connor T. and Fredrik Johansson and Adel Daoud},
  journal={Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)},
  year={2023},
  volume={213},
  pages={531-552}
}
```

[2.] [Connor T. Jerzak](https://github.com/cjerzak/), [Fredrik Johansson](https://github.com/fredjoha), [Adel Daoud](https://github.com/adeldaoud). Integrating
Earth Observation Data into Causal Inference: Challenges and
Opportunities. *ArXiv
Preprint*, 2023. [`arxiv.org/pdf/2301.12985.pdf`](https://arxiv.org/pdf/2301.12985.pdf)
[\[Replication Data\]](https://doi.org/10.7910/DVN/QLCSVR)
```
@article{JJD-Confounding,
  title={Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities},
  author={Jerzak, Connor T. and Fredrik Johansson and Adel Daoud},
  journal={ArXiv Preprint},
  year={2023},
  volume={},
  pages={},
  publisher={}
}
```

[3.] [Connor T. Jerzak](https://github.com/cjerzak/), [Adel Daoud](https://github.com/adeldaoud). CausalImages: An R Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images. *ArXiv Preprint*, 2023. [`arxiv.org/pdf/2301.12985.pdf`](https://arxiv.org/pdf/2310.00233.pdf)
```
@article{JerDao2023,
  title={CausalImages: An R Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images},
  author={Jerzak, Connor T. and Adel Daoud},
  journal={ArXiv Preprint},
  year={2023},
  volume={},
  pages={},
  publisher={}
}
```


<!---
[<img src="https://i0.wp.com/connorjerzak.com/wp-content/uploads/2024/08/EO_WorkflowVizV52.png?resize=1187%2C1536&ssl=1">](https://connorjerzak.com/gci-overview/)
--->

```

                     ██████  ██████  ██
                     ██  ██  ██      ██
                     ██████  ██      ██
                     ██      ██      ██
                     ██      ██████  ██

                         PLANETARY
                      CAUSAL INFERENCE

```
