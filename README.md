# `causalimages`: An R Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images

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
| [**Image/Video Embeddings**](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/GetImageEmbeddings_Tutorial.R) | [**Speeding Up Code**](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/UsingTfRecords_Tutorial.R)

Replication data: 
[**Heterogeneity Paper**](https://doi.org/10.7910/DVN/O8XOSF) | 
[**De-confounding Paper**](https://doi.org/10.7910/DVN/QLCSVR)

_Beta package version:_ [`GitHub.com/cjerzak/causalimages-software`](https://github.com/cjerzak/causalimages-software)

_January 2024 update:_ `causalimages` now uses a [JAX](https://en.wikipedia.org/wiki/Google_JAX) backend for improved causal image and image-sequence model performance. 

# What is causalimages?<a id="description"></a>

Causal inference has entered a new stage where novel data sources are being integrated into the study of cause and effect. Image information is a particularly promising data stream in this context: it widely available and richly informative in social science and bio-medical contexts. 

This package, `causalimages`, enables causal analysis with images. For example, the function, `AnalyzeImageHeterogeneity`, performs the image-based treatment effect heterogeneity decomposition described in [Jerzak, Johansson, and Daoud (2023)](https://proceedings.mlr.press/v213/jerzak23a/jerzak23a.pdf). This function can be used, for example, to determine which neighorhoods are most responsive to an anti-poverty intervention using earth observation data from, e.g., satellites. In the bio-medical domain, this function could be used to model the kinds of patients who would be most responsive to interventions on the basis of pre-treatment diagnostic imaging. See [References](#references) for a link to replication data for the image heterogeneity paper; see [this tutorial](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_FullTutorial.R) for a walkthrough using the replication data. 

The function, `AnalyzeImageConfounding`, performs the image-based deconfounding analysis described in [Jerzak, Johansson, and Daoud (2023+)](https://arxiv.org/pdf/2301.12985.pdf). This function can be used, for example, to control for confounding factors correlated with both neighorhood wealth and aid decisions in observational studies of development. In the bio-medical context, this function could be used to control for confounding variables captured diagnostic imaging in order to improve observational inference.

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

*1. Build package backend.* This establishes the necessary modules, including JAX and Equinox, used in the causal image modeling. We attempt to establish GPU acceleration where that hardware is available. For tutorial, see [`tutorials/BuildBackend_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/BuildBackend_Tutorial.R) for more information. You can try using `conda="auto"` or finding the correct paty to the conda executable by typing ``where conda`` in the terminal: 
```
causalimages::BuildBackend(conda = "/Users/cjerzak/miniforge3/bin/python")
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

*4. Perform causal image analysis.* Finally, you may also want to perform a causal analysis using the image or image sequence data. For a tutorial on image-based treatment effect heterogeneity, see [`tutorials/AnalyzeImageHeterogeneity_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_Tutorial.R). For a tutorial on image-based confounding analysis, see [`tutorials/AnalyzeImageConfounding_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageConfounding_Tutorial.R). 
```
# for help, see also: 
?causalimages::AnalyzeImageHeterogeneity
?causalimages::AnalyzeImageConfounding
``` 

# Image Heterogeneity Tutorial<a id="tutorial"></a>
The most up-to-date tutorials are found in the [`tutorials`](https://github.com/cjerzak/causalimages-software/tree/main/tutorials) folder of this GitHub. Here, we also provide an abbreviated tutorial using an image heterogeneity analysis. 

## Load in Tutorial Data
After we've loaded in the package, we can get started running an analysis. We'll start by loading in tutorial data: 
```
data(  causalimagesTutorialData )
```
Once we've read in the data, we can explore its structure: 
```
# outcome, treatment, and covariate information: 
summary( obsW ) # treatment vector 
summary( obsY ) # outcome vector 
summary( LongLat ) # long-lat coordinates for each unit
summary( X ) # other covariates 

# image information: 
dim( FullImageArray ) # dimensions of the full image array in memory 
head( KeysOfImages ) # image keys associated with the images in FullImageArray
head( KeysOfObservations ) # image keys of observations to be associated to images via KeysOfImages
```
We can also analyze the images that we'll use in this analysis. 
```
# plot the second band of the third image
causalimages::image2(FullImageArray[3,,,2])

# plot the first band of the first image
causalimages::image2(FullImageArray[1,,,1])
```
We're using rather small image bricks around each long/lat coordinate so that this tutorial code is memory efficient. In practice, your images will be larger and you'll usually have to read them in from desk (with those instructions outlined in the `acquireImageFxn` function that you'll specify). We have an example of that approach later in the tutorial. 

## Writing the `acquireImageFxn`
One important part of the image analysis pipeline is writing a function that acquires the appropriate image data for each observation. This function will be fed into the `acquireImageFxn` argument of the package functions. There are two ways that you can approach this: (1) you may store all images in `R`'s memory, or you may (2) save images on your hard drive and read them in when needed. The second option will be more common for large images. 

You will write your `acquireImageFxn` to take in a single argument: `keys`.
- `keys` (a positional argument) is a character or numeric vector. Each value of `keys` refers to a unique image object that will be read in. If each observation has a unique image associated with it, perhaps `imageKeysOfUnits = 1:nObs`. In the example we'll use, multiple observations map to the same image. 


### When Loading All Images in Memory 
In this tutorial, we have all the images in memory in the `FullImageArray` array. We can write an `acquireImageFxn` function like so: 
```
acquireImageFromMemory <- function(keys, training = F){
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
image2( ImageSample[3,,,1] )
```

### When Reading in Images from Disk 
For most applications of large-scale causal image analysis, we won't be able to read whole set of images into `R`'s memory. Instead, we will specify a function that will read images from somewhere on your harddrive. You can also experiment with other methods---as long as you can specify a function that returns an image when given the appropriate `imageKeysOfUnits` value, you should be fine. See [`tutorials/AnalyzeImageHeterogeneity_Tutorial.R`](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_Tutorial.R) for a full example. 

## Analyzing the Sample Data 
Now that we've established some understanding of the data and written the `acquireImageFxn`, we are ready to proceed with the initial use of the causal image decomposition. 

*Note: The images used here are heavily clipped to keep this tutorial fast; the model parameters chosen here are selected to make training rapid too. The function output here should therefore not be interpreted too seriously.* 

```
ImageHeterogeneityResults <- AnalyzeImageHeterogeneity(
          # data inputs
          obsW =  obsW,
          obsY = obsY,
          imageKeysOfUnits =  KeysOfObservations,
          X = X, 
          
          # inputs to control where visual results are saved as PDF or PNGs 
          # (these image grids are large and difficult to display in RStudio's interactive mode)
          plotResults = T,
          figuresPath = "~/Downloads/CausalImagesTutorial",
          figuresTag = "CausalImagesTutorial",

          # other modeling options
          kClust_est = 2,
          nSGD = 400L, # make this larger for real applications
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
We thank [James Bailie](https://jameshbailie.github.io/), [Cindy Conlin](https://www.linkedin.com/in/cindy-conlin-540197/), [Devdatt Dubhashi](https://sites.google.com/view/devdattdubhashi/home), [Felipe Jordan](http://www.felipejordanc.com/), [Mohammad Kakooei](https://www.chalmers.se/en/persons/kakooei/), [Eagon Meng](https://independent.academia.edu/EagonMeng), [Xiao-Li Meng](https://statistics.fas.harvard.edu/people/xiao-li-meng), and [Markus Pettersson](https://www.chalmers.se/en/persons/markpett/) for valuable feedback on this project. We also thank [Xiaolong Yang](https://xiaolong-yang.com/) for excellent research assistance.

# References<a id="references"></a>
[1.] Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Image-based Treatment Effect Heterogeneity. *Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)*, 213: 531-552, 2023. [\[Article PDF\]](https://proceedings.mlr.press/v213/jerzak23a/jerzak23a.pdf) [\[Summary PDF\]](https://connorjerzak.com/wp-content/uploads/2023/04/ImageHeterogeneitySummary.pdf)  [\[Replication Data\]](https://www.dropbox.com/s/xy8xvva4i46di9d/Public%20Replication%20Data%2C%20YOP%20Experiment.zip?dl=0) [\[Replication Data Tutorial\]](https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_FullTutorial.R) [\[Dataverse\]](https://doi.org/10.7910/DVN/O8XOSF)

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

[2.] Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities. *ArXiv Preprint*, 2023. [`arxiv.org/pdf/2301.12985.pdf`](https://arxiv.org/pdf/2301.12985.pdf) [\[Dataverse\]](https://doi.org/10.7910/DVN/QLCSVR)
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

[3.] Connor T. Jerzak, Adel Daoud. CausalImages: An R Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images. *ArXiv Preprint*, 2023. [`arxiv.org/pdf/2301.12985.pdf`](https://arxiv.org/pdf/2310.00233.pdf)
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

[<img src="https://connorjerzak.com/wp-content/uploads/2023/03/pexels-photo-60132.jpeg" width="500" height="400">](https://proceedings.mlr.press/v213/jerzak23a.html)

