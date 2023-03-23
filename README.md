# `causalimages`: `R` Package for Causal Inference with Earth Observation, Bio-medical, and Social Science Images 

*This software is under beta release. The first stable release will be sometime in spring of 2023. Email `connor.jerzak@gmail.com` for feature requests.*

# Repository Structure
This repository contains two main components. The first component is code of the `causalimages` software package. At the moment, the package includes code for performing the image-based heterogeneity decomposition described in Jerzak, Johansson, and Daoud (2023). In future releases, we will add functionality for using images in observational causal inference (i.e., as a proxy for confounding variables). 

The second component is replication data. The replication data for the Youth Opportunities Experiment is found in `data/Replication Data/YOP Experiment`. This includes both outcome data and geo-referenced satellite images. We will keep adding to this replication repository with new research being performed over time. 


# Package Download Instructions 
From within `R`, you may download via the `devtools` package. In particular, use 

```
devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")
```

Then, to load the software, use 
```
library(   causalimages  ) 
```

# Tutorial
*Under construction.*
## Load in Tutorial Data
After we've loaded in the package, we can get started running an analysis. We'll start by loading in tutorial data: 
```
data(UgandaYOP)
```
Once we've read in the data, we can explore the structure of it a bit: 
```
head( Wobs ) # treatment vector 
head( Yobs ) # outcome vector 
head( X ) # long-lat coordinates for each unit
head( X ) # other covariates 
```
We can also analyze the images that we'll use in this analysis. Note that we're using rather small image bricks around each long/lat coordinate so that this tutorial code is memory efficient. In practice, your images will be larger and you'll usually have to read them in from desk (with those instructions outlined in the `acquireImageRepFxn` function that you'll specify). We have an example of that approach next. 

## Writing the `acquireImageRepFxn`
One important part of the image analysis pipeline is writing a function that acquires the appropriate image data for each observation. This function will be fed into the `acquireImageRepFxn` argument of the package functions. There are two ways that you can approach this: (1) you may store all images in `R`'s memory, or you may (2) save images on your harddrive and read them in when needed. The second option will be more common for large images. 

You will write your `acquireImageRepFxn` to take in two arguments: `keys` and `training` 
- `training` simplify specifies whether to treat the images as in training mode or inference mode (e.g., you may want to randomly flip images around their left-right axis during training, but not in inference mode). 
- `keys` is a character or numeric vector. Each value of `keys` refers to a unique image object that will be read in. If each observation has a unique image associated with it, perhaps `keys = 1:nObs`. In the example we'll use, multiple observations map to the same image. 

function(keys,training = F)

### When Storing All Images in Memory 


### When Reading in Images from Disk 
For most applications of causal image analysis, we won't be able to read whole set of images into `R`'s memory. Instead, we will specify a function that will read images from somewhere on your harddrive. You can also experiment with other methods---as long as you can specify a function that returns an image when given the appropriate `imageKeys` value, you should be fine. Here's an example of an `acquireImageRepFxn` that reads images from disk: 
```
acquireImageRepFromDisk <- function(keys,training = F){
  # IMPORTANT! This is illustration code only; it is not designed to run on your local computer 
  # initialize an array shell to hold image slices
  array_shell <- array(NA,dim = c(1L,imageHeight,imageWidth,NBANDS))

  # iterate over keys:
  # -- images are referenced to keys
  # -- keys are referenced to units (to allow for duplicate images uses)
  array_ <- sapply(keys,function(key_){
    # iterate over all image bands (NBANDS = 3 for RBG images)
    for(band_ in 1:NBANDS){
      # place the image in the correct place in the array
      array_shell[,,,band_] <-
        (as.matrix(data.table::fread( # note the use of data.table::fread to speed up reading in image to memory
          input = sprintf("./Data/Uganda2000_processed/GeoKey%s_BAND%s.csv",
                          key_,
                          band_),header = F)[-1,] ))
    }
    return( array_shell )
  },
  simplify="array")  #using simplify = "array" combines images slices together

  # convert images to tensorflow array for further processing
  array_ <- tf$squeeze(tf$constant(array_,dtype=tf$float32),0L)
  array_ <- tf$transpose(array_,c(3L,0L,1L,2L))
  return( array_ )
}
```
## Analyzing Tutorial Data


# Future Development Plan
We now have in beta release code for interpretably decomposing treatment effect heterogeneity by image. In the next stage, we will implement two more functionalities: (1) confounder adjustment via image and (2) causal image system simulation. Core machine learning modules are written in `tensorflow+tensorflow_probability`; subsequent versions may be transfered over to `equinox+oryx+jax`. 

We are committed to the long-term development of this repository and welcome community involvement. 

# Acknowledgements
We thank James Bailie, Cindy Conlin, Devdatt Dubhashi, Felipe Jordan, Mohammad Kakooei, Eagon Meng, Xiao-Li Meng, and Markus Pettersson for valuable feedback on this project and software. We also thank Xiaolong Yang for excellent research assistance.

# References
Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Image-based Treatment Effect Heterogeneity. *Forthcoming in Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)*, 2023. [`arxiv.org/pdf/2206.06417.pdf`](https://arxiv.org/pdf/2206.06417.pdf)

Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities. *ArXiv Preprint*, 2023. [`arxiv.org/pdf/2301.12985.pdf`](https://arxiv.org/pdf/2301.12985.pdf)
