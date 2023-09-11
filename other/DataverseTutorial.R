#!/usr/bin/env Rscript

################################
# For an up-to-date version of this tutorial, see
# https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_FullTutorial.R
################################

# set new wd
setwd(sprintf('%s/Public Replication Data, YOP Experiment/',
              gsub(download_folder,pattern="\\.zip",replace="")))

# see directory contents
list.files()

# images saved here
list.files(  "./Uganda2000_processed"  )

# individual-level data
UgandaDataProcessed <- read.csv(  "./UgandaDataProcessed.csv"  )

# unit-level covariates (many covariates are subject to missingness!)
dim( UgandaDataProcessed )
table( UgandaDataProcessed$female )
table( UgandaDataProcessed$age )

# approximate longitude + latitude for units
UgandaDataProcessed$geo_long
UgandaDataProcessed$geo_lat

# image keys of units (use for referencing satellite images)
UgandaDataProcessed$geo_long_lat_key

# an experimental outcome
UgandaDataProcessed$Yobs

# treatment variable
UgandaDataProcessed$Wobs

# information on keys linking to satellite images for all of Uganda
# (not just experimental context, use for constructing transportability maps)
UgandaGeoKeyMat <- read.csv(  "./UgandaGeoKeyMat.csv"  )
tail( UgandaGeoKeyMat )

# Geo-referenced satellite images are saved in
# "./Uganda2000_processed/GeoKey%s_BAND%s.csv",
# where GeoKey%s denotes the image key associated with each observation and
# BAND%s refers to one of 3 bands in the satellite imagery.
# See https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_FullTutorial.R
# for up-to-date useage information.
