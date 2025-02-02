Replication Data for: Image-based Treatment Effect Heterogeneity 

Connor Thomas Jerzak, Fredrik Daniel Johansson, Adel Daoud Proceedings of the Second Conference on Causal Learning and Reasoning, PMLR 213:531-552, 2023.

UgandaDataProcessed.csv contains individual-level data from the YOP experiment. In the dataset, geo_long and geo_lat refer to the approximate geo-referenced long/lat of experimental units. The variable, geo_long_lat_key, refers to the image key associated with each location. Experimental outcomes are stored in Yobs. Treatment variable is stored in Wobs. See the tutorial for more information. 

UgandaGeoKeyMat.csv contains information on keys linking to satellite images for all of Uganda for the transportability analysis. 

Geo-referenced satellite images are saved in "./Uganda2000_processed/GeoKey%s_BAND%s.csv", where GeoKey%s denotes the image key associated with each observation and BAND%s refers to one of 3 bands in the satellite imagery.

For more information, see: https://github.com/cjerzak/causalimages-software/blob/main/tutorials/AnalyzeImageHeterogeneity_FullTutorial.R 
