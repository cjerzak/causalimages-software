Replication Data for:  Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities 

Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities. ArXiv Preprint, 2023.

YandW_mat.csv contains individual-level observational data. In the dataset, LONGITUDE and LATITUDE refer to the approximate geo-referenced long/lat of observational units.  Experimental outcomes are stored in Yobs. Treatment variable is stored in Wobs. See the tutorial for more information. The unique image key for each observational unit is saved in UNIQUE_ID.

Geo-referenced satellite images are saved in
"./Nigeria2000_processed/%s_BAND%s.csv"", where the first "%s" refers to the the image key associated with each observation (saved in UNIQUE_ID in YandW_mat.csv) and BAND%s refers to one of 3 bands in the satellite imagery.

For more information, see: https://github.com/cjerzak/causalimages-software/
