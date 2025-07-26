CausalImageHeterogeneity_plot <- function(){
  message("Plotting results...")
  MPList <- list(cienv$jmp$Policy(compute_dtype="float32",
                            param_dtype="float32",
                            output_dtype="float32"),
                 cienv$jmp$DynamicLossScale(loss_scale = cienv$jnp$array(2^15,dtype = cienv$jnp$float32),
                                      min_loss_scale = cienv$jnp$array(1.,dtype = cienv$jnp$float32),
                                      period = 20L))
  if(heterogeneityModelType == "variational_minimal"){
    Tau_mean_vec_n <- as.numeric2( Tau_mean_vec )
    synth_seq <- seq(min(Tau_mean_vec_n - 2 * as.numeric2(Tau_sd_vec),Tau_mean_vec_n - 2 * as.numeric2(Tau_sd_vec)),
                     max(Tau_mean_vec_n + 2 * as.numeric2(Tau_sd_vec),Tau_mean_vec_n + 2 * as.numeric2(Tau_sd_vec)),
                     length.out=1000)
    
    if(truthKnown <- ("ClusterProbs" %in% ls(envir = globalenv()))){
      my_density <- density(ClusterProbs)
      my_density$y <- my_density$y
      synth_seq <- seq(min(my_density$x,na.rm=T),max(my_density$x,na.rm=T),length.out=100)
    }
    
    for(kr_ in 1:kClust_est){
      eval(parse(text = sprintf("d%s <- dnorm(synth_seq, mean = Tau_mean_vec_n[kr_], sd = Tau_sd_vec[kr_] )",  kr_)))
    }
    pdf(sprintf("%s/HeteroSimTauDensity%s_ExternalFigureKey%s.pdf",
                figuresPath, heterogeneityModelType, figuresTag))
    {
      par(mar=c(5,5,1,1))
      numbering_seq <- 1:kClust_est
      col_seq <- 1:kClust_est 
      if(!truthKnown){
        my_density <- eval(parse(text = sprintf("
                        data.frame('y'=max(c(%s)))
                        ", paste(paste("d",1:kClust_est,sep=""), collapse = ","))  ))
      }
      plot(my_density,
           col = ifelse(truthKnown,yes="darkgray",no="white"),
           lty = 2, cex = 0, lwd = 3,
           xlim = c(min(synth_seq),max(synth_seq)),
           ylim = c(0,max(my_density$y,na.rm=T)*1.5),
           cex.lab = 2, main = "",
           ylab = "Density",
           xlab = "Per Image Treatment Effect")
      if(!truthKnown){ axis(2,cex.axis = 1) }
      #points(tau_vec, rep(0,times=kClust_est),
      points(Tau_mean_vec_n, rep(0,times=kClust_est),
             col = col_seq,
             pch = "|", cex = 4)
      for(krk_ in 1:kClust_est){
        points(synth_seq,
               eval(parse(text = sprintf("d%s",krk_))),
               type = "l", col= col_seq[krk_],lwd = 3)
      }
      legends_seq_vec <- c(expression("True"~p(Y[i](1)-Y[i](0)~"|"~M[i])),
                           sapply(numbering_seq, function(numbering_){
                             eval(parse(text=sprintf('
              expression(hat(p)~"("~Y[i](1)-Y[i](0)~"|"~Z[i]==%s~")")',numbering_)))}))
      lty_seq_vec <- c(2,1,1)
      col_seq_vec <- c("gray",col_seq)
      if(!truthKnown){
        legends_seq_vec <- legends_seq_vec[-1]
        col_seq_vec <- col_seq_vec[-1]
        lty_seq_vec <- lty_seq_vec[-1]
      }
      legend("topleft", legend = legends_seq_vec,
             box.lwd = 0, box.lty = 0, cex = 2,
             lty = lty_seq_vec, col = col_seq_vec, lwd = 3)
    }
    dev.off()
    
    if(truthKnown){
      order_ <- order(ClusterProbs)
      if(cor(ClusterProbs, ClusterProbs_est) > 0){
        # we do this so the coloring stays consistent
        col_dim <- rank(ClusterProbs_est)#gtools::quantcut(ClusterProbs_est, q = 100)
      }
      if(cor(ClusterProbs, ClusterProbs_est) < 0){
        col_dim <- rank(-ClusterProbs_est)#gtools::quantcut(ClusterProbs_est, q = 100)
      }
      pdf(sprintf("%s/HeteroSimCluster_ExternalFigureKey%s.pdf",
                  figuresPath, figuresTag))
      {
        par(mar=c(5,5,1,1))
        plot( ClusterProbs[order_],
              1:length(order_)/length(order_),
              xlab = "Per Image Treatment Effect",
              ylab = "Empirical CDF(x)",pch = 19,
              col = viridis::magma(n=length(col_dim),alpha=0.9)[col_dim][order_],
              cex = 1.5, cex.lab = 2)
        legend("topleft",
               box.lwd = 0, box.lty = 0,
               pch = 19, #box.col = "white",
               col = c(viridis::magma(5)[2],
                       viridis::magma(5)[3],
                       viridis::magma(5)[4]),
               cex = 2,
               legend=c("Higher Clust 1 Prob.", "...", "Higher Clust 2 Prob."))
      }
      dev.off()
    }
  }
  
  par(mfrow=c(1,1))
  gc(); py_gc$collect()
  plotting_coordinates_list <- list(); typePlot_counter <- 0
  ep_LabelSmooth <- cienv$jnp$array(0.01)
  
  dLogProb_dImage <- cienv$jax$grad(LogProb_Image <- function(ModelList, ModelList_fixed, m, vseed, StateList, seed, MPList){
    if(SharedImageRepresentation){
      m <- cienv$jnp$squeeze( ImageRepArm_batch_R(ifelse(optimizeImageRep, yes = list(ModelList), no = list(ModelList_fixed))[[1]],
                                            cienv$jnp$expand_dims(m,0L),
                                            StateList, seed, MPList, T)[[1]] ) 
    }
    
    PROBS_ <- sapply(1L:nMonte_salience, function(itr){
      cienv$jnp$expand_dims(GetTau(ModelList, ModelList_fixed,
                             m, cienv$jnp$add(cienv$jnp$expand_dims(vseed,0L),itr),
                             StateList, cienv$jnp$add(seed,itr), MPList, T)[[1]],0L)
    })
    PROBS_ <- cienv$jax$nn$softmax( cienv$jnp$concatenate(PROBS_,0L) )
    PROBS_ <- cienv$jnp$add(cienv$jnp$multiply(cienv$jnp$subtract(cienv$jnp$array(1.), ep_LabelSmooth),PROBS_),
                      cienv$jnp$divide(ep_LabelSmooth, cienv$jnp$array(2.))) # label smoothing
    PROBS_ <- cienv$jnp$sum( cienv$jnp$multiply(cienv$jnp$array(c(1,rep(0,times = kClust_est-1))),
                                    cienv$jnp$mean(cienv$jnp$log(PROBS_), 0L) )) # # log prob of cat 1
    return(  PROBS_  ) # mean log prob
  }, 2L)
  ImGrad_fxn <- cienv$eq$filter_jit( function(ModelList, ModelList_fixed, m, vseed, StateList, seed, MPList){
    # cast to float32
    ModelList <- MPList[[1]]$cast_to_param( ModelList )
    ModelList_fixed <- MPList[[1]]$cast_to_param( ModelList_fixed )
    StateList <- MPList[[1]]$cast_to_param( StateList )
    m <- MPList[[1]]$cast_to_param( m )
    m <- cienv$jax$device_put(m, cienv$jax$devices('cpu')[[1]])
    
    ImageGrad_o <- dLogProb_dImage(ModelList, ModelList_fixed,
                                   m, vseed,
                                   StateList, seed, MPList)
    reduceDim <- ifelse( dataType == "video", yes = 3L, no = 2L)
    ImageGrad_L2 <- cienv$jnp$linalg$norm(ImageGrad_o+0.000001, axis = reduceDim, keepdims = T)
    ImageGrad_mean <- cienv$jnp$mean(ImageGrad_o, axis = reduceDim, keepdims = T)
    return( list(ImageGrad_L2,  # salience magnitude
                 ImageGrad_mean) ) # salience direction
    
  }, device = cienv$jax$devices('cpu')[[1]])
  video_plotting_fxn <- function(){
    message("Image seq results plot...")
    plotting_coordinates_mat <- c()
    total_counter <- 0;
    for(k_ in 1:rows_){
      used_coordinates <- c()
      for(i in 1:5){
        gc(); py_gc$collect()
        total_counter <- total_counter + 1
        rfxn <- function(xer){xer}
        bad_counter <- 0;isUnique_ <- F; while(isUnique_ == F){
          BreakTies <- function(x){x + runif(length(x),-1e-3,1e-3)}
          if(typePlot == "uncertainty"){
            main_ <- letters[  total_counter  ]
            
            # plot images with largest std's
            valBrokenTies <- BreakTies(ClusterProbs_std[,k_])
            sorted_unique_prob_k <- sort(rfxn(unique(valBrokenTies)),decreasing=T)
            im_i <- which(valBrokenTies == sorted_unique_prob_k[i+bad_counter])[1]
          }
          if(grepl(typePlot,pattern = "mean")){
            main_ <- total_counter
            
            # plot images with largest cluster probs
            if(typePlot ==  "mean"){
              valBrokenTies <- BreakTies(ClusterProbs_lower_conf[,k_])
              sorted_unique_prob_k <- sort(rfxn(unique(valBrokenTies)),decreasing=T)
              im_i <- which(valBrokenTies == sorted_unique_prob_k[i+bad_counter])[1]
            }
            
            # plot images with largest lower confidence
            if(typePlot ==  "mean_upperConf"){
              valBrokenTies <- BreakTies(ClusterProbs_est_full[,k_])
              sorted_unique_prob_k <- sort(rfxn(unique(valBrokenTies)),decreasing=T)
              im_i <- which(valBrokenTies == sorted_unique_prob_k[i+bad_counter])[1]
            }
          }
          
          coordinate_i <- c(long[im_i], lat[im_i])
          if(i > 1){
            isUnique_ <- F; if(!is.null(long)){
              dist_m <- geosphere::distm(x = coordinate_i,
                                         y = cbind(f2n(used_coordinates[,3]),
                                                   f2n(used_coordinates[,4])),
                                         fun = geosphere::distHaversine)
              bad_counter <- bad_counter + 1
              if(all(dist_m >= 1000)){isUnique_ <- T}
            }
            if(is.null(long)){
              bad_counter <- bad_counter + 1
              isUnique_ <- !( imageKeysOfUnits[im_i] %in% used_coordinates[,2] )
            }
          }
          if(i == 1){ isUnique_<-T }
        }
        used_coordinates <- rbind(used_coordinates,
                                  c("observation_index" = im_i,
                                    "key" = imageKeysOfUnits[im_i],
                                    "long" = coordinate_i[1],
                                    "lat" = coordinate_i[2]))
        
        # load in video
        setwd(orig_wd); ds_next_in <- GetElementFromTfRecordAtIndices(
          uniqueKeyIndices = which(unique(imageKeysOfUnits) %in% imageKeysOfUnits[im_i]),
          filename = file,
          readVideo = useVideoIndicator,
          image_dtype = image_dtype_tf,
          nObs = length(unique(imageKeysOfUnits)) ); setwd(new_wd)
        ds_next_in <- ds_next_in[[1]]
        if(length(ds_next_in$shape) == 4){ ds_next_in <- cienv$tf$expand_dims(ds_next_in, 0L) }
        
        PlotWithMean <- grepl(typePlot,pattern = "mean")
        animation::saveGIF({
          # panel 1 of GIF
          par(mfrow=c(1,1+1*PlotWithMean))
          orig_scale_im_raster <-  cienv$np$array( ds_next_in[1,,,,plotBands[1]] )
          nTimeSteps <- dim(ds_next_in[1,, , ,])[1]
          IG <- np$array(  ImGrad_fxn(ModelList,
                                      ModelList_fixed,
                                      InitImageProcessFn(cienv$jnp$array(ds_next_in),  cienv$jax$random$key(600L), inference = T)[0,,,],  # m
                                      cienv$jax$random$key(400L),
                                      StateList,
                                      cienv$jax$random$key(430L),
                                      MPList)[[1]] )
          for (t_ in 1L:nTimeSteps) {
            plotRBG <- !(length(plotBands) < 3 | dim(ds_next_in)[length(dim(ds_next_in))] < 3)
            par(mar = margins_gif <- c(5,5,3,1))
            if(!plotRBG){
              par(mar = margins_gif)
              causalimages::image2(
                as.matrix2( orig_scale_im_raster[t_,,] ),
                main = main_, cex.main = 4,
                cex.lab = 1.5, col.lab = k_, col.main = k_,
                xlab = ifelse(!is.null(long),
                              yes = sprintf("Long: %s, Lat: %s",
                                            fixZeroEndings(round(coordinate_i,2L)[1],2L),
                                            fixZeroEndings(round(coordinate_i,2L)[2],2L)),
                              no = ""))
              animation::ani.pause()  # Pause to make sure it gets rendered
            }
            if(plotRBG){
              orig_scale_im_raster <- raster::brick(
                0.0001 + (cienv$np$array(cienv$tf$cast(ds_next_in[1,t_, , ,plotBands], cienv$tf$float32) )) +
                  0*runif(length(cienv$np$array(cienv$tf$cast(ds_next_in[1,t_, , ,plotBands], cienv$tf$float32 ))),
                          min = 0, max = 0.01) ) # with random jitter
              stretch <- ifelse(
                any(apply(cienv$np$array(cienv$tf$cast(ds_next_in[1,t_, , ,plotBands],cienv$tf$float32)), 3,sd) < 1.),
                yes = "", no = "lin")
              raster::plotRGB(  orig_scale_im_raster,
                                margins = T,
                                r = 1, g = 2, b = 3,
                                main = main_,
                                cex.lab = 1.5, col.lab = k_,
                                xlab = ifelse(!is.null(long),
                                              yes = sprintf("Long: %s, Lat: %s",
                                                            fixZeroEndings(round(coordinate_i,2L)[1],2L),
                                                            fixZeroEndings(round(coordinate_i,2L)[2],2L)),
                                              no = ""),
                                col.main = k_, cex.main=4,  stretch = stretch)
            }
            
            # salience plotting routine
            if(PlotWithMean){
              ylab_ <- ""; if(i==1){
                tauk <- np$array(Tau_mean_vec)[k_]
                ylab_ <- eval(parse(text = sprintf("expression(hat(tau)[%s]==%.3f)",k_,tauk)))
                if(orthogonalize == T){
                  ylab_ <- eval(parse(text = sprintf("expression(hat(tau)[%s]^{phantom() ~ symbol('\136') ~ phantom()}==%.3f)",k_, tauk)))
                }
              }
              
              #obtain video gradient magnitudes
              {
                nColors <- 1000
                { #if(i == 1){
                  # pos/neg breaks should be on the same scale across observation
                  gradMag_breaks <- try(sort(quantile((c(IG)),probs = seq(0,1,length.out = nColors),na.rm=T)),T)
                  if("try-error" %in% class(gradMag_breaks)){gradMag_breaks <-  seq(-1, 1,length.out = nColors) }
                  if(!"try-error" %in% class(gradMag_breaks)){
                    if(any(is.infinite(gradMag_breaks))){
                      gradMag_breaks <-  seq(-1, 1,length.out = nColors)
                    }} }
                
                # panel 2 - magnitude
                par(mar = margins_gif)
                image(t(IG[t_,,,1])[,nrow(IG[t_,,,1]):1L],
                      col = viridis::magma(nColors - 1),
                      main = main_, cex.main = 1.5, col.main = "white",
                      ylab = "",cex.axis = 3,
                      cex.axis = (cex_tile_axis <- 4), col.axis=k_,
                      breaks = gradMag_breaks, axes = F)
                animation::ani.pause()  # Pause to make sure it gets rendered
              }}
          }
        },
        movie.name = sprintf("%s/HeteroSimCluster_ExternalFigureKey%s_Type%s_Ortho%s_k%s_i%s.gif",
                             figuresPath,  figuresTag, typePlot, orthogonalize, k_, i),
        ani.height = 480*1, ani.width = 480*(1+PlotWithMean),
        autobrowse = F, autoplay = F)
      }
      plotting_coordinates_mat <- try(rbind(plotting_coordinates_mat, used_coordinates ),T)
    }
    return( plotting_coordinates_mat )
  }
  image_plotting_fxn <- function(){
    message("Image results plot...")
    pdf(sprintf("%s/VisualizeHeteroReal_%s_%s_%s_ExternalFigureKey%s.pdf", figuresPath, heterogeneityModelType,typePlot,orthogonalize,figuresTag),
        height = ifelse(grepl(typePlot,pattern = "mean"), yes = 4*rows_*3, no = 4),
        width = 4*nExamples)
    {
      #a
      if(grepl(typePlot,pattern = "mean")){
        par(mar=c(2, 5.9, 3, 0.5))
        layout_mat_orig <- layout_mat <- matrix(c(1:nExamples*3-3+1,
                                                  1:nExamples*3-1,
                                                  1:nExamples*3), nrow = 3, byrow = T)
        for(kr_ in 2:kClust_est){
          layout_mat <- rbind(layout_mat,
                              layout_mat_orig+max(layout_mat))
        }
      }
      if(typePlot %in% c("uncertainty")){
        par(mar=c(2,2,4,2)); layout_mat <- t(1:nExamples)
      }
      layout(mat = layout_mat,
             widths = rep(2,ncol(layout_mat)),
             heights = rep(2,nrow(layout_mat)))
      plotting_coordinates_mat <- c()
      total_counter <- 0
      for(k_ in 1:rows_){
        used_coordinates <- c()
        for(i in 1:5){
          gc(); py_gc$collect()
          #if(k_ == 2 & typePlot == "mean"){ }
          #if(k_ == 2 & i == 1){  }
          message(sprintf("Type Plot: %s; k_: %s, i: %s", typePlot, k_, i))
          total_counter <- total_counter + 1
          rfxn <- function(xer){xer}
          bad_counter <- 0;isUnique_ <- F; while(isUnique_ == F){
            BreakTies <- function(x){x + runif(length(x),-1e-3,1e-3)}
            if(typePlot == "uncertainty"){
              main_ <- letters[  total_counter  ]
              
              # plot images with largest std's
              valBrokenTies <- BreakTies(ClusterProbs_std[,k_])
              sorted_unique_prob_k <- sort(rfxn(unique(valBrokenTies)),decreasing=T)
              im_i <- which(valBrokenTies == sorted_unique_prob_k[i+bad_counter])[1]
            }
            if(grepl(typePlot,pattern = "mean")){
              main_ <- total_counter
              
              # plot images with largest lower confidence
              if(typePlot ==  "mean"){
                valBrokenTies <- BreakTies(ClusterProbs_lower_conf[,k_])
                sorted_unique_prob_k <- sort(rfxn(unique(valBrokenTies)),decreasing=T)
                im_i <- which(valBrokenTies == sorted_unique_prob_k[i+bad_counter])[1]
              }
              
              # plot images with largest cluster probs
              if(typePlot ==  "mean_upperConf"){
                valBrokenTies <- BreakTies(ClusterProbs_est_full[,k_])
                sorted_unique_prob_k <- sort(rfxn(unique(valBrokenTies)),decreasing=T)
                im_i <- which(valBrokenTies == sorted_unique_prob_k[i+bad_counter])[1]
              }
            }
            
            coordinate_i <- c(long[im_i], lat[im_i])
            if(  i > 1  ){
              isUnique_ <- F; if(!is.null(long)){
                dist_m <- geosphere::distm(x = coordinate_i,
                                           y = cbind(f2n(used_coordinates[,3]),
                                                     f2n(used_coordinates[,4])),
                                           fun = geosphere::distHaversine)
                bad_counter <- bad_counter + 1
                if(all(dist_m >= 1000)){isUnique_ <- T}
              }
              if(is.null(long)){
                bad_counter <- bad_counter + 1
                isUnique_ <- !( imageKeysOfUnits[im_i] %in% used_coordinates[,2] )
              }
            }
            if(i == 1){ isUnique_<-T }
          }
          
          used_coordinates <-  rbind(used_coordinates,
                                     c("observation_index"=im_i,
                                       "key"=imageKeysOfUnits[im_i],
                                       "long" = coordinate_i[1],
                                       "lat" = coordinate_i[2]))
          message(sprintf("k: %i i: %i, im_i: %i, long/lat: %.3f, %.3f",
                         as.integer(k_), as.integer(i), as.integer(im_i),
                         long[im_i], lat[im_i]))
          # load in image
          setwd(orig_wd); ds_next_in <- GetElementFromTfRecordAtIndices(
            uniqueKeyIndices = which(unique(imageKeysOfUnits) %in% imageKeysOfUnits[im_i]),
            filename = file,
            readVideo = useVideoIndicator,
            image_dtype = image_dtype_tf,
            nObs = length(unique(imageKeysOfUnits)) ); setwd(new_wd)
          ds_next_in <- cienv$jnp$array(  ds_next_in[[1]] )
          if(length(ds_next_in$shape) == 3 ){ ds_next_in <- cienv$jnp$expand_dims(ds_next_in, 0L) }
          
          if(length(plotBands) < 3){
            orig_scale_im_raster <-  np$array(ds_next_in)[1,,,plotBands[1]]
            causalimages::image2(
              as.matrix2( orig_scale_im_raster ),
              main = main_, cex.main = 4,
              cex.lab = 2.5, col.lab = k_, col.main = k_,
              xlab = ifelse(!is.null(long),
                            yes = sprintf("Long: %s, Lat: %s",
                                          fixZeroEndings(round(coordinate_i,2L)[1],2L),
                                          fixZeroEndings(round(coordinate_i,2L)[2],2L)), no = ""))
          }
          if(length(plotBands) >= 3){
            orig_scale_im_raster <- raster::brick( 0.0001 +
                                                     0*runif(length(np$array(ds_next_in)[1, , ,plotBands]), min = 0, max = 0.01) + # random jitter
                                                     (np$array(ds_next_in)[1,,,plotBands]))
            stretch <- ifelse( any(apply(np$array(ds_next_in)[1, , ,plotBands], 3,sd) < 1.), yes = "", no = "lin")
            raster::plotRGB(  orig_scale_im_raster, margins = T,
                              mar = (margins_vec <- (ep_<-1e-6)*c(1,3,1,1)),
                              main = main_,
                              cex.lab = 2.5, col.lab = k_,
                              xlab = ifelse(!is.null(long),
                                            yes = sprintf("Long: %s, Lat: %s",
                                                          fixZeroEndings(round(coordinate_i,2L)[1],2L),
                                                          fixZeroEndings(round(coordinate_i,2L)[2],2L)),
                                            no = ""),
                              col.main = k_, cex.main=4,  stretch = stretch)
          }
          if(grepl(typePlot,pattern = "mean")){
            ylab_ <- ""; if(i==1){
              tauk <- np$array(Tau_mean_vec)[k_]
              ylab_ <- eval(parse(text = sprintf("expression(hat(tau)[%s]==%.3f)",k_,tauk)))
              if(orthogonalize == T){
                ylab_ <- eval(parse(text = sprintf("expression(hat(tau)[%s]^{phantom() ~ symbol('\136') ~ phantom()}==%.3f)",k_, tauk)))
              }
              axis(side = 2,at=0.5,labels = ylab_,pos=-0.,tick=F,cex.axis=cex_tile_axis <- 4,
                   col.axis=k_)
            }
            
            #obtain image gradients
            {
              gc(); py_gc$collect()
              IG <- ImGrad_fxn(ModelList, ModelList_fixed,
                               InitImageProcessFn(cienv$jnp$array(ds_next_in), cienv$jax$random$key(600L), inference = T)[0,,,],  # m
                               cienv$jax$random$key(400L),
                               StateList,
                               cienv$jax$random$key(430L),
                               MPList)
              IG[[1]] <- np$array(IG[[1]])[,,1] # magnitude
              IG[[2]] <- np$array(IG[[2]])[,,1] # direction
              
              { #if(i == 1){
                nColors <- 1000
                IG_forBreaks <- IG[[1]] + runif(length(IG[[1]]),-0.0000000000001, 0.0000000000001)
                gradMag_breaks <- try(sort(quantile((c(IG_forBreaks)),probs = seq(0,1,length.out = nColors),na.rm=T)),T)
                if("try-error" %in% class(gradMag_breaks)){gradMag_breaks <-  seq(-1, 1,length.out=nColors) }
                
                # pos/neg breaks should be on the same scale across observation
                IG_forBreaks <- IG[[2]] + runif(length(IG[[2]]),-0.0000000000001, 0.0000000000001)
                pos_breaks <- try(sort( quantile(c(IG_forBreaks[IG_forBreaks>=0]),probs = seq(0,1,length.out=nColors/2),na.rm=T)),T)
                if("try-error" %in% class(pos_breaks)){pos_breaks <-  seq(0, 1,length.out=nColors/2) }
                
                neg_breaks <- try(sort(quantile(c(IG_forBreaks[IG_forBreaks<=0]),probs = seq(0,1,length.out=nColors/2),na.rm=T)),T)
                if("try-error" %in% class(neg_breaks)){neg_breaks <-  seq(-1, 0,length.out=nColors/2) }
              }
              
              # magnitude
              magPlot <- image(t(IG[[1]])[,nrow(IG[[1]]):1],
                               col = viridis::magma(nColors - 1),
                               breaks = gradMag_breaks, axes = F)
              if("try-error" %in% class(magPlot)){ message("magPlot broken") }
              ylab_ <- ""; if(i==1){
                try(axis(side = 2,at=0.5,labels = "Salience Magnitude",
                         pos=-0.,tick=F, cex.axis=3, col.axis=k_),T)
              }
              
              # direction
              dirPlot <- image(t(IG[[2]])[,nrow(IG[[2]]):1],
                               col = c(hcl.colors(nColors/2-1L,"reds"),
                                       hcl.colors(nColors/2 ,"blues")),
                               breaks = c(neg_breaks,pos_breaks), axes = F)
              if("try-error" %in% class(dirPlot)){message("dirPlot broken")}
              ylab_ <- ""; if(i==1){
                try( axis(side = 2,at=0.5,labels = "Salience Direction",
                          pos=-0.,tick=F, cex.axis=3, col.axis=k_),  T)
              }
            }
          }
        }
        plotting_coordinates_mat <- rbind(plotting_coordinates_mat, used_coordinates )
      }
    }
    dev.off()
    return( plotting_coordinates_mat )
  }
  for(typePlot in (typePlot_vec <- c("mean","uncertainty"))){ 
    typePlot_counter <- typePlot_counter + 1
    rows_ <- kClust_est; nExamples <- 5
    if(typePlot == "uncertainty"){ rows_ <- 1L }
    
    message("Starting plot routine...")
    plotting_coordinates_mat_ <- ifelse(dataType == "video", yes = list(video_plotting_fxn), no = list(image_plotting_fxn))[[1]]()
    
    if("try-error" %in% class(plotting_coordinates_mat_)){
      message('if("try-error" %in% class(plotting_coordinates_mat_)){')
      browser()
    }
    try(dev.off(),T)
    plotting_coordinates_list[[typePlot_counter]] <- plotting_coordinates_mat_
  }
  try({ names(plotting_coordinates_list) <- typePlot_vec},T)
  par(mfrow=c(1,1))
}