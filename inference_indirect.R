directory="~/DP2LM/testing"
setwd(directory)#main directory

lam=c(0.3,0.5,0.7,0.9,1.1,1.3,1.5,2.5,3.5,5)#tuning candidates for SCAD penalty, which should be adjusted by users
a=3.7
#write the SCAD peanlty function as '.r' file
for(i in 1:length(lam)){
  sink(paste0("scad",i,".R"))
  cat("scad=function(weights){")
  cat("\n")
  cat(paste0("lam=", lam[i]))
  cat("\n")
  cat("out1=(lam * tf$reduce_sum(tf$math$abs(weights)))*tf$reduce_sum(tf$cast(tf$less_equal(tf$abs(weights), lam),tf$float32))")
  cat("\n")
  cat(paste0("out2=(2*",a,"*lam*tf$reduce_sum(tf$math$abs(weights)) - tf$reduce_sum(tf$math$square(weights)) - (lam)^2)/(2*(",a,"-1))*tf$reduce_sum(tf$cast(tf$greater(tf$abs(weights), lam), tf$float32))*tf$reduce_sum(tf$cast(tf$less_equal(tf$abs(weights),",a,"*lam), tf$float32))"))
  cat("\n")
  cat(paste0("out3=lam^2*(",a,"+1)/2*tf$reduce_sum(tf$cast(tf$greater(tf$abs(weights),",a,"*lam), tf$float32))"))
  cat("\n")
  cat("return(out1+out2+out3)")
  cat("\n")
  cat("}")
  sink()
}



#network parameters for outcome-mediator model
length=2; width=50; dropout=0.99
epoch=1000; batch=4
#number of tuning candidates for regularization
lam_length=length(lam)
#training:testing ratio for total model network structure selection
split=0.7
#network parameters for total model
length.ttl=c(2,3); width.ttl=c(3,5,10); dropout.ttl=c(0.9,0.95, 0.99)
epoch.ttl=3000; batch.ttl=4
#network parameters for constructing testing statistic
length.test=2; width.test=50; dropout.test=0.99
epoch.test=1000; batch.test=4


DP2LM_IE_test <- function(data,length,width,dropout,epoch,batch,directory,lam_length,split,length.ttl,width.ttl,dropout.ttl,epoch.ttl,batch.ttl,length.test,width.test,dropout.test,epoch.test,batch.test) {
  
  library(keras)
  library(tensorflow)
  library(reticulate)
  use_python(paste(Sys.getenv("CONDA_PREFIX"), "bin/python", sep="/"))
  
  
  
  
  
  x=data$x
  M=data$M
  z=data$z
  y=data$y
  
  p=dim(M)[2]#dim for mediator
  q=dim(x)[2]#dim for exposure variable
  r=dim(z)[2]#dim for confounder
  n=dim(y)[1]#sample size
  
  t1=Sys.time()
  
  #####################################################
  ###############Outcome-mediator model################
  #####################################################
  
  
  
  #Training sets including M,x,z
  x.train1=list()
  
  for(j in 1:p){
    x.train1[[j]]=M[,j]
  }
  for(j in 1:q){
    x.train1[[p+j]]=x[,j]
  }
  x.train1[[p+q+1]]=z
  y.train1=y
  
  
  
  HBIC=df=c()
  
  
  for(lam in 1:lam_length){
    
    source(paste0(directory,"/scad",lam,".R"))
    
    
    #Linear effect for M
    inputs1=outputs1=list()
    for(i in 1:p){
      inputs1[[i]] <- layer_input(shape = shape(1), name = paste0("mediator",i))
      outputs1[[i]] <- layer_dense(inputs1[[i]], units = 1, use_bias = FALSE, activation = NULL, kernel_regularizer = scad, kernel_initializer = "zero")
    }
    #Linear effect for x
    for(i in 1:q){
      inputs1[[i+p]] <- layer_input(shape = shape(1), name = paste0("exposure",i+p))
      outputs1[[i+p]] <- layer_dense(inputs1[[i+p]], units = 1, use_bias = FALSE, activation = NULL, kernel_initializer = "normal")
    }
    
    if(length==2){
      input.z <- layer_input(shape = shape(r), name = "confounder")
      z1 <- layer_dense(input.z, width, activation = "relu", name = "layer_1")
      z1.drop <- layer_dropout(z1, rate=dropout)
      z2 <- layer_dense(z1.drop, width, activation = "relu", name = "layer_2")
      z2.drop <- layer_dropout(z2, rate=dropout)
      inputs1[[p+q+1]]=input.z
      outputs1[[p+q+1]] <- layer_dense(z2.drop, 1, name = "output1")
    }else if(length==3){
      input.z <- layer_input(shape = shape(r), name = "confounder")
      z1 <- layer_dense(input.z, width[pp], activation = "relu", name = "layer_1")
      z1.drop <- layer_dropout(z1, rate=dropout)
      z2 <- layer_dense(z1.drop, width[pp], activation = "relu", name = "layer_2")
      z2.drop <- layer_dropout(z2, rate=dropout)
      z3 <- layer_dense(z2.drop, width[pp], activation = "relu", name = "layer_3")
      z3.drop <- layer_dropout(z3, rate=dropout)
      inputs1[[p+q+1]]=input.z
      outputs1[[p+q+1]] <- layer_dense(z3.drop, 1, name = "output1")
    }
    
    
    
    #Add all outputs
    layer.out1=layer_add(outputs1)
    model.all1 <- keras_model(inputs = inputs1, outputs = layer.out1)
    model.all1 %>% compile(
      loss = 'mse',
      optimizer = optimizer_adam(),
      metrics = c('mse')
    )
    
    
    
    
    
    history <- model.all1 %>% fit(
      x=x.train1,
      y=y.train1,
      epochs = epoch, batch_size = batch
    )
    
    
    #get weights for mediators
    var.weights1=get_weights(model.all1)
    weights.all=c()
    
    if(length==2){
      for(i in 5:(p+4)){
        weights.all=c(weights.all, var.weights1[[i]])
      }
    }else if(length==3){
      for(i in 7:(p+6)){
        weights.all=c(weights.all, var.weights1[[i]])
      }
    }
    
    
    
    y.predict=c(predict(model.all1,x.train1))
    
    
    
    df=c(df,sum(abs(weights.all) >= 1e-2))
    
    HBIC = c(HBIC,log10(sum((c(y.train1) - c(y.predict))^2)) + 2*(sum(abs(weights.all) >= 1e-2))*log10(log10(n))*log10(p+q)/n)
    
    
    gc()
    
  }
  
  ind.best=which(HBIC==min(HBIC))
  tuning.best=c(1:lam_length)[ind.best[1]]
  
  
  source(paste0(directory,"/scad",tuning.best,".R"))
  
  
  #Linear effect for M
  inputs1=outputs1=list()
  for(i in 1:p){
    inputs1[[i]] <- layer_input(shape = shape(1), name = paste0("mediator",i))
    outputs1[[i]] <- layer_dense(inputs1[[i]], units = 1, use_bias = FALSE, activation = NULL, kernel_regularizer = scad, kernel_initializer = "zero")
  }
  #Linear effect for x
  for(i in 1:q){
    inputs1[[i+p]] <- layer_input(shape = shape(1), name = paste0("exposure",i+p))
    outputs1[[i+p]] <- layer_dense(inputs1[[i+p]], units = 1, use_bias = FALSE, activation = NULL, kernel_initializer = "normal")
  }
  
  
  
  if(length==2){
    input.z <- layer_input(shape = shape(r), name = "confounder")
    z1 <- layer_dense(input.z, width, activation = "relu", name = "layer_1")
    z1.drop <- layer_dropout(z1, rate=dropout)
    z2 <- layer_dense(z1.drop, width, activation = "relu", name = "layer_2")
    z2.drop <- layer_dropout(z2, rate=dropout)
    inputs1[[p+q+1]]=input.z
    outputs1[[p+q+1]] <- layer_dense(z2.drop, 1, name = "output1")
  }else if(length==3){
    input.z <- layer_input(shape = shape(r), name = "confounder")
    z1 <- layer_dense(input.z, width, activation = "relu", name = "layer_1")
    z1.drop <- layer_dropout(z1, rate=dropout)
    z2 <- layer_dense(z1.drop, width, activation = "relu", name = "layer_2")
    z2.drop <- layer_dropout(z2, rate=dropout)
    z3 <- layer_dense(z2.drop, width, activation = "relu", name = "layer_3")
    z3.drop <- layer_dropout(z3, rate=dropout)
    inputs1[[p+q+1]]=input.z
    outputs1[[p+q+1]] <- layer_dense(z3.drop, 1, name = "output1")
  }
  
  
  
  #Add all outputs
  layer.out1=layer_add(outputs1)
  model.all1 <- keras_model(inputs = inputs1, outputs = layer.out1)
  model.all1 %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(),
    metrics = c('mse')
  )
  
  
  
  history <- model.all1 %>% fit(
    x=x.train1,
    y=y.train1,
    epochs = epoch, batch_size = batch
  )
  
  var.weights1=get_weights(model.all1)
  
  
  
  weights.all=c()
  if(length==2){
    for(i in 5:(p+4)){
      weights.all=c(weights.all, var.weights1[[i]])
    }
    DE.est=var.weights1[[p+5]]
  }else if(length==3){
    for(i in 7:(p+6)){
      weights.all=c(weights.all, var.weights1[[i]])
    }
    DE.est=var.weights1[[p+7]]
  }
  
  y.predict=c(predict(model.all1,x.train1))
  
  active=which((abs(weights.all)>1e-1))#selected mediators by SCAD
  M.new=M[,active]
  p.new=length(active)
  sig1=sum((c(y.predict) - c(y.train1))^2)/(n-p.new-q)
  
  #####################################################
  ###################Total model#######################
  #####################################################
  
  
  index.tuning=sample(1:n, ceiling(split*n), replace = FALSE)
  x.train2=x.train2.train=x.train2.test=list()
  for(j in 1:q){
    x.train2[[j]]=x[,j]
    x.train2.train[[j]]=x[index.tuning,j]
    x.train2.test[[j]]=x[-index.tuning,j]
  }
  x.train2[[q+1]]=z
  x.train2.train[[q+1]]=z[index.tuning,]
  x.train2.test[[q+1]]=z[-index.tuning,]
  y.train2=y
  y.train2.train=y[index.tuning,]
  y.train2.test=y[-index.tuning,]
  
  
  
  loss=array(NA,c(length(length.ttl),length(width.ttl),length(dropout.ttl)))
  
  
  
  
  
  
  
  for(ll in 1:length(length.ttl)){
    for(pp in 1:length(width.ttl)){
      for(ss in 1:length(dropout.ttl)){
        
        
        
        
        inputs2=outputs2=list()
        #Linear effect for x
        for(i in 1:q){
          inputs2[[i]] <- layer_input(shape = shape(1), name = paste0("exposure",i))
          outputs2[[i]] <- layer_dense(inputs2[[i]], units = 1, use_bias = FALSE, activation = NULL, kernel_initializer = "normal")
        }
        #Nonlinear effect for z
        if(length.ttl[ll]==2){
          input.z <- layer_input(shape = shape(r), name = "confounder")
          z1 <- layer_dense(input.z, width.ttl[pp], activation = "relu", name = "layer_1")
          z1.drop <- layer_dropout(z1, rate=dropout.ttl[ss])
          z2 <- layer_dense(z1.drop, width.ttl[pp], activation = "relu", name = "layer_2")
          z2.drop <- layer_dropout(z2, rate=dropout.ttl[ss])
          inputs2[[q+1]]=input.z
          outputs2[[q+1]] <- layer_dense(z2.drop, 1, name = "output1")
        }else if(length.ttl[ll]==3){
          input.z <- layer_input(shape = shape(r), name = "confounder")
          z1 <- layer_dense(input.z, width.ttl[pp], activation = "relu", name = "layer_1")
          z1.drop <- layer_dropout(z1, rate=dropout.ttl[ss])
          z2 <- layer_dense(z1.drop, width.ttl[pp], activation = "relu", name = "layer_2")
          z2.drop <- layer_dropout(z2, rate=dropout.ttl[ss])
          z3 <- layer_dense(z2.drop, width.ttl[pp], activation = "relu", name = "layer_3")
          z3.drop <- layer_dropout(z3, rate=dropout.ttl[ss])
          inputs2[[q+1]]=input.z
          outputs2[[q+1]] <- layer_dense(z3.drop, 1, name = "output1")
        }
        
        #Add all outputs
        layer.out2=layer_add(outputs2)
        model.all2 <- keras_model(inputs = inputs2, outputs = layer.out2)
        model.all2 %>% compile(
          loss = 'mse',
          optimizer = optimizer_adam(),
          #optimizer = optimizer_rmsprop(),
          metrics = c('mse')
        )
        
        
        
        history <- model.all2 %>% fit(
          x=x.train2.train,
          y=y.train2.train,
          epochs = epoch.ttl, batch_size = batch.ttl
        )
        
        
        y.predict=c(predict(model.all2,x.train2.test))
        
        
        
        loss[ll,pp,ss] = mean((c(y.train2.test) - c(y.predict))^2)
        
        rm(model.all2)
        rm(inputs2)
        rm(outputs2)
        rm(layer.out2)
        rm(history)
        
        gc()
      }
    }
  }
  
  ind.best=which(loss==min(loss), arr.ind =TRUE)
  l.best=length.ttl[ind.best[1]]
  p.best=width.ttl[ind.best[2]]
  s.best=dropout.ttl[ind.best[3]]
  
  
  
  
  inputs2=outputs2=list()
  #Linear effect for x
  for(i in 1:q){
    inputs2[[i]] <- layer_input(shape = shape(1), name = paste0("exposure",i))
    outputs2[[i]] <- layer_dense(inputs2[[i]], units = 1, use_bias = FALSE, activation = NULL, kernel_initializer = "normal")
  }
  #Nonlinear effect for z
  if(l.best==2){
    input.z <- layer_input(shape = shape(r), name = "confounder")
    z1 <- layer_dense(input.z, p.best, activation = "relu", name = "layer_1")
    z1.drop <- layer_dropout(z1, rate=s.best)
    z2 <- layer_dense(z1.drop, p.best, activation = "relu", name = "layer_2")
    z2.drop <- layer_dropout(z2, rate=s.best)
    inputs2[[q+1]]=input.z
    outputs2[[q+1]] <- layer_dense(z2.drop, 1, name = "output1")
  }else if(l.best==3){
    input.z <- layer_input(shape = shape(r), name = "confounder")
    z1 <- layer_dense(input.z, p.best, activation = "relu", name = "layer_1")
    z1.drop <- layer_dropout(z1, rate=s.best)
    z2 <- layer_dense(z1.drop, p.best, activation = "relu", name = "layer_2")
    z2.drop <- layer_dropout(z2, rate=s.best)
    z3 <- layer_dense(z2.drop, p.best, activation = "relu", name = "layer_3")
    z3.drop <- layer_dropout(z3, rate=s.best)
    inputs2[[q+1]]=input.z
    outputs2[[q+1]] <- layer_dense(z3.drop, 1, name = "output1")
  }
  
  
  #Add all outputs
  layer.out2=layer_add(outputs2)
  model.all2 <- keras_model(inputs = inputs2, outputs = layer.out2)
  model.all2 %>% compile(
    loss = 'mse',
    optimizer = optimizer_adam(),
    #optimizer = optimizer_rmsprop(),
    metrics = c('mse')
  )
  #Training sets including M,x,z
  
  
  
  history <- model.all2 %>% fit(
    x=x.train2,
    y=y.train2,
    epochs = epoch.ttl, batch_size = batch.ttl
  )
  
  var.weights2=get_weights(model.all2)
  
  
  
  if(l.best==2){
    ttl.est=var.weights2[[5]]
  }else if(l.best==3){
    ttl.est=var.weights2[[7]]
  }
  
  
  
  y.predict=c(predict(model.all2,x.train2))
  
  sig=(sum((c(y.predict)-c(y.train2))^2))/(n-1)
  
  IE.est=ttl.est-DE.est
  
  sig2=sig-sig1
  
  
  #############################################################
  ###############Asymptotic covariance matrix##################
  #############################################################
  
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = width.test, activation = "relu", input_shape = c(r),kernel_initializer = "normal", keras::constraint_maxnorm(max_value = 1, axis = 0))
  model %>% layer_dropout(rate=dropout.test)
  for(xx in  1:length.test){
    model %>% layer_dense(units = width.test, activation = "relu",kernel_initializer = "normal", keras::constraint_maxnorm(max_value = 1, axis = 0))
    model %>% layer_dropout(rate=dropout.test)
  }
  model %>% layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(),
    metrics = c('mse')
  )
  
  
  
  if(length(p.new)!=0){
    est=list()
    for(kk in 1:p.new){
      history <- model %>% fit(
        z, M.new[,kk], 
        epochs = epoch.test, batch_size = batch.test
      )
      est[[kk]]=model %>% predict(z) - c(M.new[,kk])#residual
    }
    history <- model %>% fit(
      z, x, 
      epochs = epoch.test, batch_size = batch.test
    )
    est[[p.new+1]]=model %>% predict(z) - c(x)#residual
    Omega=matrix(NA, p.new+1,p.new+1)
    for(ii in 1:(p.new+1)){
      for(jj in 1:(p.new+1)){
        Omega[ii,jj]=mean(est[[ii]]*est[[jj]])
      }
    }
    
    stat=n*(IE.est)^2*(sig2*(Omega[p.new+1,p.new+1])^(-1) + sig1*(solve(Omega)[p.new+1,p.new+1] - (Omega[p.new+1,p.new+1])^(-1)))^(-1)
    
  }else{
    history <- model %>% fit(
      z, x, 
      epochs = epoch.test, batch_size = batch.test
    )
    est=model %>% predict(z)- c(x)#residual
    Omega=mean((est)^2)
    stat=n*(IE.est)^2*(sig2*(Omega[p.new+1,p.new+1])^(-1))^(-1)
  }
  
  p.value=pchisq(stat,q,lower.tail = F)
  
  
  t2=Sys.time()
  t3=t2-t1
  
  record=list(DE.est=DE.est, IE.est=IE.est, ttl.est= ttl.est, mediator.weights=weights.all, HBIC=HBIC, df=df, t3=t3, p.value=p.value)
  
  return(record)
}
