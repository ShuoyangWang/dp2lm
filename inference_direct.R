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


DP2LM_DE_test <- function(data,length,width,dropout,epoch,batch,directory,lam_length) {
  
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
  
  
  
  #####################################################
  #######################RSS1##########################
  #####################################################
  
  
  
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
  
  RSS1=sum((c(y.predict) - c(y.train1))^2)
  
  
  
  
  
  #####################################################
  #######################RSS0##########################
  #####################################################
  #Training sets including M,x,z
  x.train1=list()
  
  for(j in 1:p){
    x.train1[[j]]=M[,j]
  }
  x.train1[[p+1]]=z
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
    
    
    if(length==2){
      input.z <- layer_input(shape = shape(r), name = "confounder")
      z1 <- layer_dense(input.z, width, activation = "relu", name = "layer_1")
      z1.drop <- layer_dropout(z1, rate=dropout)
      z2 <- layer_dense(z1.drop, width, activation = "relu", name = "layer_2")
      z2.drop <- layer_dropout(z2, rate=dropout)
      inputs1[[p+1]]=input.z
      outputs1[[p+1]] <- layer_dense(z2.drop, 1, name = "output1")
    }else if(length==3){
      input.z <- layer_input(shape = shape(r), name = "confounder")
      z1 <- layer_dense(input.z, width[pp], activation = "relu", name = "layer_1")
      z1.drop <- layer_dropout(z1, rate=dropout)
      z2 <- layer_dense(z1.drop, width[pp], activation = "relu", name = "layer_2")
      z2.drop <- layer_dropout(z2, rate=dropout)
      z3 <- layer_dense(z2.drop, width[pp], activation = "relu", name = "layer_3")
      z3.drop <- layer_dropout(z3, rate=dropout)
      inputs1[[p+1]]=input.z
      outputs1[[p+1]] <- layer_dense(z3.drop, 1, name = "output1")
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
  
  if(length==2){
    input.z <- layer_input(shape = shape(r), name = "confounder")
    z1 <- layer_dense(input.z, width, activation = "relu", name = "layer_1")
    z1.drop <- layer_dropout(z1, rate=dropout)
    z2 <- layer_dense(z1.drop, width, activation = "relu", name = "layer_2")
    z2.drop <- layer_dropout(z2, rate=dropout)
    inputs1[[p+1]]=input.z
    outputs1[[p+1]] <- layer_dense(z2.drop, 1, name = "output1")
  }else if(length==3){
    input.z <- layer_input(shape = shape(r), name = "confounder")
    z1 <- layer_dense(input.z, width, activation = "relu", name = "layer_1")
    z1.drop <- layer_dropout(z1, rate=dropout)
    z2 <- layer_dense(z1.drop, width, activation = "relu", name = "layer_2")
    z2.drop <- layer_dropout(z2, rate=dropout)
    z3 <- layer_dense(z2.drop, width, activation = "relu", name = "layer_3")
    z3.drop <- layer_dropout(z3, rate=dropout)
    inputs1[[p+1]]=input.z
    outputs1[[p+1]] <- layer_dense(z3.drop, 1, name = "output1")
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
  
  RSS0=sum((c(y.predict) - c(y.train1))^2)
  
  stat=(RSS0-RSS1)/(RSS1/(n-q))
  
  t2=Sys.time()
  t3=t2-t1
  
  record=list(DE.est=DE.est, stat=stat, t3=t3)
  
  return(record)
}
