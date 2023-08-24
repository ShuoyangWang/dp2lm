Data.example=function(index){
  p=200#dim for mediator
  q=1#dim for exposure variable
  r=8#dim for confounder
  n=100
  
  
  set.seed(index)
  
  
  
  
  
  
  x=matrix((rnorm(n*q)), n,q)
  z=matrix((rnorm(n*r)), n,r)
  Gamma=matrix(rep(1,p),q,p)#True gammas are all 1
  #f1_z=apply(z, 1, function(x) max(min(exp(-(x[1]*x[2] + x[3]))*(x[4]*x[5]),15),-15))
  #f1_z=apply(z, 1, function(x) exp(-(x[1]*x[2] + x[3])))
  f1_z=apply(z, 1, function(x) 2*x[1]*x[2]+3*cos(pi*(x[3])^2*x[4])*x[5]+1/(x[6]+5)+2*sin(-pi*(x[7])^3*x[8]))
  
  Sigma=diag(1,p,p)
  for(i in 1:(p-1)){
    for(j in (i+1):p){
      Sigma[i,j]=Sigma[j,i]=0.5^(abs(i-j))
    }
  }
  e1=mvtnorm::rmvnorm(n, rep(0,p), Sigma)
  M=x%*%Gamma+f1_z+e1
  alpha0=matrix(c(3,4,5,rep(0,p-3)), p, 1)
  alpha1=matrix(2,q,1)
  #f2_z=apply(z, 1, function(x) max(min(exp(-(x[1]*x[2]*x[3])-(x[4]*x[5])),15),-15))
  #f2_z=apply(z, 1, function(x) (x[1])^2*x[2]+sin(x[3]*x[4]*x[5])+exp(-(x[6]*x[7] + x[8])))
  f2_z=apply(z, 1, function(x) 10+3*(x[1])^2*x[2]+sin(x[3]*x[4]*x[5])+log((x[6]*x[7] + x[8]+12)))
  
  error=rnorm(n)
  
  y=M%*%alpha0+x%*%alpha1+f2_z+error
  
  return(list(x=x,z=z,M=M,y=y))
}




