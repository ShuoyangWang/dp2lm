# DP2LM: Deep Neural Network based penalized partially linear mediation model with high-dimensional mediators
------------------------------------------------

# Mediation model
- Outcome-mediator model:
![model1](https://latex.codecogs.com/svg.image?&space;y=\alpha_{\textit{m}}^\intercal&space;m&plus;\alpha_{\textit{e}}^\intercal&space;x&plus;f\left(z\right)&plus;\epsilon_1.)
- Mediator-exposure model:
![model2](https://latex.codecogs.com/svg.image?m=\gamma_{\textit{e}}^\intercal&space;x&plus;g\left(z\right)&plus;\epsilon_2.)
- Total model
![model3](https://latex.codecogs.com/svg.image?&space;y=\theta_{\textit{e}}^\intercal&space;x&plus;h(z)&plus;\epsilon_3.)
- ![covariates](https://latex.codecogs.com/svg.image?\text{Exposure}:x\in\mathbb{R}^q,\text{mediator}:m\in\mathbb{R}^p,\text{confounder}:z\in\mathbb{R}^r)
- Direct effect: ![dirrect](https://latex.codecogs.com/svg.image?\alpha_{\textit{e}})
- Indirect effect: ![indirect](https://latex.codecogs.com/svg.image?\beta_{\textit{e}}=\theta_{\textit{e}}-\alpha_{\textit{e}})
-------------------------------------------------------------

# Estimation via deep neural networks 
- Estimation of direct effect:

![direct](https://latex.codecogs.com/svg.image?\left(\hat{\alpha}_{\textit{m}},\hat{\alpha}_{\textit{e}},\hat{f}\right)={argmin}\frac{1}{n}\sum_{i=1}^n\left(y_i-\alpha_{\textit{m}}^\intercal&space;m_{i}-\alpha_{\textit{e}}^\intercal&space;x_{i}-f(z_i)\right)^2&plus;\sum_{j=1}^p&space;P_{\lambda}\left(\mid\alpha_{\textit{m}j}\mid\right))

![Scad](https://latex.codecogs.com/svg.image?&space;P'_{\lambda}(t)=\lambda\left(\mathbb{I}\left(t\leq\lambda\right)&plus;\frac{\left(a\lambda-t\right)_&plus;}{\left(a-1\right)\lambda}\mathbb{I}\left(t>\lambda\right)\right),a=3.7)

- Estimation of indirect effect:

  ![total](https://latex.codecogs.com/svg.image?\left(\hat{\theta}_{\textit{e}},\hat{h}\right)={argmin}\frac{1}{n}\sum_{i=1}^n\left(y_i-\theta_{\textit{e}}^\intercal&space;x_{i}-h(z_i)\right)^2)

  ![indirect](https://latex.codecogs.com/svg.image?\hat{\beta}_{\textit{e}}=\hat{\theta}_{\textit{e}}-\hat{\alpha}_{\textit{e}})

-------------------------------------------------------------
# Inference via deep neural networks
-------------------------------------------------------------

# Deep neural network hyperparameters and structures
- L: number of layers 
- p: neurons per layer (uniform for all layers)
- s: dropout rate (data dependent)
- Loss function: mean squared loss
- Batch size: data dependent
- Epoch number: data dependent
- Activation function: ReLU
- Optimizer: Adam
- Regularizer: SCAD
-------------------------------------------------------------

# Function descriptions
- "estimation.R": estimation of direct effect via regularization and indirect effect by difference method using neural networks.
- "inference_direct.R": inference of direct effect using neural networks
- "inference_indirect.R" inference of indirect effect using neural networks
-------------------------------------------------------------

# Examples
- "data.R": a data generating example.
