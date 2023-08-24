# DP2LM: Deep Neural Network based penalized partially linear mediation model with high-dimensional mediators
------------------------------------------------

# Mediation model
- Outcome-mediator model:
![model1](https://latex.codecogs.com/svg.image?&space;y=\alpha_{\textit{m}}^\intercal&space;m&plus;\alpha_{\textit{e}}^\intercal&space;x&plus;f\left(z\right)&plus;\epsilon_1.)
- Mediator-exposure model:
![model2](https://latex.codecogs.com/svg.image?m=\gamma_{\textit{e}}^\intercal&space;x&plus;g\left(z\right)&plus;\epsilon_2.)
- Total model
![model3](https://latex.codecogs.com/svg.image?&space;y=\theta_{\textit{e}}^\intercal&space;x&plus;h(z)&plus;\epsilon_3.)
- ![covariates](https://latex.codecogs.com/svg.image?\text{Response}:y\in\mathbb{R},\text{exposure}:x\in\mathbb{R}^q,\text{mediator}:m\in\mathbb{R}^p,\text{confounder}:z\in\mathbb{R}^r)
- Direct effect: ![dirrect](https://latex.codecogs.com/svg.image?\alpha_{\textit{e}})
- Indirect effect: ![indirect](https://latex.codecogs.com/svg.image?\beta_{\textit{e}}=\theta_{\textit{e}}-\alpha_{\textit{e}})
-------------------------------------------------------------

# Estimation via deep neural networks 

-------------------------------------------------------------
# Inference via deep neural networks
-------------------------------------------------------------

# Deep Neural Network Hyperparameters and Structures
- L: number of layers 
- p: neurons per layer (uniform for all layers)
- s: dropout rate (data dependent)
- Loss function: absolute value loss/ square loss/ huber loss/ check loss
- Batch size: data dependent
- Epoch number: data dependent
- Activation function: ReLU
- Optimizer: Adam 
-------------------------------------------------------------

# Function descriptions
- "rdnn.R": robust dnn estimation for multi-dimensional funtional data, with dimension no more than 4. More details can be found in the file.
-------------------------------------------------------------

# Examples
- "example.R": 2D and 3D functional data regression examples. Cauchy and Slash distributed measurement errors are added to the observations. 
