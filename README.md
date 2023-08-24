# DP2LM: Deep Neural Network based penalized partially linear mediation model with high-dimensional mediators
------------------------------------------------

# Mediation model
- Outcome-mediator model
![model1](https://latex.codecogs.com/svg.image?&space;y=\alpha_{\textit{m}}^\intercal&space;m&plus;\alpha_{\textit{e}}^\intercal&space;x&plus;f\left(z\right)&plus;\epsilon_1.)
- Mediator-exposure model
![model2](https://latex.codecogs.com/svg.image?m=\gamma_{\textit{e}}^\intercal&space;x&plus;g\left(z\right)&plus;\epsilon_2.)





- ![X](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BX%7D_%7Bj%7D%5Cin%20%5Cmathbb%7BR%7D%5Ed): fixed vector of length d for the j-th observational point
- ![Y](https://latex.codecogs.com/gif.latex?Y_%7Bij%7D): scalar random variable for the i-th subject and j-th observational point
- ![error](https://latex.codecogs.com/gif.latex?%5Cepsilon_%7Bi%7D%5Cleft%28%5Cmathbf%7BX%7D_j%5Cright%29): error random process with measurement error for the i-th subject and j-th observational point
- ![n](https://latex.codecogs.com/gif.latex?n): sample size
- ![N](https://latex.codecogs.com/gif.latex?N): number of observational points
- ![f](https://latex.codecogs.com/gif.latex?f_0%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D): true function to estimate

# Deep Neural Network Model input and output
- Input: ![X](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BX%7D_%7Bj%7D) (uniform among all i for the same j)
- Output: ![Y](https://latex.codecogs.com/gif.latex?Y_%7Bij%7D)
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
