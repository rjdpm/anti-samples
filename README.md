## Table of Contents
- [A Novel Anti-Sample Generation Technique for Effective Machine Unlearning](#about)
- [Requirments](#Requirments)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# A Novel Anti-Sample Generation Technique for Effective Machine Unlearning
In this paper, we propose a novel algorithm that effectively neutralizes the influence of a specific data subset on an existing trained model. Our algorithm generates noise that acts in opposition to the target data subset, preserving a significant amount of model performance on the remaining datapoints. We presented two results: ___Unlearning-30___ and ___Unlearning-100___ corresponding to retain data percentage. Empirical results on ResNet18 model with CIFAR-10 dataset are presented in `test_example_single_class_unleaning_30.ipynb` and `test_example_single_class_unleaning_100.ipynb`.

## Requirments
To run the notebook followings are needed:  
UBUNTU: 22.04 <br> CUDA: 12.4 <br> Python3: 3.12.2 <br> Anaconda: 24.5.0 <br> Pip3: 24.2 (Conda) <br>  

All dependencies can be installed using `conda_environment.yml` file. To create the environment run the following command:  
`conda env create -f conda_environment.yml`  

