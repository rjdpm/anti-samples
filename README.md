# A Novel Anti-Sample Generation Technique for Effective Machine Unlearning

## 📑 Table of Contents
- [Overview](#-overview)
- [Requirements](#%EF%B8%8F-requirements)
- [Setup Instructions](#%EF%B8%8F-setup-instructions)
- [Usages](#-usages)
- [Proof of Concept](#-proof-of-concept-toy-example)
- [License](#-license)
- [Citation](#-citation)

---

## 📖 Overview
In this paper, we propose a novel algorithm that effectively neutralizes the influence of a specific data subset on an existing trained model. Our algorithm generates noise that acts in opposition to the target data subset while preserving a significant amount of model performance on the remaining datapoints. Code to generate anti-samples is given in [UNMUNGE.py](https://github.com/rjdpm/anti-samples/blob/main/UNMUNGE.py).

We present two results:  
- **Unlearning-30**: 30% of the retain data is used in the train-to-unlearn step.  
- **Unlearning-100**: 100% of the retain data is used in the train-to-unlearn step.  

Empirical results for these are demonstrated on a ResNet18 model using the CIFAR-10 dataset. The experiments can be found in the following notebooks:  
- **Unlearning-30:** [test_example_single_class_unleaning_30.ipynb](https://github.com/rjdpm/anti-samples/blob/main/test_example_single_class_unleaning_30.ipynb)  
- **Unlearning-100:** [test_example_single_class_unleaning_100.ipynb](https://github.com/rjdpm/anti-samples/blob/main/test_example_single_class_unleaning_100.ipynb)

---

## 🛠️ Requirements
To run the notebooks, ensure the following are installed:

| Software      | Version   |
|---------------|-----------|
| **Ubuntu**    | 22.04     |
| **CUDA**      | 12.4      |
| **Python3**    | 3.12.2    |
| **Anaconda**  | 24.5.0    |
| **Pip3**       | 24.2 (via Conda) |

---

## ⚙️ Setup Instructions

### Install Dependencies
All required dependencies can be installed using the provided [conda_environment.yml](https://github.com/rjdpm/anti-samples/blob/main/conda_environment.yml) file and following commands:  

1. Create the Conda environment:
   ```bash
   conda env create -f conda_environment.yml
   
2. Activate the environment:
   ```bash
   conda activate conda_env
   
After activating the environment, open and run the desired notebook. Get pre-trained ResNet18 network from [here](https://drive.google.com/file/d/1VFkBE7C8aAKxFdYd1O-HQzSSMBkwgD9B/view?usp=drive_link).

## 🚀 Usages

#### To perform `Single Class Unlearning` task on other models and datasets:

1. Change values of the following variables in [single_class_unlearning.py](https://github.com/rjdpm/anti-samples/blob/main/single_class_unlearning.py):  
   ```python
   dataset_name = 'cifar10'  # Options: 'cifar10', 'svhn', 'mnist', 'fashionMNIST'
   model_name = 'MobileNet_v2'  # Options: 'ResNet9', 'AllCNN', 'ResNet18', 'MobileNet_v2'
   retain_data_percent = 30  # Options: 30, 100


3. Add Execute permission:  
   ```bash
   chmod +x single_class_unlearning.py

4. Run: 
   ```bash
   ./single_class_unlearning.py


#### To perform `Multi-Class Unlearning` task on other models and datasets:  

1. Change the values of the aforementioned desired variables in [multi_class_unlearning.py](https://github.com/rjdpm/anti-samples/blob/main/multi_class_unlearning.py).  
2. Add Execute permission:  
   ```bash
   chmod +x multi_class_unlearning.py

3. Run: 
   ```bash
   ./multi_class_unlearning.py

Change different parameter values in [config.py](https://github.com/rjdpm/anti-samples/blob/main/config.py) for parameter and hyperparameter tuning.

## 💭 Proof of Concept (Toy Example)  

A toy example for proof-of-concept experiment is provided in [unlearn_comparison_toy_example.ipynb](https://github.com/rjdpm/anti-samples/blob/main/unlearn_comparison_toy_example.ipynb). This example gives the overview of the retain accuracy problem for noise generation based unlearning techniques.

## 📜 License  
This project is licensed under the Creative Commons Zero v1.0 Universal License. See the [LICENSE](https://github.com/rjdpm/anti-samples/blob/main/LICENSE.md) file for details.

## 📝 Citation  
To cite this paper use:
```latex
@misc{rajdeep2024ANAGTEMU,
      title={A Novel Anti-Sample Generation Technique for Effective Machine Unlearning}, 
      author={Rajdeep Mondal and Soumitra Samanta},
      year={2024}
}


