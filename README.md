# A Novel Anti-Sample Generation Technique for Effective Machine Unlearning

## 📑 Table of Contents
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Requirements](#%EF%B8%8F-requirements)
- [Setup Instructions](#%EF%B8%8F-setup-instructions)
- [Usages](#-usages)
- [Proof of Concept](#-proof-of-concept-toy-example)
- [License](#-license)
- [Citation](#-citation)

---

## 📖 Overview
In this paper, we propose a novel algorithm that effectively neutralizes the influence of a specific data subset on an existing trained model. Our algorithm generates noise that acts in opposition to the target data subset while preserving a significant amount of model performance on the remaining datapoints. Code to generate anti-samples is given in [anti_samples.py](https://github.com/rjdpm/anti-samples/blob/PReMI/anti_samples.py).

## 📁 Repository Structure

| File | Description |
|------|--------------|
| **anti_samples.py** | Core implementation of the *anti-sample generation algorithm*. |
| **config_munge.py** | Central configuration file containing hyperparameters for models and datasets. |
| **load_datasets.py** | Dataset loading utilities supporting multiple datasets (e.g., CIFAR-10, MNIST, SVHN, etc.). |
| **models.py** | Model definitions used across unlearning and comparison experiments. |
| **noises_casia_webface_test.ipynb** | Implementation of the **UNSIR** unlearning method on the *CASIA-WebFace* dataset for comparison. |
| **test_example_single_class_unleaning.ipynb** | Demonstration notebook for *single-class unlearning* using ResNet18 on CIFAR-10. |
| **unlearn_comparison_toy_example.ipynb** | Toy example comparing our *Anti-Sample* method against *UNSIR* for conceptual illustration. |
| **unlearn_multi_class.py** | Main script for performing *multi-class unlearning*. |
| **unlearn_single_class.py** | Main script for performing *single-class unlearning*. |
| **utils_unlearn.py** | Helper utilities used across different unlearning modules. |
| **conda_environment.yml** | Conda environment file listing all dependencies. |
| **README.md** | Project documentation. |
| **LICENSE** | License for usage and distribution. |

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
All required dependencies can be installed using the provided [conda_environment.yml](https://github.com/rjdpm/anti-samples/blob/PReMI/conda_environment.yml) file and following commands:  

1. Create the Conda environment:
   ```bash
   conda env create -f conda_environment.yml
   
2. Activate the environment:
   ```bash
   conda activate conda_env
   
3. Clone the Repository:
    ```bash
    git clone https://github.com/rjdpm/anti-samples.git
    cd anti-samples
   
After activating the environment, open and run the desired notebook. Get pre-trained ResNet18 network from [here](https://drive.google.com/file/d/1VFkBE7C8aAKxFdYd1O-HQzSSMBkwgD9B/view?usp=drive_link).

## 🚀 Usages

#### To perform `Single Class Unlearning` task on other models and datasets:

1. Change values of the following variables in [unlarn_single_class.py](https://github.com/rjdpm/anti-samples/blob/PReMI/unlearn_single_class.py):  
   ```python
   dataset_name_list = ['cifar10']  # Options: 'cifar10', 'svhn', 'mnist', 'fashionMNIST', 'casia_webface'
   model_name_list = ['ResNet18'] # Options: 'ResNet9', 'AllCNN', 'ResNet18', 'MobileNet_v2'
   datapath = '/path/to/your/folder/'

3. Add Execute permission:  
   ```bash
   chmod +x unlarn_single_class.py

4. Run: 
   ```bash
   ./unlarn_single_class.py


#### To perform `Multi-Class Unlearning` task on other models and datasets:  

1. Change the values of the aforementioned desired variables in [unlarn_multi_class.py](https://github.com/rjdpm/anti-samples/blob/PReMI/unlearn_multi_class.py) together with:
    ```bash
    num_multiclass_list = [4]
    averaging_epochs_list = [5]
    num_unlearn_epochs = 5
    
2. Add Execute permission:  
   ```bash
   chmod +x unlarn_multi_class.py

3. Run: 
   ```bash
   ./unlarn_multi_class.py

Change different parameter values in [config_munge.py](https://github.com/rjdpm/anti-samples/blob/PReMI/config_munge.py) for parameter and hyperparameter tuning.

## 💭 Proof of Concept (Toy Example)  

A toy example for proof-of-concept experiment is provided in [unlearn_comparison_toy_example.ipynb](https://github.com/rjdpm/anti-samples/blob/PReMI/unlearn_comparison_toy_example.ipynb). This example gives the overview of the retain accuracy problem for noise generation based unlearning techniques.

## 📜 License  
This project is licensed under the Creative Commons Zero v1.0 Universal License. See the [LICENSE](https://github.com/rjdpm/anti-samples/blob/PReMI/LICENSE) file for details.

## 📝 Citation  
To cite this paper use:
```latex
@inproceedings{mondal2025NAG,
	title={A Novel Anti-Sample Generation Technique for Effective Machine Unlearning},
	author={Rajdeep Mondal and Soumitra Samanta},
	booktitle={Pattern Recognition and Machine Intelligence},
	year={2025},
}


