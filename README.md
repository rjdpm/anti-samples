# A Novel Anti-Sample Generation Technique for Effective Machine Unlearning

## 📑 Table of Contents
- [Overview](#-overview)
- [Requirements](#%EF%B8%8F-requirements)
- [Setup Instructions](#%EF%B8%8F-setup-instructions)
- [Usages](#-usages)
- [Proof of Concept](#-proof-of-concept-toy-example)
- [License](#license)
- [Citation](#citation)

---

## 📖 Overview
In this paper, we propose a novel algorithm that effectively neutralizes the influence of a specific data subset on an existing trained model. Our algorithm generates noise that acts in opposition to the target data subset while preserving a significant amount of model performance on the remaining datapoints. 

We present two results:  
- **Unlearning-30**: 30% of the retain data is used in the train-to-unlearn step.  
- **Unlearning-100**: 100% of the retain data is used in the train-to-unlearn step.  

Empirical results for these are demonstrated on a ResNet18 model using the CIFAR-10 dataset. The experiments can be found in the following notebooks:  
- `test_example_single_class_unlearning_30.ipynb`  
- `test_example_single_class_unlearning_100.ipynb`  

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

**Conda Libraries:**  
 Numpy >= 1.26.4    
 Torch >= 2.4.1     
 Torch-cuda >= 12.4    
 Pandas >= 2.2.2    
 Matplotlib >= 3.9.2    
 Seaborn >= 0.13.2   
 Scikit-learn >= 1.5.2  
 Tqdm >= 4.66.5       
 Torchinfo >= 1.8.0    


---

## ⚙️ Setup Instructions

### Install Dependencies
All required dependencies can be installed using the provided `conda_environment.yml` file. Use the following commands:  

1. Create the Conda environment:
   ```bash
   conda env create -f conda_environment.yml
   
2. Activate the environments:
   ```bash
   conda activate condapy312`
   
After activating the environment, open and run the desired notebook. Get pre-trained ResNet18 network in [here](https://drive.google.com/file/d/1VFkBE7C8aAKxFdYd1O-HQzSSMBkwgD9B/view?usp=drive_link).

## 🚀 Usages

To perform `Single Class Unlearning` task on other models and datasets:

1. Change values of the following variables:  
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


To perform `Multi-Class Unlearning` task on other models and datasets:  

1. Change the values of the aforementioned desired variables.  
2. Add Execute permission:  
   ```bash
   chmod +x multi_class_unlearning.py

3. Run: 
   ```bash
   ./multi_class_unlearning.py

## 💭 Proof of Concept (Toy Example)  

A toy example for proof-of-concept experiments is provided in the following notebook:
`unlearn_comparison_toy_example.ipynb`  
This example gives the overview of the retain accuracy problem for noise generation based unlearning techniques.

## License  
This project is licensed under the MIT License. See the [LICENSE](https://github.com/rjdpm/anti-samples/blob/main/LICENSE.md) file for details.

## Citation  
To cite this paper use:
```latex
@misc{rajdeep2024ANAGTEMU,
      title={A Novel Anti-Sample Generation Technique for Effective Machine Unlearning}, 
      author={Rajdeep Mondal and Soumitra Samanta},
      year={2024}
}


