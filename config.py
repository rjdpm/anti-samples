##############################################################################################
#% WRITER: RAJDEEP MONDAL            DATE: 20-11-2024
#% For bug and others mail me at: rdmondalofficial@gmail.com
#%--------------------------------------------------------------------------------------------
##############################################################################################

from load_datasets import *

__all__ = [
    'config_all',
    'dict_datasets'
    ]


config_all_allcnn = {'cifar10':{'learning_rate':1e-3,
                            'unlearn_scale_lr':2.3,
                            'batch_size':128,
                            'num_train_epochs':30,
                            'num_unlearn_epochs':2,
                            'local_variance':10,
                            'size_multiplier':1,
                            'p':0.85,
                            'tail_randomized':20,
                            'solver_type':'adam'},
                
                    
                     'fashionMNIST':{'learning_rate':1e-3,  
                              'unlearn_scale_lr':3.6, 
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':10,
                              'size_multiplier':1,
                              'p':0.95,
                              'tail_randomized':50,
                              'solver_type':'adam'},
                
                
                    'mnist':{'learning_rate':1e-3,
                              'unlearn_scale_lr':4.8,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':10,
                              'size_multiplier':1,
                              'p':0.85,
                              'tail_randomized':20,
                              'solver_type':'adam'},
                
                
                     'svhn':{'learning_rate':1e-3,
                             'unlearn_scale_lr':4,
                             'batch_size':128,
                             'num_train_epochs':30,
                             'num_unlearn_epochs':2,
                            'local_variance':1,
                            'size_multiplier':1,
                            'p':0.95,
                            'tail_randomized':50,
                            'solver_type':'adam'}
                     
          }


config_all_mobilenet_v2 = {'cifar10':{'learning_rate':1e-3,
                            'unlearn_scale_lr':1.8,
                            'batch_size':128,
                            'num_train_epochs':30,
                            'num_unlearn_epochs':2,
                            'local_variance':5,
                            'size_multiplier':1,
                            'p':0.95,
                            'tail_randomized':30,
                            'solver_type':'adam'},
                
                    
                     'fashionMNIST':{'learning_rate':1e-3,
                              'unlearn_scale_lr':2,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':1,
                              'size_multiplier':2,
                              'p':0.95,
                              'tail_randomized':50,
                              'solver_type':'adam'},
                
                    'mnist':{'learning_rate':1e-3,
                              'unlearn_scale_lr':3.5,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':10,
                              'size_multiplier':1,
                              'p':0.90,
                              'tail_randomized':50,
                              'solver_type':'adam'},
                
                     'svhn':{'learning_rate':1e-3,
                             'unlearn_scale_lr':3.1,
                             'batch_size':128,
                             'num_train_epochs':30,
                             'num_unlearn_epochs':5,
                            'local_variance':1,
                            'size_multiplier':1,
                            'p':0.85,
                            'tail_randomized':20,
                            'solver_type':'adam'}
          }

config_all_resnet9 = {'cifar10':{'learning_rate':1e-3,
                            'unlearn_scale_lr':2.5,
                            'batch_size':128,
                            'num_train_epochs':30,
                            'num_unlearn_epochs':5,
                            'local_variance':10,
                            'size_multiplier':2,
                            'p':0.85,
                            'tail_randomized':20,
                            'solver_type':'adam'},
                
                
                    'fashionMNIST':{'learning_rate':1e-3,
                              'unlearn_scale_lr':4.5,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':1,
                              'size_multiplier':1,
                              'p':0.85,
                              'tail_randomized':30,
                              'solver_type':'adam'},
                
                    'mnist':{'learning_rate':1e-3,
                              'unlearn_scale_lr':7,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':10,
                              'size_multiplier':1,
                              'p':0.85,
                              'tail_randomized':50,
                              'solver_type':'adam'},
                
                     'svhn':{'learning_rate':1e-3,
                             'unlearn_scale_lr':4.5,
                             'batch_size':128,
                             'num_train_epochs':30,
                             'num_unlearn_epochs':2,
                            'local_variance':1,
                            'size_multiplier':1,
                            'p':0.85,
                            'tail_randomized':50,
                            'solver_type':'adam'}
          }

config_all_resnet18 = {'cifar10':{'learning_rate':1e-3,
                            'unlearn_scale_lr':2.3,
                            'batch_size':128,
                            'num_train_epochs':30,
                            'num_unlearn_epochs':5,
                            'local_variance':5,
                            'size_multiplier':1,
                            'p':0.95,
                            'tail_randomized':30,
                            'solver_type':'adam'},
                     
                    
                     'fashionMNIST':{'learning_rate':1e-3,
                              'unlearn_scale_lr':3.3,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':1,
                              'size_multiplier':1,
                              'p':0.85,
                              'tail_randomized':50,
                              'solver_type':'adam'},
                
                    'mnist':{'learning_rate':1e-3,
                              'unlearn_scale_lr':4,
                              'batch_size':128,
                              'num_train_epochs':30,
                              'num_unlearn_epochs':2,
                              'local_variance':10,
                              'size_multiplier':2,
                              'p':0.85,
                              'tail_randomized':30,
                              'solver_type':'adam'},
                
                     'svhn':{'learning_rate':1e-3,
                             'unlearn_scale_lr':3.3,
                             'batch_size':128,
                             'num_train_epochs':30,
                             'num_unlearn_epochs':2,
                            'local_variance':10,
                            'size_multiplier':1,
                            'p':0.85,
                            'tail_randomized':50,
                            'solver_type':'adam'}
          }


config_all = { 'AllCNN':config_all_allcnn, 'MobileNet_v2':config_all_mobilenet_v2, 'ResNet9':config_all_resnet9, 'ResNet18':config_all_resnet18}

dict_datasets = {'cifar10':load_cifar10, 'fashionMNIST':load_fashionmnist, 'mnist':load_mnist, 'svhn':load_svhn}
