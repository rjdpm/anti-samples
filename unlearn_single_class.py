#!/usr/bin/env python3
import os, sys
import numpy as np
import copy
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils_unlearn import *
from anti_samples import *
from load_datasets import *
from config_munge import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Codes.data_utils_local.Generalised_data_utils import list2json, load_json

from datetime import datetime
import pytz
ist = pytz.timezone('Asia/Kolkata')
datetime_now = datetime.now(ist).strftime('%Y%m%d_[%H:%M:%S]')

##################################################### Inputs ######################################################################

dataset_name_list = ['casia-webface']#['mnist', 'fashionMNIST', 'svhn', 'cifar10']
model_name_list = ['ResNet18']#['MobileNet_v2', 'ResNet18', 'AllCNN', 'ResNet9']#
# Number of datapoints selected from each retain class for unlearning
retain_data_percent_list = [100]#[30, 100]
unlearn_type = 'Single_class_Unlearn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrained_models_folder_name = 'retrained_models'
unlearned_models_folder_name = 'unlearned_models'
delete_saved_unlearned_models = True
#---------------------------------------------------------------------------------------------------------------------------------
#=================================================================================================================================
           
for dataset_name in dataset_name_list:
    for model_name in model_name_list:
        try:
            #=================================================================================================================================
            # Loading configurations for a particular model and dataset
            config = config_all[model_name][dataset_name]
            all_result_folder_path = './results'
            result_savepath = ''.join([all_result_folder_path, '/', dataset_name, '/', model_name, '/'])
            unlearned_models_path = ''.join([result_savepath, '/', unlearn_type, '/', unlearned_models_folder_name])
            #---------------------------------------------------------------------------------------------------------------------------------
            #=================================================================================================================================
            
            # Number of datapoints selected from each retain class for unlearning
            if dataset_name == 'mnist' or dataset_name == 'fashionMNIST':
                num_input_channels = 1
                num_classes = 10
                padding = 2
                # H, W = image_size[0], image_size[1]
            elif dataset_name == 'svhn' or dataset_name == 'cifar10' or dataset_name == 'cifar100':
                num_input_channels = 3
                num_classes = 10
                if dataset_name == 'cifar100':
                    num_classes = 20#100
                padding = 0
            elif dataset_name == 'casia-webface':
                num_input_channels = 3
                num_classes = 300
                padding = 0
                random_uclasses_path = ''.join([result_savepath, '/', unlearn_type, '/', 'unlearn_classes.json'])
            else:
                print('Details about data not found.')
            
            for retain_data_percent in retain_data_percent_list:

                ##################################################### Parameters ##################################################################
                learning_rate = config['learning_rate']
                batch_size = config['batch_size']
                num_train_epochs = config['num_train_epochs']
                num_unlearn_epochs = 5
                solver_type = 'adam'
                
                unlearn_scale_lr = config['unlearn_scale_lr'][0] if retain_data_percent==30 else config['unlearn_scale_lr'][1]
                local_variance = config['local_variance'][0] if retain_data_percent==30 else config['local_variance'][1]
                size_multiplier = config['size_multiplier'][0] if retain_data_percent==30 else config['size_multiplier'][1]
                p = config['p'][0] if retain_data_percent==30 else config['p'][1]
                tail_randomized=config['tail_randomized'][0] if retain_data_percent==30 else config['tail_randomized'][1]
                
                print('\n\n')
                print('-'*80)
                print(f'{'Dataset':50} = {dataset_name}')
                print(f'{'Model':50} = {model_name}')
                print(f'{"Learning Rate":50} = {learning_rate}')
                print(f'{'Scaling factor of learning rate for Unlearning':50} = {unlearn_scale_lr}')
                print(f'{"Batch Size":50} = {batch_size}')
                print(f'{'Number of Training Epochs':50} = {num_train_epochs}')
                print(f'{'Number of Unlearning Epochs':50} = {num_unlearn_epochs}')
                print(f'{"Percentage of Retain Data selected from each class":50} = {retain_data_percent}')
                print(f'{'Local Varience':50} = {local_variance}')
                print(f'{'Size Multiplier':50} = {size_multiplier}')
                print(f'{'Acceptance Probability of the Unmunged Samples':50} = {p}')
                print(f'{'Tail Randomizer':50} = {tail_randomized}')
                print(f'{'Solver Type':50} = {solver_type}')
                # print(f'Number of Generated Data = {no_generated_data}')
                print('-'*80)
                print('='*80)
                print('\n\n')

                ################################################ Loading Datasets ##############################################################
                # For local system
                #datapath = '/home/dell/Workspace/Codes/Datasets/Torchvision_Data/'

                # For GPU's
                datapath = '/home/rajdeep/workspace/Datasets/Torchvision_Data/'
                if dataset_name == 'svhn':
                    datapath = ''.join([datapath, 'SVHN_Data/'])
                if dataset_name == 'casia-webface':
                        datapath = '/home/rajdeep/workspace/Datasets/CASIA-WebFace/casia-webface-dataset.pkl.gz'

                #-------------------------------------------------------------------------------------------------------------------------------

                train_data, test_data = dict_datasets[dataset_name](datapath)# Try to write as a if else with manual function
                # print(f'Size of train data = {train_data.__sizeof__()}')
                train_loader = DataLoader( train_data,
                                        batch_size,
                                        shuffle=True
                                        )
                test_loader = DataLoader(test_data,
                                        batch_size,
                                        shuffle=True
                                        )
                image_size = train_loader.dataset[0][0].shape

                print('='*80)
                print('Data stats:')
                print('-'*80)
                print('Train data size (num_samples x 1-sample size): {} x {}' .format(len(train_data), train_data[0][0].shape))
                print('Test data size (num_samples x 1-sample size): {}  x {}' .format(len(test_data), test_data[0][0].shape))
                print('-'*80)
                print('='*80)
                #--------------------------------------------------------------------------------------------------------------------------

                ########################################################## Data Partition ####################################################
                # Separating Data and Labels from Dataloader

                print('Separating Data and Labels from Dataloader:')
                print('-------------------------------------------')

                input_train_data = []#[None]*len(train_loader)
                input_train_labels = []#[None]*len(train_loader)

                input_test_data = []#[None]*len(test_loader)
                input_test_labels = []#[None]*len(test_loader)

                for data, label in train_data:
                    input_train_data.append(data)
                    input_train_labels.append(label)
                    
                    
                for data, label in test_data:
                    input_test_data.append(data)
                    input_test_labels.append(label)  

                print('Number of unique class: {}\nand classes are:\n{}' .format(len(list(set(input_train_labels))), list(set(input_train_labels))))
                input_train_data = torch.stack(input_train_data)
                input_train_labels = torch.from_numpy(np.array(input_train_labels))
                input_test_data = torch.stack(input_test_data)
                input_test_labels = torch.from_numpy(np.array(input_test_labels))
                print(f'Number of Train Data = {len(input_train_data)}, Number of Test Data = {len(input_test_data)}\n')
                print('-'*80)
                print('='*80)
                print('\n\n')
                #-----------------------------------------------------------------------------------------------------------------------------
                unlearn_classes = sorted(np.unique(input_train_labels.numpy()))
                if dataset_name == 'casia-webface':
                    if not os.path.isfile(random_uclasses_path):
                        unlearn_classes = list(np.random.choice(unlearn_classes, size=10, replace=False).astype(int))
                        list2json(input_list=unlearn_classes,
                                    filepath=''.join([result_savepath, '/', unlearn_type, '/']),
                                    filename= 'unlearn_classes')
                    else:
                        unlearn_classes=load_json(random_uclasses_path)
                num_uclasses = len(unlearn_classes)
                
                ## Fixing seed for Reproduciblity
                np_seed = config['np_seed']
                torch_seed = config['torch_seed']
                np.random.seed(np_seed)#60#20#30
                torch.manual_seed(torch_seed)
                if torch.cuda.is_available():
                    # torch.cuda.empty_cache()
                    torch.cuda.manual_seed_all(torch_seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    
                #-----------------------------------------------------------------------------------------------------------------------------
                #=============================================================================================================================    
                unlearn_train_acc_unlearning = [None]*num_uclasses
                retain_train_acc_unlearning = [None]*num_uclasses
                unlearn_test_acc_unlearning = [None]*num_uclasses
                retain_test_acc_unlearning = [None]*num_uclasses

                retain_train_acc_retrained = [None]*num_uclasses
                unlearn_train_acc_retrained = [None]*num_uclasses
                retain_test_acc_retrained = [None]*num_uclasses
                unlearn_test_acc_retrained = [None]*num_uclasses

                all_train_retain_acc = [None]*num_uclasses
                all_train_unlearn_acc = [None]*num_uclasses
                all_test_retain_acc = [None]*num_uclasses
                all_test_unlearn_acc = [None]*num_uclasses

                unlearn_time_list = [None]*num_uclasses
        
                all_classwise_acc = OrderedDict({'Classes':list(range(num_classes))}) 
                        
                for i, unlearn_cls in tqdm(enumerate(unlearn_classes), total=len(unlearn_classes)):

                    # unlearn_cls = cls#config['unlearn_cls']#0

                    ## Creating Utility Object ##
                    obj_model = utils_(image_size=(image_size[1], image_size[2]),
                                        num_input_channels = image_size[0],
                                        num_classes = num_classes,
                                        learning_rate = learning_rate,
                                        batch_size = batch_size,
                                        num_epochs = num_train_epochs,
                                        padding = padding,
                                        data_name = dataset_name,
                                        unlearn_cls = unlearn_cls,
                                        model_name=model_name,
                                        result_savepath = result_savepath,
                                        retrained_models_folder_name = retrained_models_folder_name,
                                        unlearned_models_folder_name = unlearned_models_folder_name,
                                        solver_type = solver_type,
                                        unlearn_type=unlearn_type
                                        )
                    
                    #Initializing Retraining Model
                    retrain_model = copy.deepcopy(obj_model)
                    
                    #Initializing Retraining Model
                    unlearn_model = copy.deepcopy(obj_model)
                    
                    
                    obj_model.train_data  = train_data
                    obj_model.test_data = test_data
                    ############################################## Loading & Testing the model ################################################

                    if not os.path.isfile(obj_model.best_model_save_path):
                        print(f'No pre-trained model found in: {obj_model.best_model_save_path}')
                        obj_model.count_epoch = 0
                        obj_model.train()
                        
                    # Loading The Original Model
                    obj_model.load_network()
                    obj_model.network.eval()

                    print('Testing main model on train data:\n')
                    confusion_matrix, train_acc, train_classwise_accuracy, train_retain_acc, train_unlearn_acc = accuracy(obj_model.network,
                                                                                                                        train_loader,
                                                                                                                        unlearn_cls=unlearn_cls,
                                                                                                                        num_classes = num_classes
                                                                                                                        )
                    print(f'Test accuracy on Train data = {train_acc}')  
                    print(f'Accuracy on Train retain Data = {train_retain_acc}')
                    print(f'Accuracy on Train unlearn Data = {train_unlearn_acc}')
                    print(f'Test classwise accuracy on Train data = {train_classwise_accuracy}\n')
                    all_train_retain_acc[i] = train_retain_acc
                    all_train_unlearn_acc[i] = train_unlearn_acc
                    all_classwise_acc['Train'] = train_classwise_accuracy
                    print('-'*80)
                    
                    print('Testing main model on train data:\n')
                    confusion_matrix, test_acc, test_classwise_accuracy, test_retain_acc, test_unlearn_acc = accuracy(obj_model.network,
                                                                                                                    test_loader,
                                                                                                                    unlearn_cls=unlearn_cls,
                                                                                                                    num_classes = num_classes
                                                                                                                    )
                    print(f'Test accuracy on Test data = {test_acc}')
                    print(f'Accuracy on Test retain Data = {test_retain_acc}')
                    print(f'Accuracy on Test unlearn Data = {test_unlearn_acc}')
                    print(f'Test classwise accuracy on Test data = {test_classwise_accuracy}\n')
                    all_test_retain_acc[i] = test_retain_acc
                    all_test_unlearn_acc[i] = test_unlearn_acc
                    all_classwise_acc['Test'] = test_classwise_accuracy
                    print('-'*80)
                    print('='*80)
                    print('\n\n')
                    #---------------------------------------------------------------------------------------------------------------------------

                    #Separating Unlearn and Retain Data

                    print('Separating Unlearn and Retained Data from the full training data:')
                    print('-----------------------------------------------------------------')
                    unlearn_data = input_train_data[input_train_labels == unlearn_cls]
                    unlearn_data_shape = unlearn_data.shape
                    retain_train_data = input_train_data[input_train_labels != unlearn_cls]
                    retain_train_labels = input_train_labels[input_train_labels != unlearn_cls]
                    retain_test_data = input_test_data[input_test_labels != unlearn_cls]
                    retain_test_labels = input_test_labels[input_test_labels != unlearn_cls]

                    #Dataloader only for retain data
                    temp = list(zip(retain_train_data,retain_train_labels))
                    obj_model.retain_loader_train = DataLoader(temp,
                                                        batch_size=batch_size,
                                                        shuffle=True)
                    temp = list(zip(retain_test_data, retain_test_labels))
                    obj_model.retain_loader_test = DataLoader(temp,
                                                        batch_size = batch_size,
                                                        shuffle = True
                                                        )
                    print(f'Number of retain data = {len(retain_train_data)} and unlearn data = {len(unlearn_data)}')
                    print('Separation complete.\n')
                    print('-'*80)
                    print('='*80)
                    print('\n\n')
                    #-------------------------------------------------------------------------------------------------------------------------------
                    ################################################################################################################################

                    #Chossing random subsets of fixed size from each retain classes in training data
                    print(f'Selecting {retain_data_percent}-percent random samples from each classes:')
                    print('-------------------------------------------------')
                    retain_data = []
                    retain_labels = []
                    train_index = []
                    labels = list(set(retain_train_labels.numpy()))
                    for label in tqdm(labels):
                        idx = (input_train_labels == label)
                        temp = np.where(idx == True)[0]
                        no_retain_data = int(len(temp)*(retain_data_percent/100))
                        data_idx = np.random.permutation(temp)[:no_retain_data]
                        train_index.extend(data_idx)#Position of the selected images in the training dataset
                        cls_data = input_train_data[data_idx]
                        cls_labels = input_train_labels[data_idx]
                        retain_data.extend(cls_data)
                        retain_labels.extend(cls_labels)     

                    unlearn_retain_data = torch.stack(retain_data) # Contains 'no_retain_data' number of data from each class
                    unlearn_retain_labels = torch.from_numpy(np.array(retain_labels))
                    no_generated_data = no_retain_data#5000#
                    print(f'Number of selected Retained Data = {len(unlearn_retain_data)}\n')
                    print(f'Retained labels are: {labels}')
                    print('-'*80)
                    print('='*80)
                    print('\n\n')
                    #-------------------------------------------------------------------------------------------------------------------------------

                    
                    ######################################################### Unlearning the Model ########################################################
                    print('='*80)
                    print('-'*80)
                    print(f'Unlearning Step(Class = {unlearn_cls})')
                    print('---------------------------')
                    
                    ## Delete Pre-Saved Unlearned Models
                    if delete_saved_unlearned_models:
                        print('='*80)
                        if not os.path.isfile(obj_model.best_unlearn_model_save_path):
                            print('Unlearned Models not exists.')
                        else:
                            print(f'\nDeleting Pre-Saved Unlearned Model:{obj_model.best_unlearn_model_save_path}')
                            os.remove(obj_model.best_unlearn_model_save_path)
                            print(f'Model deleted:{obj_model.best_unlearn_model_save_path}')
                        print('-'*80)
                        print('='*80)
                            
                    unlearn_model.result_savepath = obj_model.result_savepath_unlearned
                    if not os.path.isfile(obj_model.best_unlearn_model_save_path):
                        print(f'No Unlearned model found in: {obj_model.best_unlearn_model_save_path}')
                        print('Unlearning the Model:')
                        print('---------------------')
                        ############################################### Generating UNMUNGED Samples ####################################################

                        retain_train_data = retain_train_data.reshape(len(retain_train_data), -1)
                        unlearn_retain_data = unlearn_retain_data.reshape(len(unlearn_retain_data), -1)
                        unlearn_data = unlearn_data.reshape(len(unlearn_data), -1)
                        print('\n\nGenerating UNMUNGED Samples:')
                        print('----------------------------')
                        unmunge, farthest_point, pairwise_distance, unmunge_time = generate_antisamples(unlearn_data= unlearn_data,
                                                                            retain_data=unlearn_retain_data,#retain_train_data,#
                                                                            retain_labels=unlearn_retain_labels, #retain_train_labels,#
                                                                            local_variance = local_variance,
                                                                            size_multiplier = size_multiplier,
                                                                            p = p,
                                                                            tail_randomized = tail_randomized,
                                                                            no_generated_data=no_generated_data,
                                                                            eps=0.01,
                                                                            convex_combination=True#
                                                                            )
                        
                        ## Generating Noise From Random Convex Combination
                        # unmunge, unmunge_time = random_convex_combination(data = retain_train_data,
                        #                                        labels=retain_train_labels,
                        #                                      no_of_req_data=5,
                        #                                       no_of_generated_data=no_retain_data
                        #                                      )
                        
                        ## Generating Noise From Standard Normal 
                        # unmunge, unmunge_time = torch.randn_like(retain_train_data[:no_generated_data]), 0
                        
                        ## For Training without Antisamples
                        # unmunge, unmunge_time = unlearn_data[:no_generated_data], 0
    
                        data_shape = (len(retain_train_data), ) + input_train_data.shape[1:]
                        retain_train_data = retain_train_data.reshape(*data_shape)

                        data_shape = (len(unlearn_retain_data), ) + input_train_data.shape[1:]
                        unlearn_retain_data = unlearn_retain_data.reshape(*data_shape)

                        data_shape = (len(unmunge), ) + input_train_data.shape[1:]
                        unmunge_data = unmunge.reshape(*data_shape)
                        
                        unlearn_labels = torch.tensor(unlearn_cls).repeat(len(unmunge_data))
                        unlearn_data = unlearn_data.reshape(*unlearn_data_shape)
                        print('Antisample generation complete.')
                        print('-------------------------------')
                        print('\n\n')
                        #--------------------------------------------------------------------------------------------------------------------------------


                        # Creating Dataloader only for Unlearned Data
                        temp = list(zip(unlearn_data,unlearn_labels))
                        obj_model.unlearn_dataloader = DataLoader(temp,
                                                                batch_size=batch_size,
                                                                shuffle=True
                                                                )
                        #--------------------------------------------------------------------------------------------------------------------------------
                        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        
                        
                        # Merging the Selected Retained Data and Generated UNMUNGED Data
                        #---------------------------------------------------------------
                        ##for selected retain data:
                        retain_data = unlearn_retain_data 
                        retain_labels = unlearn_retain_labels
                        
                        ##for full retain data:
                        # retain_data = retain_train_data  
                        # retain_labels = retain_train_labels
                        
                        train_unlearn_images =  torch.concat((retain_data, unmunge_data))
                        train_unlearn_labels = torch.concat((retain_labels, unlearn_labels))
                        temp = list(zip(train_unlearn_images,train_unlearn_labels))
                        obj_model.unlearn_loader_all = DataLoader(temp,
                                                                batch_size=batch_size,
                                                                shuffle=True)
                        #--------------------------------------------------------------------------------------------------------------------------------

                        optimizer = optim.Adam(params=obj_model.network.parameters(), lr = unlearn_scale_lr*learning_rate)
                        model_unlearn, obj_model.unlearn_epoch_train_losses, total_time = obj_model.unlearn(model=obj_model.network,
                                                                            optimizer= optimizer,
                                                                            dataloader=obj_model.unlearn_loader_all,
                                                                            num_epochs=num_unlearn_epochs
                                                                            )
                        
                        unlearn_time_list[i] = total_time
                        
                    else:
                        print(f'Found existing unlearned model in: {obj_model.best_unlearn_model_save_path}')
                    print(f'Loading unlearned model from: {obj_model.best_unlearn_model_save_path}')
                    unlearn_model.load_unlearn_network()
                    unlearn_model.network.eval()
                    print('-'*80)
                    print('\n\n')
                    #---------------------------------------------------------------------------------------------------------------------------------

                    ################################################## Testing the Unlearned Model ###################################################
                    print('Testing on Train Data:')
                    print('----------------------')
                    confusion_matrix, acc, unlearn_train_classwise_accuracy, train_retain_acc_cls, train_unlearn_acc_cls = accuracy(unlearn_model.network,
                                                                                                                                    train_loader,
                                                                                                                                    unlearn_cls=unlearn_cls,
                                                                                                                                    num_classes = num_classes
                                                                                                                                    )
                    print(f'Classwise Accuracy on Train Data after Unlearning(Unlearn Class - {unlearn_cls}):\n {unlearn_train_classwise_accuracy}\n')
                    print(f'Accuracy of Unlearned Model on Retain Data in Train Dataset(Unlearn Class - {unlearn_cls}) = {train_retain_acc_cls}')
                    print(f'Accuracy of Unlearned Model on Unlearn Data in Train Dataset(Unlearn Class - {unlearn_cls}) = {train_unlearn_acc_cls}')
                    
                    retain_train_acc_unlearning[i] = train_retain_acc_cls
                    unlearn_train_acc_unlearning[i] = train_unlearn_acc_cls
                    all_classwise_acc['Train_Unlearn_'+str(unlearn_cls)] = unlearn_train_classwise_accuracy

                    print('\n\nTesting on Test Data:')
                    print('---------------------')
                    confusion_matrix, acc, unlearn_test_classwise_accuracy, test_retain_acc_cls, test_unlearn_acc_cls = accuracy(unlearn_model.network,
                                                                                                                                test_loader,
                                                                                                                                unlearn_cls=unlearn_cls,
                                                                                                                                num_classes = num_classes
                                                                                                                                )
                    print(f'Classwise Accuracy on Test Data after Unlearning(Unlearn Class - {unlearn_cls}) :\n {unlearn_test_classwise_accuracy}\n')
                    print(f'Accuracy of Unlearned Model on Retain Data in Test Dataset(Unlearn Class - {unlearn_cls}) = {test_retain_acc_cls}')
                    print(f'Accuracy of Unlearned Model on Unlearn Data in Test Dataset(Unlearn Class - {unlearn_cls}) = {test_unlearn_acc_cls}')
                    
                    retain_test_acc_unlearning[i] = test_retain_acc_cls
                    unlearn_test_acc_unlearning[i] = test_unlearn_acc_cls
                    all_classwise_acc['Test_Unlearn_'+str(unlearn_cls)] = unlearn_test_classwise_accuracy
                    print('-'*120)
                    print('='*120)
                    print('\n\n')
                    #---------------------------------------------------------------------------------------------------------------------------------
                    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                                
                    
                    ############################################################ Retraining the Model #################################################################################
                    
                    print(f'\n\n############################# Retraining the Model for Unlearn Class - {unlearn_cls} #########################################################')
                    print('=======================================================================================================================================')
                    temp =list(zip(retain_train_data,retain_train_labels))
                    retrain_model.train_data = temp
                    temp = list(zip(retain_test_data,retain_test_labels))
                    retrain_model.test_data = temp
                    # retrain_model.train_data = list(zip(retain_train_data,retain_train_labels))
                    # retrain_model.test_data = obj_model.test_data
                    retrain_model.best_model_save_path = obj_model.best_retrain_model_save_path
                    retrain_model.result_savepath = obj_model.result_savepath_retrained
                    retrain_model.model_save_name = '_retrain_'+str(unlearn_cls)
                    if not os.path.isfile(obj_model.best_retrain_model_save_path):
                        print(f'No Retrained model found in: {obj_model.best_retrain_model_save_path}')
                        retrain_model.count_epoch = 0
                        retrain_model.train()

                    retrain_model.load_network()
                    retrain_model.network.eval()
                    
                    print(f'\nPerformance on Train Data after Retraining(U - {unlearn_cls})')
                    print('-------------------------------------------------')
                    confusion_matrix, acc, retrain_train_classwise_accuracy, retrain_train_retain_acc_cls, retrain_train_unlearn_acc_cls = accuracy(retrain_model.network.to(device),
                                                                                                                                                    train_loader,
                                                                                                                                                    unlearn_cls=unlearn_cls, 
                                                                                                                                                    num_classes = num_classes
                                                                                                                                                    )
                    print(f'Classwise Accuracy on Train Data after Retraining(Unlearn Class - {unlearn_cls}) :\n {retrain_train_classwise_accuracy}')
                    print(f'Accuracy of Retrained Model on Retain Data in Train Dataset = {retrain_train_retain_acc_cls}')
                    print(f'Accuracy of Retrained Model on Unlearn Data in Train Dataset = {retrain_train_unlearn_acc_cls}')
                    
                    retain_train_acc_retrained[i] = retrain_train_retain_acc_cls ###Accuracy of Retrained Model on Retain Data in Train Dataset
                    unlearn_train_acc_retrained[i] = retrain_train_unlearn_acc_cls ###Accuracy of Retrained Model on Unlearn Data in Train Dataset
                    all_classwise_acc['Train_Retrain_'+str(unlearn_cls)] = retrain_train_classwise_accuracy
                    print('-'*80)


                    print(f'\n\n Performance on Test Data after Retraining(U - {unlearn_cls})')
                    print('------------------------------------------------')
                    confusion_matrix, acc, retrain_test_classwise_accuracy, retrain_test_retain_acc_cls, retrain_test_unlearn_acc_cls = accuracy(retrain_model.network.to(device),
                                                                                                                                                test_loader,
                                                                                                                                                unlearn_cls=unlearn_cls, 
                                                                                                                                                num_classes = num_classes
                                                                                                                                                )
                    print(f'Classwise Accuracy on Test Data after Retraining(Unlearn Class - {unlearn_cls}):\n{retrain_test_classwise_accuracy}')
                    print(f'Accuracy of Retrained Model on Retain Data in Test Dataset = {retrain_test_retain_acc_cls}')
                    print(f'Accuracy of Retrained Model on Unlearn Data in Test Dataset = {retrain_test_unlearn_acc_cls}')
                    print('-'*120)
                    print('='*120)
                    print('\n\n')    
                    
                    retain_test_acc_retrained[i] = retrain_test_retain_acc_cls
                    unlearn_test_acc_retrained[i] = retrain_test_unlearn_acc_cls
                    all_classwise_acc['Test_Retrain_'+str(unlearn_cls)] = retrain_test_classwise_accuracy
                    
                    ## Saving Accuracies ##
                    all_accuracy_savepath = ''.join([obj_model.result_savepath,
                                                    unlearn_type,
                                                    #  '_results_gaussian_unlearn_2_epochs/'
                                                    #  '_results_random_convex_unlearn_2_epochs/'
                                                    #  '_results_unmunge_unlearn_2_epochs/'
                                                    #  '_results_without_antisamples/'
                                                    #  '_results_random_convex_combination/'
                                                    #  '_results_standard_gaussian/'
                                                    '_results/'
                                                    ])
                    obj_model.create_folder(all_accuracy_savepath)
                    classwise_accuracy_savepath = ''.join([all_accuracy_savepath,#obj_model.result_savepath,
                                                datetime_now, '_',
                                                obj_model.model_name, '_',
                                                obj_model.data_name, '_',
                                                'classwise_accuracy_all',
                                                '_scllr_',str(unlearn_scale_lr).replace('.', 'o'),
                                                '_nunlep_', str(num_unlearn_epochs),
                                                '_lv_', str(local_variance),
                                                '_smul_', str(size_multiplier),
                                                '_p_', str(p).replace('.', 'o'),
                                                '_tran_', str(tail_randomized),
                                                '_ngnrtd_', str(no_generated_data),
                                                '_prcnt_', str(retain_data_percent),
                                                '.csv'])
                    # df_acc = pd.DataFrame.from_dict(all_classwise_acc)
                    # df_acc.to_csv(path_or_buf = classwise_accuracy_savepath, index=False)
                ########################################## Saving the accurcies ##########################################################
                    unlearn_retain_accuracy_savepath = ''.join([all_accuracy_savepath,#obj_model.result_savepath,
                                                datetime_now, '_',
                                                obj_model.model_name, '_',
                                                obj_model.data_name, '_',
                                                'unlearn_retain_accuracy_all',
                                                '_scllr_',str(unlearn_scale_lr).replace('.', 'o'),
                                                '_nunlep_', str(num_unlearn_epochs),
                                                '_lv_', str(local_variance),
                                                '_smul_', str(size_multiplier),
                                                '_p_', str(p).replace('.', 'o'),
                                                '_tran_', str(tail_randomized),
                                                '_ngnrtd_', str(no_generated_data),
                                                '_prcnt_', str(retain_data_percent),
                                                '.csv'])
                    print(f'Saving Accuracies in path = {unlearn_retain_accuracy_savepath}')
                    dict_acc = OrderedDict({'Unlearn_Class':list(unlearn_classes[:i+1]),
                                            'retain_train_acc_unlearn':retain_train_acc_unlearning[:i+1],
                                            'unlearn_train_acc_unlearn':unlearn_train_acc_unlearning[:i+1],
                                            'retain_test_acc_unlearn':retain_test_acc_unlearning[:i+1],
                                            'unlearn_test_acc_unlearn':unlearn_test_acc_unlearning[:i+1],
                                            
                                            'retain_train_acc_retrain':retain_train_acc_retrained[:i+1],
                                            'unlearn_train_acc_retrain':unlearn_train_acc_retrained[:i+1],
                                            'retain_test_acc_retrain':retain_test_acc_retrained[:i+1],
                                            'unlearn_test_acc_retrain':unlearn_test_acc_retrained[:i+1]
                                            })
                    df_acc = pd.DataFrame.from_dict(dict_acc)
                    df_acc.to_csv(path_or_buf = unlearn_retain_accuracy_savepath, index=False)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Saving Retain & Unlearn Accuracies over all dataset
                avg_train_retain_acc = round((sum(all_train_retain_acc)/num_uclasses),2)
                avg_train_unlearn_acc = round((sum(all_train_unlearn_acc)/num_uclasses),2)
                avg_test_retain_acc = round((sum(all_test_retain_acc)/num_uclasses),2)
                avg_test_unlearn_acc = round((sum(all_test_unlearn_acc)/num_uclasses),2)

                accuracy_savepath = ''.join([all_accuracy_savepath,#obj_model.result_savepath,
                                                obj_model.model_name, '_',
                                                obj_model.data_name, '_',
                                                'avg_train_test_accuracies_single_class',
                                                # '_scllr_',str(unlearn_scale_lr).replace('.', 'o'),
                                                # '_nunlep_', str(num_unlearn_epochs),
                                                # '_lv_', str(local_variance),
                                                # '_smul_', str(size_multiplier),
                                                # '_p_', str(p).replace('.', 'o'),
                                                # '_tlran_', str(tail_randomized),
                                                # '_ngnrtd_', str(no_generated_data),
                                                # '_prcnt_', str(retain_data_percent),
                                                '.csv'])
                dict_acc = OrderedDict({'_':['retain', 'unlearn', 'All'],
                                        'Train':[avg_train_retain_acc, avg_train_unlearn_acc, train_acc],
                                        'Test':[avg_test_retain_acc, avg_test_unlearn_acc, test_acc]})
                df_acc = pd.DataFrame.from_dict(dict_acc)
                df_acc.to_csv(path_or_buf = accuracy_savepath, index=False)

                # Average Retain & Unlearn Accuracies over all dataset after unlearning the Original Model   
                avg_retain_train_acc_unlearning = round((sum(retain_train_acc_unlearning)/num_uclasses),2)
                avg_unlearn_train_acc_unlearning = round((sum(unlearn_train_acc_unlearning)/num_uclasses),2)
                avg_retain_test_acc_unlearning = round((sum(retain_test_acc_unlearning)/num_uclasses),2)
                avg_unlearn_test_acc_unlearning = round((sum(unlearn_test_acc_unlearning)/num_uclasses),2)

                # Average Retain & Unlearn Accuracies over all dataset after retraining the Model   
                avg_retain_train_acc_retrained = round((sum(retain_train_acc_retrained)/num_uclasses),2)
                avg_unlearn_train_acc_retrained = round((sum(unlearn_train_acc_retrained)/num_uclasses),2)
                avg_retain_test_acc_retrained = round((sum(retain_test_acc_retrained)/num_uclasses),2)
                avg_unlearn_test_acc_retrained = round((sum(unlearn_test_acc_retrained)/num_uclasses),2)

                # Saving the Avg Accuracies in the existing .csv file
                avg_results = OrderedDict({'Unlearn_Class':['Mean'],
                                        'retain_train_acc_unlearn':[avg_retain_train_acc_unlearning],
                                        'unlearn_train_acc_unlearn':[avg_unlearn_train_acc_unlearning],
                                        'retain_test_acc_unlearn':[avg_retain_test_acc_unlearning],
                                        'unlearn_test_acc_unlearn':[avg_unlearn_test_acc_unlearning],
                                        'retain_train_acc_retrain':[avg_retain_train_acc_retrained],
                                        'unlearn_train_acc_retrain':[avg_unlearn_train_acc_retrained],
                                        'retain_test_acc_retrain':[avg_retain_test_acc_retrained],
                                        'unlearn_test_acc_retrain':[avg_unlearn_test_acc_retrained]
                                        })
                avg_results = pd.DataFrame.from_dict(avg_results)
                avg_results.to_csv(unlearn_retain_accuracy_savepath, mode='a', index=False, header=False)
                
                #Calculating the Standard Deviation
                df = pd.read_csv(unlearn_retain_accuracy_savepath)       
                if 'Std' not in list(df['Unlearn_Class']):
                    std_dict = OrderedDict({'Unlearn_Class':'Std'})
                    for header in df.columns[1:]:
                        std = round_up(df.loc[:len(df)-2, header].std(), 2)
                        mean = round_up(df.loc[:len(df)-2, header].mean(), 2)
                        std_dict[header] = std
                    df = df._append(std_dict, ignore_index = True)
                    df.to_csv(path_or_buf=unlearn_retain_accuracy_savepath, index = False)
                    
                    
                print(f'Average Retain loss on Train Data after Unlearning = {avg_retain_train_acc_unlearning}')
                print(f'Average Unlearn loss on Train Data after Unlearning = {avg_unlearn_train_acc_unlearning}')
                print(f'Average Retain loss on Test Data after Unlearning = {avg_retain_test_acc_unlearning}')
                print(f'Average Unlearn loss on Test Data after Unlearning = {avg_unlearn_test_acc_unlearning}')

                print(f'Average Retain loss on Train Data after Retraining = {avg_retain_train_acc_retrained}')
                print(f'Average Unlearn loss on Train Data after Retraining = {avg_unlearn_train_acc_retrained}')
                print(f'Average Retain loss on Test Data after Retraining = {avg_retain_test_acc_retrained}')
                print(f'Average Unlearn loss on Test Data after Retraining = {avg_unlearn_test_acc_retrained}')
                
                # print(f'Time needed for total unlearning: Mean = {np.array(unlearn_time_list).mean()}, Std = {np.array(unlearn_time_list).std()}')
                
                print('-'*140)
                print('='*140)
                print('#'*140)
                
        except Exception as e:
            with open(f'{all_accuracy_savepath}/{datetime_now}_runtime_error.txt', 'w') as f:
                f.write(str(e))
            print(f'Error in Unlearning the Model: {str(e)}')
            print('-'*80)
            print('='*80)
            print('\n\n')
            continue
