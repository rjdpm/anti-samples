#!/usr/bin/env python3

##############################################################################################
#% WRITER: RAJDEEP MONDAL            DATE: 20-11-2024
#% For bug and others mail me at: rdmondalofficial@gmail.com
#%--------------------------------------------------------------------------------------------
##############################################################################################


import os
import numpy as np
import copy
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils_unlearn import *
from UNMUNGE import *
from load_datasets import *
from config import *
from dataloader import *

##################################################### Inputs #################################
#---------------------------------------------------------------------------------------------
dataset_name = 'cifar10'#'cifar10', 'svhn', 'mnist' , 'fashionMNIST', 'cifar100'#
model_name = 'MobileNet_v2'#'ResNet9', 'LeNet32', 'AllCNN', 'ResNet18', 'MobileNet_v2'#
retain_data_percent = 30#100#
unlearn_type = 'Single_Class_Unlearn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrained_models_folder_name = 'retrained_models'
unlearned_models_folder_name = 'unlearned_models'
delete_saved_unlearned_models = True
#---------------------------------------------------------------------------------------------

# Loading configurations for a particular model and dataset
config = config_all[model_name][dataset_name]
all_result_folder_path = './results'
result_savepath = ''.join([all_result_folder_path, '/', dataset_name, '/', model_name, '/'])
unlearned_models_path = ''.join([result_savepath, '/', unlearn_type, '/', unlearned_models_folder_name])
#=============================================================================================

# Number of datapoints selected from each retain class for unlearning
if dataset_name == 'mnist' or dataset_name == 'fashionMNIST':
    num_input_channels = 1
    num_classes = 10
    padding = 2
elif dataset_name == 'svhn' or dataset_name == 'cifar10':
    num_input_channels = 3
    num_classes = 10
    padding = 0
else:
    print('Details about data not found.')
    
####################################### Parameters ############################################

learning_rate = config['learning_rate']
unlearn_scale_lr = config['unlearn_scale_lr']
batch_size = config['batch_size']
num_train_epochs = config['num_train_epochs']
num_unlearn_epochs = config['num_unlearn_epochs']
local_variance = config['local_variance']
size_multiplier = config['size_multiplier']
p = config['p']
tail_randomized=config['tail_randomized']
solver_type = config['solver_type']
no_generated_data = config['no_generated_data']
convex_combination = True
eps = 0.01

print('\n\n')
print('-'*80)
print(f'Dataset = {dataset_name}')
print(f'Model = {model_name}')
print(f'Learning Rate = {learning_rate}')
print(f'Scaling factor of learning rate for Unlearning = {unlearn_scale_lr}')
print(f'Batch Size = {batch_size}')
print(f'Number of Training Epochs = {num_train_epochs}')
print(f'Number of Unlearning Epochs = {num_unlearn_epochs}')
print(f'Percentage of Retain Data selected from each class = {retain_data_percent}')
print(f'Local Varience = {local_variance}')
print(f'Size Multiplier = {size_multiplier}')
print(f'Acceptance Probability of the Unmunged Samples = {p}')
print(f'Tail Randomizer = {tail_randomized}')
print(f'Solver Type = {solver_type}')
print(f'Number of Generated Data = {no_generated_data}')
print('-'*80)
print('='*80)
print('\n\n')

####################################### Loading Datasets ######################################

# For GPU's
datapath = '/home/rajdeep/workspace/Codes/Datasets/Torchvision_Data/'
if dataset_name == 'svhn':
    datapath = ''.join([datapath, 'SVHN_Data/'])
if dataset_name == 'cifar100':
    datapath = ''.join([datapath, 'CIFAR100/'])

#----------------------------------------------------------------------------------------------

train_data, test_data = dict_datasets[dataset_name](datapath)
train_loader = DataLoader(train_data,
                          batch_size,
                          shuffle=True,
                          num_workers = 4
                          )
test_loader = DataLoader(test_data,
                         batch_size,
                         shuffle=True,
                         num_workers = 4
                         )
image_size = train_loader.dataset[0][0].shape

print('='*80)
print('Data stats:')
print('-'*80)
print('Train data size (num_samples x 1-sample size): {} x {}' .format(len(train_data), train_data[0][0].shape))
print('Test data size (num_samples x 1-sample size): {}  x {}' .format(len(test_data), test_data[0][0].shape))
print('-'*80)
print('='*80)
#----------------------------------------------------------------------------------------------

########################################## Data Partition #####################################

# Separating Data and Labels from Dataloader
print('Separating Data and Labels from Dataloader:')
print('-------------------------------------------')

input_train_data = []
input_train_labels = []

input_test_data = []
input_test_labels = []

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
#----------------------------------------------------------------------------------------------

unlearn_train_acc_unlearning = [None]*num_classes
retain_train_acc_unlearning = [None]*num_classes
unlearn_test_acc_unlearning = [None]*num_classes
retain_test_acc_unlearning = [None]*num_classes

retain_train_acc_retrained = [None]*num_classes
unlearn_train_acc_retrained = [None]*num_classes
retain_test_acc_retrained = [None]*num_classes
unlearn_test_acc_retrained = [None]*num_classes

all_train_retain_acc = [None]*num_classes
all_train_unlearn_acc = [None]*num_classes
all_test_retain_acc = [None]*num_classes
all_test_unlearn_acc = [None]*num_classes

unlearn_time_list = [None]*num_classes

all_classwise_acc = OrderedDict({'Classes':list(range(num_classes))})
#----------------------------------------------------------------------------------------------
#==============================================================================================
    
for cls in tqdm(range(num_classes)):

    unlearn_cls = cls#config['unlearn_cls']#0
    ## Creating Utility Object ##
    obj_model = utils_(image_size=(image_size[1], image_size[2]),
                        num_input_channels = image_size[0],
                        num_classes = num_classes,
                        learning_rate = learning_rate,
                        batch_size = batch_size,
                        num_epochs = num_train_epochs,
                        padding = padding,
                        model_save_name = '',
                        data_name = dataset_name,
                        model_name=model_name,
                        unlearn_cls = unlearn_cls,
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
    ############################### Loading & Testing the model ###############################

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
    all_train_retain_acc[unlearn_cls] = train_retain_acc
    all_train_unlearn_acc[unlearn_cls] = train_unlearn_acc
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
    all_test_retain_acc[unlearn_cls] = test_retain_acc
    all_test_unlearn_acc[unlearn_cls] = test_unlearn_acc
    all_classwise_acc['Test'] = test_classwise_accuracy
    print('-'*80)
    print('='*80)
    print('\n\n')
    #------------------------------------------------------------------------------------------

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
                                               shuffle=True,
                                               num_workers = 4
                                               )
    temp = list(zip(retain_test_data, retain_test_labels))
    obj_model.retain_loader_test = DataLoader(temp,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = 4
                                              )
    print(f'Number of retain data = {len(retain_train_data)} and unlearn data = {len(unlearn_data)}')
    print('Separation complete.\n')
    print('-'*80)
    print('='*80)
    print('\n\n')
    #------------------------------------------------------------------------------------------
    ###########################################################################################

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
    print(f'Number of selected Retained Data = {len(unlearn_retain_data)}\n')
    print(f'Retained labels are: {labels}')
    print('-'*80)
    print('='*80)
    print('\n\n')
    #------------------------------------------------------------------------------------------
    ################################### Unlearning the Model ##################################
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
        ############################ Generating UNMUNGED Samples ##############################

        retain_train_data = retain_train_data.reshape(len(retain_train_data), -1)
        unlearn_retain_data = unlearn_retain_data.reshape(len(unlearn_retain_data), -1)
        unlearn_data = unlearn_data.reshape(len(unlearn_data), -1)
        print('\n\nGenerating UNMUNGED Samples:')
        print('----------------------------')
        
        print('='*80)
        antisamples, _, _, antisample_generation_time = UNMUNGE_(unlearn_data= unlearn_data,
                                                retain_data=retain_train_data,#unlearn_retain_data,
                                                retain_labels=retain_train_labels,#unlearn_retain_labels,
                                                local_variance = local_variance,
                                                size_multiplier = size_multiplier,
                                                p = p,
                                                tail_randomized = tail_randomized,
                                                no_generated_data=no_generated_data,
                                                eps=eps,
                                                convex_combination=convex_combination
                                                )


        data_shape = (len(retain_train_data), ) + input_train_data.shape[1:]
        retain_train_data = retain_train_data.reshape(*data_shape)

        data_shape = (len(unlearn_retain_data), ) + input_train_data.shape[1:]
        unlearn_retain_data = unlearn_retain_data.reshape(*data_shape)

        data_shape = (len(antisamples), ) + input_train_data.shape[1:]
        antisamples = antisamples.reshape(*data_shape)
        
        unlearn_labels = torch.tensor(unlearn_cls).repeat(len(antisamples))
        unlearn_data = unlearn_data.reshape(*unlearn_data_shape)
        print('Antisample generation complete.')
        print('-------------------------------')
        print('\n\n')
        #------------------------------------------------------------------------------------------


        # Creating Dataloader only for Unlearned Data
        temp = list(zip(unlearn_data,unlearn_labels))
        obj_model.unlearn_dataloader = DataLoader(temp,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers = 4
                                                )
        #------------------------------------------------------------------------------------------
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        # Merging the Selected Retained Data and Generated UNMUNGED Data
        #---------------------------------------------------------------
        ##for selected retain data:
        retain_data = unlearn_retain_data 
        retain_labels = unlearn_retain_labels
        
        ##for full retain data:
        # retain_data = retain_train_data  
        # retain_labels = retain_train_labels
        
        train_unlearn_images =  torch.concat((retain_data, antisamples))
        train_unlearn_labels = torch.concat((retain_labels, unlearn_labels))
        temp = list(zip(train_unlearn_images,train_unlearn_labels))
        obj_model.unlearn_loader_all = DataLoader(temp,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers = 4
                                                )
        #------------------------------------------------------------------------------------------

        optimizer = optim.Adam(params=obj_model.network.parameters(), lr = unlearn_scale_lr*learning_rate)
        model_unlearn, obj_model.unlearn_epoch_train_losses, unlearn_time = obj_model.unlearn(model=obj_model.network,
                                                            optimizer= optimizer,
                                                            dataloader=obj_model.unlearn_loader_all,
                                                            num_epochs=num_unlearn_epochs
                                                            )
        total_time = unlearn_time + antisample_generation_time
        unlearn_time_list[unlearn_cls] = total_time
    else:
        print(f'Found existing unlearned model in: {obj_model.best_unlearn_model_save_path}')
    print(f'Loading unlearned model from: {obj_model.best_unlearn_model_save_path}')
    unlearn_model.load_unlearn_network()
    unlearn_model.network.eval()
    print('-'*80)
    print('\n\n')
    #----------------------------------------------------------------------------------------------
    #################################### Testing the Unlearned Model ##############################
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
    
    retain_train_acc_unlearning[cls] = train_retain_acc_cls
    unlearn_train_acc_unlearning[cls] = train_unlearn_acc_cls
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
    
    retain_test_acc_unlearning[cls] = test_retain_acc_cls
    unlearn_test_acc_unlearning[cls] = test_unlearn_acc_cls
    all_classwise_acc['Test_Unlearn_'+str(unlearn_cls)] = unlearn_test_classwise_accuracy
    print('-'*120)
    print('='*120)
    print('\n\n')
    #------------------------------------------------------------------------------------------------------
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                
    
    ########################################## Retraining the Model #######################################
    
    print(f'\n\n############################# Retraining the Model for Unlearn Class - {unlearn_cls} #########################################################')
    print('=======================================================================================================================================')
    temp =list(zip(retain_train_data,retain_train_labels))
    retrain_model.train_data = temp
    temp = list(zip(retain_test_data,retain_test_labels))
    retrain_model.test_data = temp
    
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
    
    retain_train_acc_retrained[cls] = retrain_train_retain_acc_cls ###Accuracy of Retrained Model on Retain Data in Train Dataset
    unlearn_train_acc_retrained[cls] = retrain_train_unlearn_acc_cls ###Accuracy of Retrained Model on Unlearn Data in Train Dataset
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
    
    retain_test_acc_retrained[cls] = retrain_test_retain_acc_cls
    unlearn_test_acc_retrained[cls] = retrain_test_unlearn_acc_cls
    all_classwise_acc['Test_Retrain_'+str(unlearn_cls)] = retrain_test_classwise_accuracy
    
    ## Saving Accuracies ##
    all_accuracy_savepath = ''.join([obj_model.result_savepath,
                                     unlearn_type,
                                     '_results',
                                     '/'
                                     ])
    obj_model.create_folder(all_accuracy_savepath)
    classwise_accuracy_savepath = ''.join([all_accuracy_savepath,
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
    df_acc = pd.DataFrame.from_dict(all_classwise_acc)
    # df_acc.to_csv(path_or_buf = classwise_accuracy_savepath, index=False)
########################################## Saving the accurcies ##########################################################
    unlearn_retain_accuracy_savepath = ''.join([all_accuracy_savepath,
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
    dict_acc = OrderedDict({'Unlearn_Class':list(range(cls+1)),
                             'retain_train_acc_unlearn':retain_train_acc_unlearning[:cls+1],
                             'unlearn_train_acc_unlearn':unlearn_train_acc_unlearning[:cls+1],
                             'retain_test_acc_unlearn':retain_test_acc_unlearning[:cls+1],
                             'unlearn_test_acc_unlearn':unlearn_test_acc_unlearning[:cls+1],
                             
                             'retain_train_acc_retrain':retain_train_acc_retrained[:cls+1],
                             'unlearn_train_acc_retrain':unlearn_train_acc_retrained[:cls+1],
                             'retain_test_acc_retrain':retain_test_acc_retrained[:cls+1],
                             'unlearn_test_acc_retrain':unlearn_test_acc_retrained[:cls+1]
                             })
    df_acc = pd.DataFrame.from_dict(dict_acc)
    df_acc.to_csv(path_or_buf = unlearn_retain_accuracy_savepath, index=False)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Saving Retain & Unlearn Accuracies over all dataset
avg_train_retain_acc = round((sum(all_train_retain_acc)/num_classes),2)
avg_train_unlearn_acc = round((sum(all_train_unlearn_acc)/num_classes),2)
avg_test_retain_acc = round((sum(all_test_retain_acc)/num_classes),2)
avg_test_unlearn_acc = round((sum(all_test_unlearn_acc)/num_classes),2)

accuracy_savepath = ''.join([all_accuracy_savepath,
                                obj_model.model_name, '_',
                                obj_model.data_name, '_',
                                'avg_train_test_accuracies_single_class',
                                '.csv'])
dict_acc = OrderedDict({'_':['retain', 'unlearn', 'All'],
                        'Train':[avg_train_retain_acc, avg_train_unlearn_acc, train_acc],
                        'Test':[avg_test_retain_acc, avg_test_unlearn_acc, test_acc]})
df_acc = pd.DataFrame.from_dict(dict_acc)
df_acc.to_csv(path_or_buf = accuracy_savepath, index=False)

# Average Retain & Unlearn Accuracies over all dataset after unlearning the Original Model   
avg_retain_train_acc_unlearning = round((sum(retain_train_acc_unlearning)/num_classes),2)
avg_unlearn_train_acc_unlearning = round((sum(unlearn_train_acc_unlearning)/num_classes),2)
avg_retain_test_acc_unlearning = round((sum(retain_test_acc_unlearning)/num_classes),2)
avg_unlearn_test_acc_unlearning = round((sum(unlearn_test_acc_unlearning)/num_classes),2)

# Average Retain & Unlearn Accuracies over all dataset after retraining the Model   
avg_retain_train_acc_retrained = round((sum(retain_train_acc_retrained)/num_classes),2)
avg_unlearn_train_acc_retrained = round((sum(unlearn_train_acc_retrained)/num_classes),2)
avg_retain_test_acc_retrained = round((sum(retain_test_acc_retrained)/num_classes),2)
avg_unlearn_test_acc_retrained = round((sum(unlearn_test_acc_retrained)/num_classes),2)

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

print(f'Time needed for total unlearning: Mean = {np.array(unlearn_time_list).mean()}, Std = {np.array(unlearn_time_list).std()}')
print('-'*140)
print('='*140)
print('#'*140)
