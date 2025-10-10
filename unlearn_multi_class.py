#!/usr/bin/env python3
import os, sys
import pickle
import numpy as np
import copy
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

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

dataset_name_list = ['casia-webface']#['cifar10', 'svhn', 'mnist', 'fashionMNIST']#
model_name_list = ['MobileNet_v2']#, 'AllCNN', 'MobileNet_v2', 'ResNet18']

# dataset_name_list = dataset_name_list[::-1]
# model_name_list = model_name_list[::-1]

## Number of datapoints selected from each retain class for unlearning
retain_data_percent_list = [100]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unlearn_type = 'Multiclass_Unlearning'
delete_saved_unlearned_models = True#False
retrained_models_folder_name = 'retrained_models'
unlearned_models_folder_name = 'unlearned_models'
#---------------------------------------------------------------------------------------------------------------------------------
#=================================================================================================================================
for dataset_name in tqdm(dataset_name_list):
    for model_name in tqdm(model_name_list):
        # try:
        # Number of datapoints selected from each retain class for unlearning
        if dataset_name == 'mnist' or dataset_name == 'fashionMNIST':
            num_input_channels = 1
            num_classes = 10
            padding = 2
            # H, W = image_size[0], image_size[1]
        elif dataset_name == 'svhn' or dataset_name == 'cifar10':
            num_input_channels = 3
            num_classes = 10
            if dataset_name == 'cifar100':
                num_classes = 20#100
            padding = 0
        elif dataset_name == 'casia-webface':
            num_input_channels = 3
            num_classes = 300
            padding = 0
            # random_uclasses_path = ''.join([result_savepath, '/', unlearn_type, '/', 'unlearn_classes.json'])
        else:
            print('Details about data not found.')

        #----------------------------------------------------------------------------------------------------------------------------------
        # Loading configurations for a particular model and dataset
        config = config_all[model_name][dataset_name]
        all_result_folder_path = './results'
        result_savepath = ''.join([all_result_folder_path, '/', dataset_name, '/', model_name, '/'])
        unlearned_models_path = ''.join([result_savepath, '/', unlearn_type, '/', unlearned_models_folder_name])
        all_accuracy_savepath = ''.join([result_savepath, unlearn_type, '_results/'])
        create_folder(all_accuracy_savepath)
        #==================================================================================================================================
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
            unlearn_cls = 'multiclass_unlearn'
            num_multiclass_list = [4]#[2, 4, 7, 10]
            averaging_epochs_list = [5]#[10, 5, 3, 3]# [10, 10, 10, 10]#[10, 5, 3]
            # if dataset_name == 'casia-webface':
            #     num_multiclass_list = list(np.array(num_multiclass_list)*2)
                
            # Saving and Loading random unlearn classes 
            random_classes_path = all_result_folder_path + '/'+dataset_name+'/'+model_name+'/'
            random_classes_filename = random_classes_path+model_name+'_'+dataset_name+'_random_unlearn_classes.pkl'
            if not os.path.isdir(random_classes_path):
                os.makedirs(random_classes_path)

            if not os.path.isfile(random_classes_filename):
                all_unlearn_classes = [None]*len(averaging_epochs_list)
                print('No predefined unlearn classes.')
                for j in range(len(averaging_epochs_list)):
                    all_unlearn_classes[j] = [list(np.random.choice(range(num_classes), num_multiclass_list[j], replace=False)) for i in range(averaging_epochs_list[j])]
                    # unlearn_classes = list(np.random.choice(range(num_classes), num_multiclass, replace=False))#list(range(num_classes))[:num_multiclass]#
                with open(random_classes_filename, 'wb') as fp:
                    pickle.dump(all_unlearn_classes, fp)
                list2json(input_list=all_unlearn_classes,
                            filename=model_name+'_'+dataset_name+'_random_unlearn_classes',
                            filepath=random_classes_path
                            )
                
            with open(random_classes_filename, 'rb') as fp:
                all_unlearn_classes = pickle.load(fp)
                print(f'Loding pre-defined unlearn classes from: {random_classes_filename}')
            print(f'List of unlearn classes are:\n {all_unlearn_classes}')
            print('-'*80)
            print('='*80)
            
                            
            for multiclass_idx in range(len(num_multiclass_list)):
               
                num_multiclass = num_multiclass_list[multiclass_idx]
                averaging_epochs = averaging_epochs_list[multiclass_idx]
                
                unlearn_train_acc_unlearning = [None]*averaging_epochs
                retain_train_acc_unlearning = [None]*averaging_epochs
                unlearn_test_acc_unlearning = [None]*averaging_epochs
                retain_test_acc_unlearning = [None]*averaging_epochs

                retain_train_acc_retrained = [None]*averaging_epochs
                unlearn_train_acc_retrained = [None]*averaging_epochs
                retain_test_acc_retrained = [None]*averaging_epochs
                unlearn_test_acc_retrained = [None]*averaging_epochs

                all_train_retain_acc = [None]*averaging_epochs
                all_train_unlearn_acc = [None]*averaging_epochs
                all_test_retain_acc = [None]*averaging_epochs
                all_test_unlearn_acc = [None]*averaging_epochs
                Unlearn_Class = [None]*averaging_epochs
                
                unlearn_time_list = np.array([0.]*averaging_epochs)
                
                all_classwise_acc = OrderedDict({'Classes':list(range(num_classes))})
                ################################################### Printing Used Parameters ####################################################
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
                print('-'*80)
                print('='*80)
                print('\n\n')

                ################################################ Loading Datasets ##############################################################
                # For local system
                # datapath = '/home/dell/Workspace/Codes/Datasets/Torchvision_Data/'

                # For GPU's
                datapath = '/home/rajdeep/workspace/Datasets/Torchvision_Data/'
                if dataset_name == 'svhn':
                    datapath = ''.join([datapath, 'SVHN_Data/'])
                if dataset_name == 'casia-webface':
                    datapath = '/home/rajdeep/workspace/Datasets/CASIA-WebFace/casia-webface-dataset.pkl.gz'


                train_data, test_data = dict_datasets[dataset_name](datapath)# Try to write as a if else with manual function
                train_loader = DataLoader(train_data,
                            batch_size,
                            shuffle=True
                            )
                test_loader = DataLoader(test_data,
                            batch_size,
                            shuffle=True
                            )
                image_size = train_loader.dataset[0][0].shape

                # obj_model.unlearn_cls = unlearn_cls
                print('='*80)
                print('Data stats:')
                print('-'*80)
                print('Train data size (num_samples x 1-sample size): {} x {}' .format(len(train_data), train_data[0][0].shape))
                print('Test data size (num_samples x 1-sample size): {}  x {}' .format(len(test_data), test_data[0][0].shape))
                print('-'*80)
                print('='*80)
                #-------------------------------------------------------------------------------------------------------------------------------

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

                input_train_data = torch.stack(input_train_data)
                input_train_labels = torch.from_numpy(np.array(input_train_labels))
                input_test_data = torch.stack(input_test_data)
                input_test_labels = torch.from_numpy(np.array(input_test_labels))
                print(f'Number of Train Data = {len(input_train_data)}, Number of Test Data = {len(input_test_data)}\n')
                print('-'*80)
                print('='*80)
                print('\n\n')
                #-----------------------------------------------------------------------------------------------------------------------------

                for epoch in tqdm(range(averaging_epochs)):
                    
                    unlearn_classes = all_unlearn_classes[multiclass_idx][epoch]
                    print(f'Unlearn Classes: {unlearn_classes}')
                    unlearn_classes_name = 'multiclass_' + str(unlearn_classes).replace('[', '',).replace(']', '').replace(', ', '_')

                    ## Fixing seed for Reproduciblity ##
                    np_seed = config['np_seed']
                    torch_seed = config['torch_seed']
                    np.random.seed(np_seed)#60#20#30
                    torch.manual_seed(torch_seed)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.manual_seed_all(torch_seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False

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
                                    unlearn_cls = unlearn_classes_name,
                                    solver_type = solver_type,
                                    result_savepath = result_savepath,
                                    retrained_models_folder_name = retrained_models_folder_name,
                                    unlearned_models_folder_name = unlearned_models_folder_name,
                                    unlearn_type=unlearn_type
                                    )
                    
                    
                    #Initializing Retraining Model
                    retrain_model = copy.deepcopy(obj_model)

                    #Initializing Retraining Model
                    unlearn_model = copy.deepcopy(obj_model)
                    
                    obj_model.train_data  = train_data
                    obj_model.test_data = test_data
                    #--------------------------------------------------------------------------------------------------------------------------
                    ############################################## Loading & Testing the model ################################################

                    if not os.path.isfile(obj_model.best_model_save_path):
                        print(f'No pre-trained model found in: {obj_model.best_model_save_path}')
                        obj_model.count_epoch = 0
                        obj_model.train()
                    
                    # Loading Original Model
                    obj_model.load_network()
                    obj_model.network.eval()

                    train_loader = DataLoader(obj_model.train_data, obj_model.batch_size, shuffle=True)
                    test_loader = DataLoader(obj_model.test_data, obj_model.batch_size, shuffle=True)
                    
                    print('Testing main model on train data:\n')
                    confusion_matrix, train_acc, train_classwise_accuracy, train_retain_acc, train_unlearn_acc = accuracy_multiclass_unlearn(obj_model.network,
                                                                                                                                            train_loader,
                                                                                                                                            unlearn_classes=unlearn_classes)
                    print(f'Test accuracy on Train data = {train_acc}')  
                    print(f'Accuracy on Train retain Data = {train_retain_acc}')
                    print(f'Accuracy on Train unlearn Data = {train_unlearn_acc}')
                    print(f'Test classwise accuracy on Train data = {train_classwise_accuracy}\n')
                    all_train_retain_acc[epoch] = train_retain_acc
                    all_train_unlearn_acc[epoch] = train_unlearn_acc
                    all_classwise_acc['Train'] = train_classwise_accuracy
                    
                    print('Testing main model on test data:\n')
                    confusion_matrix, test_acc, test_classwise_accuracy, test_retain_acc, test_unlearn_acc = accuracy_multiclass_unlearn(obj_model.network,
                                                                                                                                        test_loader,
                                                                                                                                        unlearn_classes=unlearn_classes)
                    print(f'Test accuracy on Test data = {test_acc}')
                    print(f'Accuracy on Test retain Data = {test_retain_acc}')
                    print(f'Accuracy on Test unlearn Data = {test_unlearn_acc}')
                    print(f'Test classwise accuracy on Test data = {test_classwise_accuracy}\n')
                    all_test_retain_acc[epoch] = test_retain_acc
                    all_test_unlearn_acc[epoch] = test_unlearn_acc
                    all_classwise_acc['Test'] = test_classwise_accuracy
                    print('-'*80)
                    print('='*80)
                    print('\n\n')
                    #---------------------------------------------------------------------------------------------------------------------------

                    #Separating Unlearn and Retain Data
                    print('Separating Unlearn and Retained Data from the full training data:')
                    print('-----------------------------------------------------------------')
                    retain_train_data = input_train_data
                    retain_train_labels = input_train_labels
                    retain_test_data = input_test_data
                    retain_test_labels = input_test_labels
                    unlearn_data = []
                    unlearn_labels = []
                    for label in unlearn_classes:
                        idx_train = retain_train_labels == label
                        idx_test = retain_test_labels == label
                        forget_set = retain_train_data[idx_train]
                        forget_labels = retain_train_labels[idx_train]
                        unlearn_data.append(forget_set)
                        unlearn_labels.append(forget_labels)
                        retain_train_data = retain_train_data[~idx_train]
                        retain_train_labels = retain_train_labels[~idx_train]
                        retain_test_data = retain_test_data[~idx_test]
                        retain_test_labels = retain_test_labels[~idx_test]
                    idx_train, idx_test, forget_set, forget_labels = None, None, None, None
                        
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

                    #----------------------------------------------------------------------------------------------------------------------------------------------------
                    #Dataloader only for retain data
                    # obj_model.retain_loader = DataLoader(list(zip(retain_train_data,retain_train_labels)), batch_size=batch_size, shuffle=True)

                    #Dataloader only for unlearn data
                    obj_model.forget_loader = DataLoader(list(zip(torch.from_numpy(np.concatenate(unlearn_data)),
                                                                torch.from_numpy(np.concatenate(unlearn_labels)))),
                                                        batch_size=batch_size,
                                                        shuffle=True)
                    print(f'Number of retain data = {len(retain_train_data)} and unlearn data = {len(np.concatenate(unlearn_data))}')
                    print('Separation complete.\n')
                    print('-'*80)
                    print('='*80)
                    print('\n\n')
                    #-----------------------------------------------------------------------------------------------------------------------------------------------------
                    #Chossing random subsets of fixed size from each retain classes
                    print(f'Selecting {retain_data_percent}-percent random samples from each classes:')
                    retain_data = []
                    unlearn_retain_labels = []
                    train_index = []
                    labels = list(set(retain_train_labels.numpy()))
                    # print(labels)
                    for label in labels:
                        idx = (input_train_labels == label)
                        temp = np.where(idx == True)[0]
                        no_retain_data = int(len(temp)*(retain_data_percent/100))
                        data_idx = np.random.permutation(temp)[:no_retain_data]
                        train_index.extend(data_idx)#Position of the selected images in the training dataset
                        cls_data = input_train_data[data_idx]
                        cls_labels = input_train_labels[data_idx]
                        retain_data.extend(cls_data)
                        unlearn_retain_labels.extend(cls_labels)     
                    # print(retain_data)
                    unlearn_retain_data = torch.stack(retain_data) # Contains 'no_retain_data' number of data from each class
                    # print(unlearn_retain_data.shape)
                    unlearn_retain_labels = torch.from_numpy(np.array(unlearn_retain_labels))
                    no_generated_data = no_retain_data
                    print(f'Number of selected Retained Data = {len(unlearn_retain_data)}\n')
                    print(f'Number of Generated MUNGE Data = {no_generated_data}\n')
                    print(f'Retained labels are: {labels}')
                    print('-'*80)
                    print('='*80)
                    print('\n\n')
                    #-------------------------------------------------------------------------------------------------------------------------------      
                    
                    
                    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    ##################################################################  Unlearning the model ############################################################################       
                    print('='*80)
                    print('-'*80)
                    print(f'Unlearning Step(Classes = {unlearn_classes}):')
                    print('---------------------------------------------\n')
                    
                    ## Delete Pre-Saved Unlearned Models
                    if delete_saved_unlearned_models:
                        print('='*80)
                        if not os.path.isfile(obj_model.best_unlearn_model_save_path):
                            print('Model not exists.')
                        else:
                            print(f'\nDeleting Pre-Saved Unlearned Model:{obj_model.best_unlearn_model_save_path}')
                            os.remove(obj_model.best_unlearn_model_save_path)
                            print(f'Model deleted:{obj_model.best_unlearn_model_save_path}')
                        print('-'*80)
                        print('='*80)
                    #############################################  Unlearning the model ################################################################
                    unlearn_model.result_savepath = obj_model.result_savepath_unlearned
                    if not os.path.isfile(obj_model.best_unlearn_model_save_path):
                        print(f'No Unlearned model found in: {obj_model.best_unlearn_model_save_path}')
                        print('Unlearning the Model:\n')
                        
                        ############################################### Generating UNMUNGED Samples ####################################################
                        retain_train_data = retain_train_data.reshape(len(retain_train_data), -1)
                        unlearn_retain_data = unlearn_retain_data.reshape(len(unlearn_retain_data), -1)
                        print('Generating UNMUNGED Samples:')
                        print('----------------------------')
                        unmunge_data = []
                        unmunge_labels = []
                        for data_idx in range(len(unlearn_data)):
                            # print(data.shape)
                            data = unlearn_data[data_idx].reshape(len(unlearn_data[data_idx]), -1)
                            unmunge, farthest_point, pairwise_distance, unmunge_time = generate_antisamples(unlearn_data= data,
                                                                                retain_data=retain_train_data,#unlearn_retain_data,#
                                                                                retain_labels=retain_train_labels,#unlearn_retain_labels, #
                                                                                local_variance = local_variance,
                                                                                size_multiplier = size_multiplier,
                                                                                p = p,
                                                                                tail_randomized = tail_randomized,
                                                                                no_generated_data=no_generated_data,
                                                                                eps=0.01,
                                                                                convex_combination=True)
                            unmunge_data.extend(unmunge.numpy())
                            unmunge_labels.extend([unlearn_classes[data_idx]]*len(unmunge))

                        data_shape = (len(retain_train_data), ) + input_train_data.shape[1:]
                        retain_train_data = retain_train_data.reshape(*data_shape)

                        data_shape = (len(unlearn_retain_data), ) + input_train_data.shape[1:]
                        unlearn_retain_data = unlearn_retain_data.reshape(*data_shape)

                        data_shape = (len(unmunge_data), ) + input_train_data.shape[1:]
                        unmunge_data = torch.from_numpy(np.array(unmunge_data).reshape(*data_shape))
                        unmunge_labels = torch.from_numpy(np.array(unmunge_labels))
                        
                        unlearn_data = torch.from_numpy(np.concatenate(unlearn_data))
                        unlearn_labels = torch.from_numpy(np.concatenate(unlearn_labels))
                        print('Antisample generation complete.')
                        print('-------------------------------')
                        print('='*60)
                        print('\n\n')
                        
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
                        
                        train_unlearn_images = torch.concat((retain_data, unmunge_data))
                        train_unlearn_labels = torch.concat((retain_labels, unmunge_labels))
                        temp = list(zip(train_unlearn_images,train_unlearn_labels))
                        obj_model.unlearn_loader_all = DataLoader(temp,
                                                                batch_size=batch_size,
                                                                shuffle=True)
                        #------------------------------------------------------------------------------------------------------------------------------
                        optimizer = optim.Adam(params=obj_model.network.parameters(), lr = unlearn_scale_lr*learning_rate)
                        model_unlearn, obj_model.unlearn_epoch_train_losses, total_time = obj_model.unlearn(model=obj_model.network,
                                                                            optimizer= optimizer,
                                                                            dataloader=obj_model.unlearn_loader_all,
                                                                            num_epochs=num_unlearn_epochs
                                                                            )
                        unlearn_time_list[epoch] = total_time
                        
                    else:
                        print(f'Found existing unlearned model in: {obj_model.best_unlearn_model_save_path}')

                    # Loading Existing Unlearned Model
                    unlearn_model.load_unlearn_network()
                    unlearn_model.network.eval()
                    #---------------------------------------------------------------------------------------------------------------------------------
                    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


                    ################################################## Testing the Unlearned Model ###################################################
                    print('Testing on Train Data:')
                    print('----------------------')
                    confusion_matrix, acc, unlearn_train_classwise_accuracy, train_retain_acc_cls, train_unlearn_acc_cls = accuracy_multiclass_unlearn(unlearn_model.network,
                                                                                                                                                    train_loader,
                                                                                                                                                    unlearn_classes=unlearn_classes
                                                                                                                                                    )
                    print(f'Classwise Accuracy on Train Data after Unlearning(Unlearn Classes - {unlearn_classes}):\n {unlearn_train_classwise_accuracy}\n')
                    print(f'Accuracy of Unlearned Model on Retain Data in Train Dataset(Unlearn Classes - {unlearn_classes}) = {train_retain_acc_cls}')
                    print(f'Accuracy of Unlearned Model on Unlearn Data in Train Dataset(Unlearn Classes - {unlearn_classes}) = {train_unlearn_acc_cls}')

                    retain_train_acc_unlearning[epoch] = train_retain_acc_cls
                    unlearn_train_acc_unlearning[epoch] = train_unlearn_acc_cls
                    all_classwise_acc['Train_Unlearn_'+unlearn_classes_name] = unlearn_train_classwise_accuracy

                    print('\n\nTesting on Test Data:')
                    print('---------------------')
                    confusion_matrix, acc, unlearn_test_classwise_accuracy, test_retain_acc_cls, test_unlearn_acc_cls = accuracy_multiclass_unlearn(unlearn_model.network,
                                                                                                                                test_loader,
                                                                                                                                unlearn_classes=unlearn_classes
                                                                                                                                )
                    print(f'Classwise Accuracy on Test Data after Unlearning(Unlearn Classes - {unlearn_classes}) :\n {unlearn_test_classwise_accuracy}\n')
                    print(f'Accuracy of Unlearned Model on Retain Data in Test Dataset(Unlearn Classes - {unlearn_classes}) = {test_retain_acc_cls}')
                    print(f'Accuracy of Unlearned Model on Unlearn Data in Test Dataset(Unlearn Classes - {unlearn_classes}) = {test_unlearn_acc_cls}')

                    retain_test_acc_unlearning[epoch] = test_retain_acc_cls
                    unlearn_test_acc_unlearning[epoch] = test_unlearn_acc_cls
                    all_classwise_acc['Test_Unlearn_'+unlearn_classes_name] = unlearn_test_classwise_accuracy
                    print('-'*120)
                    print('='*120)
                    print('\n\n')
                    #---------------------------------------------------------------------------------------------------------------------------------
                    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    
        
                    ############################################################ Retraining the Model #################################################################################

                    print(f'\n\n############################# Retraining the Model for Unlearn Classes - {unlearn_classes} #########################################################\n')
                    print('===========================================================================================================================================================')
                    temp =list(zip(retain_train_data,retain_train_labels))
                    retrain_model.train_data = temp
                    temp = list(zip(retain_test_data,retain_test_labels))
                    retrain_model.test_data = temp
                    
                    retrain_model.best_model_save_path = obj_model.best_retrain_model_save_path
                    retrain_model.result_savepath = obj_model.result_savepath_retrained
                    retrain_model.model_save_name = '_retrain_'+unlearn_classes_name
                    if not os.path.isfile(obj_model.best_retrain_model_save_path):
                        print(f'No Retrained model found in: {obj_model.best_retrain_model_save_path}')
                        retrain_model.count_epoch = 0
                        retrain_model.train()

                    retrain_model.load_network()
                    retrain_model.network.eval()
                    print('-'*80)
                    print('\n\n')


                    print(f'\nPerformance on Train Data after Retraining(U - {unlearn_classes})')
                    print('-----------------------------------------------------------------')
                    confusion_matrix, acc, retrain_train_classwise_accuracy, retrain_train_retain_acc_cls, retrain_train_unlearn_acc_cls = accuracy_multiclass_unlearn(retrain_model.network.to(device),
                                                                                                                                                                    train_loader,
                                                                                                                                                                    unlearn_classes=unlearn_classes)
                    print(f'Classwise Accuracy on Train Data after Retraining(Unlearn Classes - {unlearn_classes}) :\n {retrain_train_classwise_accuracy}')
                    print(f'Accuracy of Retrained Model on Retain Data in Train Dataset = {retrain_train_retain_acc_cls}')
                    print(f'Accuracy of Retrained Model on Unlearn Data in Train Dataset = {retrain_train_unlearn_acc_cls}')

                    retain_train_acc_retrained[epoch] = retrain_train_retain_acc_cls ###Accuracy of Retrained Model on Retain Data in Train Dataset
                    unlearn_train_acc_retrained[epoch] = retrain_train_unlearn_acc_cls ###Accuracy of Retrained Model on Unlearn Data in Train Dataset
                    all_classwise_acc['Train_Retrain_'+unlearn_classes_name] = retrain_train_classwise_accuracy
                    print('-'*80)


                    print(f'\n\nPerformance on Test Data after Retraining(U - {unlearn_classes})')
                    print('----------------------------------------------------------------')
                    confusion_matrix, acc, retrain_test_classwise_accuracy, retrain_test_retain_acc_cls, retrain_test_unlearn_acc_cls = accuracy_multiclass_unlearn(retrain_model.network.to(device),
                                                                                                                                                                    test_loader,
                                                                                                                                                                    unlearn_classes=unlearn_classes)
                    print(f'Classwise Accuracy on Test Data after Retraining(Unlearn Classes - {unlearn_classes}):\n{retrain_test_classwise_accuracy}')
                    print(f'Accuracy of Retrained Model on Retain Data in Test Dataset = {retrain_test_retain_acc_cls}')
                    print(f'Accuracy of Retrained Model on Unlearn Data in Test Dataset = {retrain_test_unlearn_acc_cls}')
                    print('-'*120)
                    print('='*120)
                    print('\n\n') 
                    #---------------------------------------------------------------------------------------------------------------------------------------------
                
                    retain_test_acc_retrained[epoch] = retrain_test_retain_acc_cls
                    unlearn_test_acc_retrained[epoch] = retrain_test_unlearn_acc_cls
                    all_classwise_acc['Test_Retrain_'+unlearn_classes_name] = retrain_test_classwise_accuracy
                    
                    classwise_accuracy_savepath = ''.join([all_accuracy_savepath,#obj_model.result_savepath,
                                                datetime_now, '_',
                                                obj_model.model_name, '_',
                                                obj_model.data_name, '_',
                                                'classwise_accuracy_all_multiclass',
                                                '_nmulcls_', str(num_multiclass),
                                                '_avgep_', str(averaging_epochs),
                                                '_scllr_', str(unlearn_scale_lr).replace('.', 'o'),
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
                    unlearn_retain_accuracy_savepath = ''.join([all_accuracy_savepath,#obj_model.result_savepath,
                                                datetime_now, '_',
                                                obj_model.model_name, '_',
                                                obj_model.data_name, '_',
                                                'unlearn_retain_accuracy_all_multiclass',
                                                '_nmulcls_', str(num_multiclass),
                                                '_avgep_', str(averaging_epochs),
                                                '_scllr_', str(unlearn_scale_lr).replace('.', 'o'),
                                                '_nunlep_', str(num_unlearn_epochs),
                                                '_lv_', str(local_variance),
                                                '_smul_', str(size_multiplier),
                                                '_p_', str(p).replace('.', 'o'),
                                                '_tran_', str(tail_randomized),
                                                '_ngnrtd_', str(no_generated_data),
                                                '_prcnt_', str(retain_data_percent),
                                                '.csv'])
                    Unlearn_Class[epoch] = unlearn_classes_name
                    print(f'Saving Accuracies in path = {unlearn_retain_accuracy_savepath}')
                    dict_acc = OrderedDict({'Unlearn_Class':Unlearn_Class[:epoch+1],
                                            'retain_train_acc_unlearn':retain_train_acc_unlearning[:epoch+1],
                                            'unlearn_train_acc_unlearn':unlearn_train_acc_unlearning[:epoch+1],
                                            'retain_test_acc_unlearn':retain_test_acc_unlearning[:epoch+1],
                                            'unlearn_test_acc_unlearn':unlearn_test_acc_unlearning[:epoch+1],
                                            
                                            'retain_train_acc_retrain':retain_train_acc_retrained[:epoch+1],
                                            'unlearn_train_acc_retrain':unlearn_train_acc_retrained[:epoch+1],
                                            'retain_test_acc_retrain':retain_test_acc_retrained[:epoch+1],
                                            'unlearn_test_acc_retrain':unlearn_test_acc_retrained[:epoch+1]
                                            })
                    df_acc = pd.DataFrame.from_dict(dict_acc)
                    df_acc.to_csv(path_or_buf = unlearn_retain_accuracy_savepath, index=False)

                # Saving Retain & Unlearn Accuracies over all dataset on the Original Model
                avg_train_retain_acc = round_up(sum(all_train_retain_acc)/averaging_epochs, 2)
                avg_train_unlearn_acc = round_up(sum(all_train_unlearn_acc)/averaging_epochs, 2)
                avg_test_retain_acc = round_up(sum(all_test_retain_acc)/averaging_epochs, 2)
                avg_test_unlearn_acc = round_up(sum(all_test_unlearn_acc)/averaging_epochs, 2)

                accuracy_savepath = ''.join([all_accuracy_savepath,#obj_model.result_savepath,
                                                obj_model.model_name, '_',
                                                obj_model.data_name, '_',
                                                'avg_train_test_accuracies_multiclass_unlearn'
                                                '_nmulcls_', str(num_multiclass),
                                                '_avgep_', str(averaging_epochs),
                                                '.csv'
                                                ])
                dict_acc = OrderedDict({'_':['retain', 'unlearn', 'All'],
                                        'Train':[avg_train_retain_acc, avg_train_unlearn_acc, train_acc],
                                        'Test':[avg_test_retain_acc, avg_test_unlearn_acc, test_acc]})
                df_acc = pd.DataFrame.from_dict(dict_acc)
                df_acc.to_csv(path_or_buf = accuracy_savepath, index=False)

                # Average Retain & Unlearn Accuracies over all dataset after unlearning the Original Model   
                avg_retain_train_acc_unlearning = round_up(sum(retain_train_acc_unlearning)/averaging_epochs, 2)
                avg_unlearn_train_acc_unlearning = round_up(sum(unlearn_train_acc_unlearning)/averaging_epochs, 2)
                avg_retain_test_acc_unlearning = round_up(sum(retain_test_acc_unlearning)/averaging_epochs, 2)
                avg_unlearn_test_acc_unlearning = round_up(sum(unlearn_test_acc_unlearning)/averaging_epochs, 2)

                # Average Retain & Unlearn Accuracies over all dataset after retraining the Model   
                avg_retain_train_acc_retrained = round_up(sum(retain_train_acc_retrained)/averaging_epochs, 2)
                avg_unlearn_train_acc_retrained = round_up(sum(unlearn_train_acc_retrained)/averaging_epochs, 2)
                avg_retain_test_acc_retrained = round_up(sum(retain_test_acc_retrained)/averaging_epochs, 2)
                avg_unlearn_test_acc_retrained = round_up(sum(unlearn_test_acc_retrained)/averaging_epochs, 2)

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
                print(f'Average Unlearn loss on Train Data after Unlearning = {avg_unlearn_train_acc_unlearning}\n')
                print(f'Average Retain loss on Test Data after Unlearning = {avg_retain_test_acc_unlearning}')
                print(f'Average Unlearn loss on Test Data after Unlearning = {avg_unlearn_test_acc_unlearning}\n')

                print(f'Average Retain loss on Train Data after Retraining = {avg_retain_train_acc_retrained}')
                print(f'Average Unlearn loss on Train Data after Retraining = {avg_unlearn_train_acc_retrained}\n')
                print(f'Average Retain loss on Test Data after Retraining = {avg_retain_test_acc_retrained}')
                print(f'Average Unlearn loss on Test Data after Retraining = {avg_unlearn_test_acc_retrained}')
                print(unlearn_time_list)
                print(f'Time needed for total unlearning: Mean = {np.array(unlearn_time_list).mean()}, Std = {np.array(unlearn_time_list).std()}')
                
                print('-'*120)
                print('='*120)
                print('\n\n')
                    
        # except Exception as e:
        #     with open(f'{all_accuracy_savepath}/{datetime_now}_runtime_error.txt', 'w') as f:
        #         f.write(str(e))
        #     print(f'Error in Unlearning the Model: {str(e)}')
        #     print('-'*80)
        #     print('='*80)
        #     print('\n\n')
        #     continue
print('-'*140)
print('='*140)
print('#'*140)
