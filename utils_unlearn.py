#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn import metrics
from collections import OrderedDict
import seaborn as sns
import math
import time
from sklearn.metrics import accuracy_score, recall_score

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models import mobilenet_v2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models import *

__all__ = [
    'create_folder',
    'utils_',
    'accuracy',
    'accuracy_multiclass_unlearn',
    'round_up',
    'parameter_value_list_from_name',
    'map_image_0t01',
    'plot_sample_with_label',
    'plot_in_0to1_range'
]

def create_folder(folder_name):
    if len(folder_name):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        
    return folder_name

class utils_:
    
    def __init__(self,
                 image_size: tuple = (32, 32),
                 num_input_channels: int = 3,
                 num_classes: int = 10,
                 learning_rate: float = 1e-3,
                 batch_size: int = 128,
                 num_epochs: int = 10,
                 padding: int|tuple = 0,
                 model_save_name: str = '',
                 data_name: str = 'mnist',
                 model_name: str = 'LeNet32',#'ResNet9',
                 unlearn_cls: str|int = 0,
                 solver_type: str = 'adam',
                 result_savepath: str = './results/',
                 retrained_models_folder_name:str = 'retrained_models',
                 unlearned_models_folder_name:str = 'unlearned_models',
                 unlearn_type: str = '',
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        self.image_size = image_size
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.padding = padding
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.unlearn_cls = unlearn_cls
        self.device = device
        self.num_epochs = num_epochs
        self.result_savepath = result_savepath#''.join(['./results/', data_name, '/', model_name, '/'])
        self.result_savepath_retrained = ''.join([self.result_savepath, unlearn_type, '/', retrained_models_folder_name, '/'])
        self.result_savepath_unlearned = ''.join([self.result_savepath, unlearn_type, '/', unlearned_models_folder_name, '/'])
        self.model_save_name = model_save_name
        self.best_model_save_path = ''.join([self.result_savepath,
                                             model_name, '_',
                                             data_name,
                                             self.model_save_name,
                                             '_best_network.pth']
                                            )
        self.best_retrain_model_save_path = ''.join([self.result_savepath_retrained,
                                             model_name, '_',
                                             data_name,
                                             self.model_save_name,
                                             '_best_retrained_network_unlrncls_',
                                             str(self.unlearn_cls), '.pth']
                                            )
        
        self.best_unlearn_model_save_path = ''.join([self.result_savepath_unlearned,
                                             model_name, '_',
                                             data_name,
                                             self.model_save_name,
                                             '_best_unlearned_network_unlrncls_',
                                             str(self.unlearn_cls), '.pth']
                                            )

        self.result_savepath = self.create_folder(self.result_savepath)
        self.result_savepath_retrained = self.create_folder(self.result_savepath_retrained)
        self.result_savepath_unlearned = self.create_folder(self.result_savepath_unlearned)
        self.test_loss_best_network = 1e10
        self.test_unlearn_loss_best_network = 0
        self.data_name = data_name
        self.model_name = model_name
        
        # if data_name == 'mnist' or data_name == 'fashionMNIST':
        #     self.num_input_channels = 1
        #     self.num_classes = 10
        #     self.padding = 2
        #     # H, W = image_size[0], image_size[1]
        # elif data_name == 'svhn' or data_name == 'cifar10' or data_name == 'cifar100':
        #     self.num_input_channels = 3
        #     self.num_classes = 10
        #     if data_name == 'cifar100':
        #         self.num_classes = 20
        #     self.padding = 0
        #     # H, W = image_size[0], image_size[1]
        # else:
        #     print('Data is not mnist, svhn or cifar10.\n')
        
        #=======================================================================
        if model_name == 'ResNet9':
            self.network = ResNet9_(n_classes=self.num_classes,
                                    num_input_channels=self.num_input_channels,
                                    padding=self.padding,
                                    dataset_name=self.data_name)
            self.network = self.network.to(self.device)
        #=======================================================================
        elif model_name == 'AllCNN':
            self.network = AllCNN_(n_channels = self.num_input_channels,
                                   n_classes = self.num_classes,
                                   padding = self.padding,
                                   dataset_name=self.data_name
                                   )
            self.network = self.network.to(self.device)
        #=======================================================================
        elif model_name == 'ResNet18':
            # self.network = ResNet18(n_classes=self.num_classes)
            self.network = resnet18(weights='IMAGENET1K_V1')
            conv_weight = self.network.conv1.weight
            fc_weight = self.network.fc.weight
            fc_bias = self.network.fc.bias
            if self.num_input_channels == 1:
                conv_weight = torch.nn.Parameter((conv_weight.sum(dim = 1).unsqueeze(dim = 1))/3)
                
            self.network.conv1 = nn.Conv2d(in_channels = self.num_input_channels,
                                           out_channels = self.network.conv1.out_channels,
                                           kernel_size=self.network.conv1.kernel_size,
                                           stride=self.network.conv1.stride,
                                           padding=self.network.conv1.padding,
                                           bias=False
                                           )
            self.network.conv1.weight = conv_weight
            self.network.fc = nn.Linear(in_features=self.network.fc.in_features,
                                        out_features=self.num_classes,
                                        bias=True)
            self.network.fc.weight = torch.nn.Parameter(fc_weight[:self.num_classes])
            self.network.fc.bias = torch.nn.Parameter(fc_bias[:self.num_classes])
        #========================================================================
            
        elif model_name == 'MobileNet_v2':
            self.network = mobilenet_v2(weights='IMAGENET1K_V1')
            conv_weight = self.network.features[0][0].weight
            fc_weight = self.network.classifier[-1].weight
            fc_bias = self.network.classifier[-1].bias
            if self.num_input_channels == 1:
                conv_weight = torch.nn.Parameter((conv_weight.sum(dim = 1).unsqueeze(dim = 1))/3)
            self.network.features[0][0] = nn.Conv2d(in_channels = self.num_input_channels,
                                                    out_channels = self.network.features[0][0].out_channels,
                                                    kernel_size=self.network.features[0][0].kernel_size,
                                                    stride=self.network.features[0][0].stride,
                                                    padding=self.network.features[0][0].padding,
                                                    bias=False)
            self.network.features[0][0].weight = conv_weight
            self.network.classifier[-1] = nn.Linear(in_features=self.network.classifier[-1].in_features,
                                                    out_features=self.num_classes,
                                                    bias=True)
            self.network.classifier[-1].weight = torch.nn.Parameter(fc_weight[:self.num_classes])
            self.network.classifier[-1].bias = torch.nn.Parameter(fc_bias[:self.num_classes])
        #========================================================================
        else:
            print('Model Name not found.')
        #========================================================================
            
        print('\nModel Architecture:\n', )
        # print('Input Size = ', (self.batch_size, self.num_input_channels, image_size[0], image_size[1]))
        summary(self.network, (self.batch_size, self.num_input_channels, image_size[0], image_size[1]), device=str("cpu"))
        self.optimization_solver()
    #-------------------------------------------------------------------------------------------------------------------------
        
        
    def data_subset(self, dataloader, subset_size = 0.2):
        
        original_dataset = dataloader.dataset
        subset_size = int(subset_size*len(original_dataset))
        subset_indices = list(np.random.permutation(np.arange(len(original_dataset)))[:subset_size])
        subset_dataset = Subset(original_dataset, subset_indices)
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=self.batch_size, shuffle=True)
        
        return subset_loader
    
    #-------------------------------------------------------------------------------------------------------------------------
    
    def create_folder(self, folder_name):
        if len(folder_name):
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)
            
        return folder_name
    
    #-------------------------------------------------------------------------------------------------------------------------
        
    # define optimizer/solver
    def optimization_solver(self):
        
        if self.solver_type=='adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='SGD':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='Rprop':
            self.optimizer = optim.Rprop(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='RMSprop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='RAdam':
            self.optimizer = optim.RAdam(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Define your optimization solver: "{}"\n' .format(self.solver_type))
    #-------------------------------------------------------------------------------------------------------------------------
    
    def loss_function(self, y_bar, y):
        
        loss = F.cross_entropy(y_bar, y, reduction = 'sum')
        
        return loss

    #-------------------------------------------------------------------------------------------------------------------------
    
    def train_1epoch(self, model, optimizer, data_loader):
        
        model.train()# switch to train model
        model = model.to(self.device)
        epoch_train_loss = 0
        with torch.autograd.set_detect_anomaly(True):
            for batch, label in data_loader:
                batch, label = batch.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                y_pred = model(batch.type(torch.float))
                loss = self.loss_function(y_pred, label)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(data_loader.dataset)
        
        return epoch_train_loss
    
    #-------------------------------------------------------------------------------------------------------------------------
    
    def test(self, model, dataloader, aprox_ndigits = 2):
        
        model.eval()
        epoch_test_loss = 0# loss accumulation
        actual_labels = []#*len(dataloader.dataset)
        predicted_labels = []#*len(dataloader.dataset)
        model.to(self.device)
        
        with torch.no_grad():# no need to calculate gradient as its for test
            for batch, labels in dataloader:
                data, labels = batch.to(self.device), labels.to(self.device)
                predicted = model(data.type(torch.float))
                loss = self.loss_function(predicted, labels)
                epoch_test_loss += loss.item()# record minibatch loss
                predicted = predicted.cpu().detach().numpy()
                predicted = np.argmax(predicted, axis=1)
                predicted_labels.extend(predicted)
                actual_labels.extend(labels.cpu().detach().numpy())
            epoch_test_loss /= len(dataloader.dataset)# record minibatch loss
            
            confusion_matrix = metrics.confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)#Computing the confusion matrix
            acc = round(accuracy_score(y_true=actual_labels, y_pred=predicted_labels) * 100, aprox_ndigits)

            # Class-wise Accuracy (Recall per class in percentage)
            classwise_accuracy = recall_score(y_true=actual_labels, y_pred=predicted_labels, average=None, zero_division=0)
            classwise_accuracy = [round(val * 100, aprox_ndigits) for val in classwise_accuracy]
            
        return epoch_test_loss, confusion_matrix, acc, classwise_accuracy, actual_labels, predicted_labels

    #-------------------------------------------------------------------------------------------------------------------------
    
    def train(self):
        
        # to record avg. losses  
        if self.count_epoch == 0:
            self.train_loss = np.zeros(self.num_epochs)
            self.test_loss = np.zeros(self.num_epochs)
            self.train_acc = np.zeros(self.num_epochs)
            self.test_acc = np.zeros(self.num_epochs)
        else:
            self.train_loss = np.concatenate((self.train_loss, np.zeros(self.num_epochs-self.count_epoch)))
            self.test_loss = np.concatenate((self.test_loss, np.zeros(self.num_epochs-self.count_epoch)))
            self.train_acc = np.concatenate((self.train_acc, np.zeros(self.num_epochs-self.count_epoch)))
            self.test_acc = np.concatenate((self.test_acc, np.zeros(self.num_epochs-self.count_epoch)))
            
        self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, self.batch_size, shuffle=True)
        
        for epoch in tqdm(np.arange(self.count_epoch, self.num_epochs)):
            
            print('\nTrain epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            # epoch_train_loss = self.train_1epoch()
            epoch_train_loss = self.train_1epoch(self.network, self.optimizer, self.train_loader)
            print('Average train loss = {}\n'.format(epoch_train_loss))
            
            # test on train data
            print('Test on train data epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            self.test_train_loader = self.data_subset(self.train_loader, 0.1)
            epoch_test_train_loss, train_confusion_matrix, epoch_test_train_acc, self.train_classwise_accuracy, _, _ = self.test(self.network, self.test_train_loader)
            self.train_loss[epoch] = epoch_test_train_loss
            self.train_acc[epoch] = epoch_test_train_acc
            print('Avg. test loss on train data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_train_loss), float(epoch_test_train_acc)))
            
            # test on test data
            print('Test on test data epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            self.test_test_loader = self.data_subset(self.test_loader, 0.3)
            epoch_test_test_loss, test_confusion_matrix, epoch_test_test_acc, self.test_classwise_accuracy, _, _ = self.test(self.network, self.test_test_loader)
            self.test_loss[epoch] = epoch_test_test_loss
            self.test_acc[epoch] = epoch_test_test_acc
            print('Avg. test loss on test data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_test_loss), float(epoch_test_test_acc)))
            
            # save the best trained network
            if((self.count_epoch==0) or (epoch_test_test_acc > self.test_acc_best_network)):             
                self.epoch_best_network = self.count_epoch
                self.train_loss_best_network = epoch_test_train_loss
                self.test_acc_best_network = epoch_test_test_acc
                self.train_confusion_matrix = train_confusion_matrix
                self.test_confusion_matrix = test_confusion_matrix
                self.save_model(epoch = self.epoch_best_network, model_save_path=self.best_model_save_path)
                print(f'Saving model in path: {self.best_model_save_path}\n')
                
            self.save_loss_csv(self.count_epoch,
                               save_csv_filename = ''.join([self.result_savepath,
                                                            self.model_name, '_',
                                                            self.data_name, '_',
                                                            'Training_Loss']
                                                           ))
            self.plot_loss_save_images(epoch=self.count_epoch,
                                       loss_type='loss',
                                       save_image_filename=''.join([self.result_savepath,
                                                                    self.model_name, '_',
                                                                    self.data_name, '_',
                                                                    'training_time_loss_plot']))
            self.plot_loss_save_images(epoch=self.count_epoch,
                                       loss_type='acc',
                                       save_image_filename=''.join([self.result_savepath,
                                                                    self.model_name, '_',
                                                                    self.data_name, '_',
                                                                    'training_time_acc_plot']))
            self.count_epoch = epoch + 1
            
        self.save_confusion_matrix(confusion_matrix=train_confusion_matrix,
                                   name = ''.join([self.result_savepath,
                                                   self.model_name, '_',
                                                   self.data_name, 
                                                   self.model_save_name, '_',
                                                   'train_confusion_matrix'],
                                                  ),
                                    title_info = f'(Epoch - {self.count_epoch})'            
                                   )
        self.save_confusion_matrix(confusion_matrix=test_confusion_matrix,
                                   name = ''.join([self.result_savepath,
                                                   self.model_name, '_',
                                                   self.data_name,
                                                   self.model_save_name, '_',
                                                   'test_confusion_matrix'],
                                                  ),
                                    title_info = f'(Epoch - {self.count_epoch})'
                                   )
    #-------------------------------------------------------------------------------------------------------------------------
    
    
    def unlearn(self, model, optimizer, dataloader, num_epochs=10):
        
        # to record avg. losses  
        self.unlearn_retain_loss = np.zeros(num_epochs)
        self.unlearn_retain_acc = np.zeros(num_epochs)
        
        self.unlearn_unlearn_loss = np.zeros(num_epochs)
        self.unlearn_unlearn_acc = np.zeros(num_epochs)
        
        self.unlearn_test_loss = np.zeros(num_epochs)
        self.unlearn_test_acc = np.zeros(num_epochs)
        # plot_in_0to1_range(test_batch[0].permute(1, 2, 0), self.train_data.classes[test_labels[0]], save_filepath='./to_see_img')
                
        model = model.to(self.device)
        # unlearn_optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.test_loader = DataLoader(self.test_data, self.batch_size, shuffle=True)
        print(f'Number of total unlearned data = {len(dataloader.dataset)}\n')
        
        total_time = 0
        epoch_unlearning_losses = [None]*num_epochs 
        for epoch in tqdm(range(num_epochs)):
            
            print('\nUnlearning epoch {}/({})\n' .format(epoch+1, num_epochs))
            epoch_unlearning_loss = 0
            
            init_time = time.time()
            for batch, label in dataloader:
                batch, label = batch.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                y_pred = model(batch.type(torch.float))
                loss = F.cross_entropy(y_pred, label)
                loss.backward()
                optimizer.step()
                epoch_unlearning_loss += loss.item()
            epoch_unlearning_loss /= len(dataloader.dataset)
            epoch_unlearning_losses[epoch] = epoch_unlearning_loss
                
            end_time = time.time()
            req_time = end_time-init_time
            total_time = total_time + req_time      
            model.eval()
            
            # epoch_unlearning_loss = self.train_1epoch(model, unlearn_optimizer, self.unlearn_loader_all)
            print('Average unlearning loss = {}\n'.format(epoch_unlearning_loss))
                        
            # test on retain data
            print('Test on retain data epoch {}/({})\n' .format(epoch+1, num_epochs))
            self.test_retain_loader = self.data_subset(self.retain_loader_test, 0.1)
            epoch_test_retain_loss, retain_confusion_matrix, epoch_test_retain_acc, self.retain_classwise_accuracy, actual_labels, _ = self.test(model, self.test_retain_loader)
            self.unlearn_retain_loss[epoch] = epoch_test_retain_loss
            self.unlearn_retain_acc[epoch] = epoch_test_retain_acc
            print('Avg. test loss on retain data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_retain_loss), float(epoch_test_retain_acc)))
            
            # test on unlearn data
            print('Test on unlearn data epoch {}/({})\n' .format(epoch+1, num_epochs))
            self.test_unlearn_loader = self.data_subset(self.unlearn_dataloader, 1)
            epoch_test_unlearn_loss, _, epoch_test_unlearn_acc, self.unlearn_classwise_accuracy, _, _ = self.test(model, self.test_unlearn_loader)
            self.unlearn_unlearn_loss[epoch] = epoch_test_unlearn_loss
            self.unlearn_unlearn_acc[epoch] = epoch_test_unlearn_acc
            print('Avg. test loss on unlearn data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_unlearn_loss), float(epoch_test_unlearn_acc)))
            
            # test on test data
            print('Test on test data epoch {}/({})\n' .format(epoch+1, num_epochs))
            self.test_test_loader = self.data_subset(self.test_loader, 0.1)
            epoch_test_test_loss, test_unlearn_confusion_matrix, epoch_test_test_acc, self.test_unlearn_classwise_accuracy, _, _ = self.test(model, self.test_test_loader)
            self.unlearn_test_loss[epoch] = epoch_test_test_loss
            self.unlearn_test_acc[epoch] = epoch_test_test_acc
            print('Avg. test loss on test data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_test_loss), float(epoch_test_test_acc)))
            
            # save the best trained network
            # if((epoch==0) or (self.test_unlearn_loss_best_network<epoch_test_test_loss)):             
                
            self.unlearn_model = model
            self.unlearn_optimizer = optimizer
            self.epoch_best_network = epoch
            self.retain_loss_best_network = epoch_test_retain_loss
            self.test_unlearn_loss_best_network = epoch_test_test_loss
            
            self.save_unlearn_model(epoch = self.epoch_best_network, model_save_path=self.best_unlearn_model_save_path)
            self.save_unlearn_loss_csv(epoch, save_csv_filename = ''.join([self.result_savepath_unlearned,
                                                                           self.model_name, '_',
                                                                           self.data_name, '_'
                                                                           'Unlearning_Loss']))
            # self.count_epoch = epoch + 1
        
        # self.save_confusion_matrix(confusion_matrix=retain_confusion_matrix,
        #                            name = ''.join([self.result_savepath_unlearned,
        #                                            self.model_name, '_',
        #                                            self.data_name,
        #                                            '_unlrncls_', str(self.unlearn_cls), '_',
        #                                             'retain_confusion_matrix']
        #                                           ),
        #                             title_info = f'(Unlearn Class - {self.unlearn_cls})',
        #                            )
        # self.save_confusion_matrix(confusion_matrix=unlearn_confusion_matrix,
        #                            name = ''.join([self.result_savepath_unlearned,
        #                                            self.model_name, '_', 
        #                                            self.data_name, 
        #                                            '_unlrncls_', str(self.unlearn_cls), '_',
        #                                            'unlearn_confusion_matrix']
        # #                                           ))
        # self.save_confusion_matrix(confusion_matrix=test_unlearn_confusion_matrix,
        #                            name = ''.join([self.result_savepath_unlearned,
        #                                            self.model_name, '_',
        #                                            self.data_name,
        #                                            '_unlrncls_', str(self.unlearn_cls), '_',
        #                                            'test_unlearn_confusion_matrix']
        #                                           ),
        #                             title_info = f'(Unlearn Class - {self.unlearn_cls})'
        #                            )
        
            
        return model, epoch_unlearning_losses, total_time
    #-------------------------------------------------------------------------------------------------------------------------
    
        
    def save_unlearn_model(self, epoch, model_save_path = './'):
        
        torch.save({
            'epoch':epoch+1,
            'unlearn_retain_loss':self.unlearn_retain_loss[:epoch+1],
            'epoch_unlearn_retain_loss':self.unlearn_retain_loss[epoch],
            'unlearn_retain_acc':self.unlearn_retain_acc[:epoch+1],
            'epoch_unlearn_retain_acc':self.unlearn_retain_acc[epoch],
            'unlearn_test_loss':self.unlearn_test_loss[:epoch+1],
            'epoch_unlearn_test_loss':self.unlearn_test_loss[epoch],
            'unlearn_test_acc':self.unlearn_test_acc[:epoch+1],
            'epoch_unlearn_test_acc':self.unlearn_test_acc[epoch],
            'unlearn_unlearn_loss':self.unlearn_unlearn_loss[:epoch+1],
            'epoch_unlearn_unlearn_loss':self.unlearn_unlearn_loss[epoch],
            'unlearn_unlearn_acc':self.unlearn_unlearn_acc[:epoch+1],
            'epoch_unlearn_unlearn_acc':self.unlearn_unlearn_acc[epoch],
            'retain_classwise_accuracy':self.retain_classwise_accuracy,
            'unlearn_classwise_accuracy':self.unlearn_classwise_accuracy,
            'test_unlearn_classwise_accuracy':self.test_unlearn_classwise_accuracy,
            'state_dict':self.unlearn_model.state_dict(),
            'optimizer':self.unlearn_optimizer.state_dict()
        }, model_save_path)
        
        
    def save_model(self, epoch, model_save_path = './'):
        
        torch.save({
            'epoch':epoch+1,
            'train_acc':self.train_acc[:epoch+1],
            'test_acc':self.test_acc[:epoch+1],
            'epoch_train_acc':self.train_acc[-1],
            'epoch_test_acc':self.test_acc[-1],
            'train_classwise_accuracy':self.train_classwise_accuracy,
            'test_classwise_accuracy':self.test_classwise_accuracy,
            'train_loss':self.train_loss[:epoch+1],
            'test_loss':self.test_loss[:epoch+1],
            'epoch_train_loss':self.train_loss[epoch],
            'epoch_test_loss':self.test_loss[epoch],
            'state_dict':self.network.state_dict(),
            'optimizer':self.optimizer.state_dict()
        }, model_save_path)
        
    #-------------------------------------------------------------------------------------------------------------------------

        
    def load_unlearn_network(self, network_path=''):
        
        if len(network_path)==0:
            network_path = self.best_unlearn_model_save_path
        
        if os.path.isfile(network_path):
            print('Loading unlearned network checkpoint from: "{}"\n'.format(network_path))
            checkpoint = torch.load(network_path, map_location=self.device)
            
            self.count_epoch = checkpoint['epoch']
            
            self.unlearn_retain_loss = checkpoint['unlearn_retain_loss']
            self.epoch_unlearn_retain_loss = checkpoint['epoch_unlearn_retain_loss']
            self.unlearn_retain_acc = checkpoint['unlearn_retain_acc']
            self.epoch_unlearn_retain_acc = checkpoint['epoch_unlearn_retain_acc']
            
            self.unlearn_test_loss = checkpoint['unlearn_test_loss']
            self.epoch_unlearn_test_loss = checkpoint['epoch_unlearn_test_loss']
            self.unlearn_test_acc = checkpoint['unlearn_test_acc']
            self.epoch_unlearn_test_acc = checkpoint['epoch_unlearn_test_acc']
            
            self.unlearn_unlearn_loss = checkpoint['unlearn_unlearn_loss']
            self.epoch_unlearn_unlearn_loss = checkpoint['epoch_unlearn_unlearn_loss']
            self.unlearn_unlearn_acc = checkpoint['unlearn_unlearn_acc']
            self.epoch_unlearn_unlearn_acc = checkpoint['epoch_unlearn_unlearn_acc']
            
            self.retain_classwise_accuracy = checkpoint['retain_classwise_accuracy']
            self.unlearn_classwise_accuracy = checkpoint['unlearn_classwise_accuracy']
            self.test_unlearn_classwise_accuracy = checkpoint['test_unlearn_classwise_accuracy']
            self.optimizer_state_dict = checkpoint['optimizer']
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print('Loaded unlearn network checkpoint from "{}"\nepoch: {}\n' 
                  .format(network_path, self.count_epoch))
            
        else:
            print('No pre-trained network checkpoint found at "{}"\n'.format(network_path))
        print('-'*80)
        print('='*80)
        
        
    def load_network(self, network_path=''):
        
        if len(network_path)==0:
            network_path = self.best_model_save_path
        
        if os.path.isfile(network_path):
            print('Loading pre-trained network checkpoint from: "{}"\n'.format(network_path))
            checkpoint = torch.load(network_path, map_location=self.device)
            
            self.count_epoch = checkpoint['epoch']
            self.train_acc = checkpoint['train_acc']
            self.test_acc = checkpoint['test_acc']
            self.train_classwise_accuracy = checkpoint['train_classwise_accuracy']
            self.test_classwise_accuracy = checkpoint['test_classwise_accuracy']
            self.train_loss = checkpoint['train_loss']
            self.test_loss = checkpoint['test_loss']
            self.epoch_train_loss = checkpoint['epoch_train_loss']
            self.epoch_test_loss = checkpoint['epoch_test_loss']
            self.epoch_train_acc = checkpoint['epoch_train_acc']
            self.epoch_test_acc = checkpoint['epoch_test_acc']
            self.optimizer_state_dict = checkpoint['optimizer']
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} test loss: {}\n' 
                  .format(network_path, self.count_epoch, self.epoch_train_loss, self.epoch_test_loss))
            
        else:
            print('No pre-trained network checkpoint found at "{}"\n'.format(network_path))
        print('-'*80)
        print('='*80)
    #-------------------------------------------------------------------------------------------------------------------------
    # plot and save (image) different losses

    def plot_loss_save_images(
        self, 
        epoch: int, 
        loss_type: str = 'loss',
        save_image_filename: str = '', 
        title: str = 'CE loss', 
        marker: str = 'None', 
        markersize: int = 10, 
        linewidth: int = 2, 
        linestyle: str = '-', 
        title_fontsize: int = 25, 
        xyticks_fontsize: int = 20 
    ) -> None:
        """
        Plot the different train, val and test loss against epochs. 
        """
        
        X = range(epoch+1)
        
        plt.plot(
            X, self.train_loss[:epoch+1] if loss_type=='loss' else self.train_acc[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='loss: train' 
        )
        plt.plot(
            X, self.test_loss[:epoch+1] if loss_type=='loss' else self.test_acc[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='loss: test' 
        )
        plt.xticks(fontsize=xyticks_fontsize)
        plt.yticks(fontsize=xyticks_fontsize)
        plt.xlabel('epoch', fontsize=title_fontsize)
        plt.ylabel('avg. loss' if loss_type=='loss' else 'avg. acc.', fontsize=title_fontsize)
        plt.grid(linestyle='--')
        plt.legend(loc='upper right')
        plt.title(title, fontsize=title_fontsize)   

            # save the plot    
        if len(save_image_filename):
            plt.savefig(
                '' .join([save_image_filename,'.png']), 
                bbox_inches='tight' 
            )#save the plot in png form
            plt.close()
        else:
            plt.show(block=True)
            # plt.show(block=False)
            # plt.pause(0.001)  # Pause to allow the plot to update
            # plt.close()

    #------------------------------------------------------
    
    
    def plot_loss(self, loss, no_of_epochs):
        epoch = np.arange(no_of_epochs)
        plt.plot(epoch, loss)
        plt.show(block=True)
        
    #save (csv) different losses
    def save_loss_csv(self,
                      epoch,
                      save_csv_filename='temp'
                      ):
        
        dict_loss = OrderedDict({'epoch':list(range(1, epoch+2)),
                                 'train_loss':self.train_loss[:epoch+1],
                                 'train_acc':self.train_acc[:epoch+1],
                                 'test_loss':self.test_loss[:epoch+1],
                                 'test_acc':self.test_acc[:epoch+1],
                                })
        df_loss = pd.DataFrame.from_dict(dict_loss)
        df_loss.to_csv(''.join([save_csv_filename, '.csv']), index=False)
        
    
    #save (csv) different losses
    def save_unlearn_loss_csv(self,
                              epoch,
                              save_csv_filename='temp'
                              ):
        
        dict_loss = OrderedDict({'epoch':list(range(1, epoch+2)),
                                 'unlearn_retain_loss':self.unlearn_retain_loss[:epoch+1],
                                 'unlearn_retain_acc':self.unlearn_retain_acc[:epoch+1],
                                 'unlearn_unlearn_loss':self.unlearn_unlearn_loss[:epoch+1],
                                 'unlearn_unlearn_acc':self.unlearn_unlearn_acc[:epoch+1],
                                 'unlearn_test_loss':self.unlearn_test_loss[:epoch+1],
                                 'unlearn_test_acc':self.unlearn_test_acc[:epoch+1],
                                })
        df_loss = pd.DataFrame.from_dict(dict_loss)
        df_loss.to_csv(''.join([save_csv_filename, '.csv']), index=False)
        
        
    def save_confusion_matrix(self, confusion_matrix, name, title_info = '', xticklabels = "auto", yticklabels = "auto"):
        
        # Plotting the confusion matrix
        plt.figure(figsize=(12, 8))
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_cm = np.divide(
            confusion_matrix.astype('float'),
            row_sums,
            out=np.zeros_like(confusion_matrix, dtype=float),
            where=row_sums != 0
        )

        # Replace NaNs (if any) with zeros (safer display)
        confusion_matrix = np.nan_to_num(normalized_cm)
        # confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        # confusion_matrix = np.nan_to_num(confusion_matrix)
        sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap="Blues", cbar=False, xticklabels=xticklabels, yticklabels=yticklabels, annot_kws={"weight": "bold", "size": 12})
        plt.title(f"Confusion Matrix {title_info}", fontweight='bold', fontsize=18)
        plt.xlabel("Predicted", fontweight='bold', fontsize=14)
        plt.ylabel("Actual", fontweight='bold', fontsize=14)
        
        # Bold tick labels
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontsize=11)

        plt.tight_layout()
        # Save the confusion matrix as a PNG file
        if name:
            plt.savefig(name + '.png', dpi=200)
            plt.close()
        else:
            plt.show()
        
        
    # def plot_random_sample(self, data, labels):
    #     idx = np.random.choice(len(data))
    #     plt.imshow(data[idx].permute(1, 2, 0), cmap='gray')
    #     plt.title('class-'+str(labels[idx].item()))
    #     plt.show()
#-------------------------------------------------------------------------------------
######################################################################################
def classwise_accuracy_func(y_pred, y_true, num_classes):
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        class_mask = (y_true == i)
        total_per_class[i] = class_mask.sum().item()
        correct_per_class[i] = (y_pred[class_mask] == i).sum().item()

    # Avoid division by zero
    classwise_acc = np.where(total_per_class > 0, (correct_per_class / total_per_class)*100, 0.0)
    classwise_acc = [round(x, 2) for x in classwise_acc]
    
    return classwise_acc          
        

def accuracy(model,
             dataloader,
             unlearn_cls,
             aprox_ndigits=2,
             num_classes=10,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device).float()
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Compute confusion matrix and basic accuracy
    cm = metrics.confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    acc = round(accuracy_score(all_labels, all_preds) * 100, aprox_ndigits)

    # Retention accuracy: correct predictions excluding unlearn class
    total_correct = np.trace(cm)
    total_unlearn_correct = cm[unlearn_cls, unlearn_cls]
    total_preds = np.sum(cm)
    total_unlearn_preds = np.sum(cm[unlearn_cls])

    retain_acc = round((total_correct - total_unlearn_correct) / (total_preds - total_unlearn_preds) * 100, aprox_ndigits)
    unlearn_acc = round(total_unlearn_correct / total_unlearn_preds * 100, aprox_ndigits) if total_unlearn_preds > 0 else 0.0

    # Class-wise accuracy using numpy
    classwise_accuracy = [
        round(cm[i, i] / cm[i].sum() * 100, aprox_ndigits) if cm[i].sum() > 0 else 0.0
        for i in range(num_classes)
    ]

    return cm, acc, classwise_accuracy, retain_acc, unlearn_acc#actual_labels, predicted_labels


def accuracy_multiclass_unlearn(model, dataloader,
                                unlearn_classes,
                                aprox_ndigits = 2,
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                ):#, num_classes = 10):

    actual_labels = []#*len(dataloader.dataset)
    predicted_labels = []#*len(dataloader.dataset)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch, labels in dataloader:
            data = batch.to(device)
            predicted = model(data.type(torch.float))
            predicted = predicted.cpu().detach().numpy()
            predicted = np.argmax(predicted, axis=1)
            predicted_labels.extend(predicted)
            actual_labels.extend(labels)
    confusion_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)#Computing the confusion matrix
    acc = round((sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))*100, aprox_ndigits)
    retain_acc = (sum(confusion_matrix.diagonal()) - sum(confusion_matrix[i, i] for i in unlearn_classes))/(np.sum(confusion_matrix) - sum(sum(confusion_matrix[i]) for i in unlearn_classes))
    retain_acc = round(retain_acc*100, aprox_ndigits)
    unlearn_acc = sum(confusion_matrix[i, i] for i in unlearn_classes)/sum(sum(confusion_matrix[i]) for i in unlearn_classes)
    unlearn_acc = round(unlearn_acc*100, aprox_ndigits)
    classwise_accuracy = classwise_accuracy_func(y_pred = predicted_labels, y_true=actual_labels, num_classes = confusion_matrix.shape[0])
    # classwise_accuracy = [round((confusion_matrix[i,i]/np.sum(confusion_matrix, axis=1)[i])*100, aprox_ndigits) for i in range(confusion_matrix.shape[0])]
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = list(range(num_classes)))
    # cm_display.plot()
    # plt.close()
    
    return confusion_matrix, acc, classwise_accuracy, retain_acc, unlearn_acc#actual_labels, predicted_labels

    
def map_image_0t01(data):
    
    """Data Shape: Height x Width x Channel"""
    
    modified_data = np.zeros_like(data)
    num_channels = data.shape[-1]
    if type(data) != np.ndarray:
        data = data.numpy()
    
    if  len(data.shape) != 3:
        raise ValueError('Input Data Dim should be: Height x Width x Channel')
    
    if num_channels not in  [1, 3]:
        raise ValueError(f'Input Data Dim should be: Height x Width x Channel. Got input of shape: {num_channels}')
        
    for channel in range(num_channels):
        max_, min_ = np.max(data[:, :, channel]), np.min(data[:, :, channel])
        modified_data[:, :, channel] = (data[:, :, channel] - min_)/(max_ - min_)
            
    
    return modified_data

def plot_sample_with_label(data,
                           label = 'Not Mentioned',
                           save_filepath = '',
                           show_img = True, 
                           figsize=(8, 6)
                           ):
   
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap='gray')
    plt.title(str(label))
    
    if save_filepath:
        if data.shape[2] != 3:
            data = data.repeat(3, 2)
        # plt.imsave(save_filepath+'.jpg', data)
        plt.savefig(save_filepath+'.png',
                    bbox_inches='tight',
                    pad_inches=0
                    )
    
    if show_img == True:    
        plt.show()
    else:
        plt.close()
    
def plot_in_0to1_range(data,
                       label = 'Not Mentioned',
                       save_filepath = '',
                       show_img = True
                       ):

    modified_data = map_image_0t01(data)
    plot_sample_with_label(modified_data,
                           label=label,
                           save_filepath=save_filepath,
                           show_img = show_img
                           )


def round_up(num = 1.987, digit = 2):
    
    if len(str(num).split('.')[-1]) >= digit:
        dec = 10**digit
        temp = math.floor(num * dec) / dec
    else:
        #print(f'Number has not {digit}-digit after point.')
        temp = num
    
    return temp


def parameter_value_list_from_name(parameter_name_list, results_path):
    
    for _, _, files in os.walk(results_path):
        all_list_files= []
        for file_ in files:
            all_list = file_.split('_')
            not_req_list = ['unlearn', 'retain', 'accuracy', 'all']
            req_list = [item for item in all_list if item not in not_req_list]
            all_list_files.append(req_list)
            
    all_list_dict = []
    float_list = ['scllr', 'p']
    all_list_files = sorted(all_list_files)
    for l in all_list_files:
        dict_ = OrderedDict({})
        for key in parameter_name_list:
            dict_['name'] = l[0]+'_'+l[1] if l[0] != 'MobileNet' else l[0]+'_'+l[1]+'_'+l[2]
            val = l[l.index(key)+1].replace('o', '.').replace('.csv', '')
            dict_[key] = float(val) if key in float_list else int(val)
        all_list_dict.append(dict_)
    out = pd.DataFrame(all_list_dict) 
    
    return out
    
    

    
