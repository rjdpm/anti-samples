#!/usr/bin/env python3

import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import time

__all__ = [
    'UNMUNGE_'
]

    
############################################## Version - 4 ################################################################################


def UNMUNGE_(unlearn_data,
             retain_data,
             retain_labels=0,
             local_variance = 10,
             size_multiplier = 1,
             p = 0.80,
             tail_randomized:int = None,
             no_generated_data = 50,
             eps = 1,
             convex_combination = True):
    
    print(f'Number of unlearn data = {len(unlearn_data)}')
    print(f'Number of retain data = {len(retain_data)}')
    flag1 = time.time()
    
    pairwise_distance = metrics.pairwise_distances(unlearn_data, retain_data)
    
    flag2 = time.time()
    
    unmunge_data = np.zeros((unlearn_data.shape[0]*size_multiplier, unlearn_data.shape[1]))
    farthest_point = [None]*unlearn_data.shape[0]*size_multiplier
    sorted_pairwise_indices = np.argsort(pairwise_distance)
    
    flag3 = time.time()
    for epoch in tqdm(range(size_multiplier)):
        if tail_randomized:
            tail_idx = np.random.choice(a = range(1, tail_randomized+1), size=unlearn_data.shape[0])
            farthest_neighbour_indices = sorted_pairwise_indices[range(sorted_pairwise_indices.shape[0]), -tail_idx]
        else:
            farthest_neighbour_indices = sorted_pairwise_indices[range(sorted_pairwise_indices.shape[0]), -1]
        farthest_neighbours = retain_data[farthest_neighbour_indices] #farthest datapoint from the chosen datapoint is taken from the retained set
        farthest_point[epoch*unlearn_data.shape[0]:(epoch + 1)*unlearn_data.shape[0]] = retain_labels[farthest_neighbour_indices]
        std = (np.abs(unlearn_data - farthest_neighbours) + eps)*local_variance
        sample = np.random.normal(loc=farthest_neighbours, scale=std)
        prob = np.random.uniform(low=0.0, high=1.0, size= unlearn_data.shape)
        anti_data = np.zeros_like(unlearn_data)
        criteria_matrix = (prob <= p)
        
        anti_data[criteria_matrix] = sample[criteria_matrix]
        anti_data[~criteria_matrix] = farthest_neighbours[~criteria_matrix]
        unmunge_data[epoch*unlearn_data.shape[0]:(epoch + 1)*unlearn_data.shape[0]] = anti_data
        
    if convex_combination == True:        
        temp_matrix = np.random.normal(loc=0, scale=1, size=(no_generated_data, unmunge_data.shape[0]))
        exp_matrix = np.exp(temp_matrix)
        softmax_values = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
        unmunge_data = np.dot(softmax_values, np.array(unmunge_data))
        
    else:
        unmunge_data = torch.from_numpy(unmunge_data)
    
    flag4 = time.time()
    unmunge_time = flag4 - flag1
    print(f'Number of Generated Data = {len(unmunge_data)}')
    print(f'Time needed to calculate Pairwise Distance = {flag2 - flag1} sec')
    print(f'Time needed for Sorting = {flag3 - flag2} sec')
    print(f'Time needed to generate the samples = {flag4 - flag3} sec')
    
    
    return torch.from_numpy(unmunge_data), farthest_point, pairwise_distance, unmunge_time
