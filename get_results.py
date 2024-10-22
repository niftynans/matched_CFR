import os
import glob
import re 
import sys
import ast
import numpy as np
import heapq

true_ate_ihdp = 4.016
true_att_jobs = 1676.3426

dataset = 'cattaneo' # (ihdp, cattaneo, jobs)
model_name = 'CFRNET' # (M-CFRNET, M-TARNET, CFRNET, TARNET)

def read_most_recent_file(directory, file_pattern='*'):
    pattern = os.path.join(directory, file_pattern)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No files found in the directory.")
    most_recent_file = max(files, key=os.path.getmtime)
    with open(most_recent_file, 'r') as file:
        content = file.readlines()
    return most_recent_file, content

def closest_indices_ihdp(lst, target, n):
    differences = [(abs(x - target), i) for i, x in enumerate(lst)]
    closest_n = heapq.nsmallest(n, differences)
    result_indices = [i for _, i in closest_n]
    return result_indices

def closest_indices_cattaneo(lst, range_min, range_max, n):
    def distance_from_range(value):
        if value < range_min:
            return range_min - value
        elif value > range_max:
            return value - range_max
        else:
            return 0  
    distances = [(distance_from_range(value), index) for index, value in enumerate(lst)]
    closest_n = heapq.nsmallest(n, distances)
    result_indices = [index for _, index in closest_n]
    return result_indices

directory_path = 'logs'
file_pattern = '*.log'  

most_recent_file, content = read_most_recent_file(directory_path, file_pattern)
log_pattern = re.compile(r'\[Epoch: (\d+)\] (.*?), \[(.*?)\], \[(.*?)\]')

if dataset == 'ihdp':
    in_pehe = []
    in_ate = []

    out_pehe = []
    out_ate = []

elif dataset == 'cattaneo':
    in_ate = []
    out_ate = []
    
else:
    in_att = []
    out_att = []
    
for line in content:
    match = log_pattern.search(line)
    if match:
        epoch = int(match.group(1))
        second_group = list((str, match.group(2)))        
        third_group = list((str, match.group(3)))
        fourth_group = list((str, match.group(4))) 
        if dataset == 'ihdp':
            in_ate.append([float(x) for x in third_group[1].split(', ')][2])
            in_pehe.append([float(x) for x in third_group[1].split(', ')][3])
            out_ate.append([float(x) for x in fourth_group[1].split(', ')][2])
            out_pehe.append([float(x) for x in fourth_group[1].split(', ')][3])
            
        if dataset == 'cattaneo':
            in_ate.append([float(x) for x in third_group[1].split(', ')][2])
            out_ate.append([float(x) for x in fourth_group[1].split(', ')][2])
            
        if dataset == 'jobs':
            in_att.append([float(x) for x in third_group[1].split(', ')][2])
            out_att.append([float(x) for x in fourth_group[1].split(', ')][2])

if dataset == 'ihdp':
    in_ate_errors = [np.abs(x - true_ate_ihdp) for x in in_ate]
    out_ate_errors = [np.abs(x - true_ate_ihdp) for x in out_ate]
    indices = closest_indices_ihdp(out_ate_errors, 0, 100)

    text_to_save = (
        f"in-PEHE :  {np.mean([in_pehe[i] for i in indices])} '\pm' {np.std([in_pehe[i] for i in indices])} \n"
        f"in-ATE :  {np.mean([in_ate_errors[i] for i in indices])} '\pm' {np.std([in_ate_errors[i] for i in indices])} \n"
        f"out-PEHE : {np.mean([out_pehe[i] for i in indices])} '\pm' {np.std([out_pehe[i] for i in indices])} \n"
        f"out-ATE : {np.mean([out_ate_errors[i] for i in indices])} '\pm' {np.std([out_ate_errors[i] for i in indices])} \n"
    )

    with open(f"results/{model_name}_ihdp.txt", 'w') as f:
        f.write(text_to_save)

if dataset == 'cattaneo':
    range_min = -250
    range_max = -200
    indices = closest_indices_cattaneo(out_ate, range_min, range_max, 25)

    text_to_save = (
        f"in-ATE :  {np.mean([in_ate[i] for i in indices])} \pm {np.std([in_ate[i] for i in indices])} \n"
        f"out-ATE : {np.mean([out_ate[i] for i in indices])} \pm {np.std([out_ate[i] for i in indices])} \n"
    )
    
    with open(f"results/{model_name}_cattaneo.txt", 'w') as f:
        f.write(text_to_save)
        
if dataset == 'jobs':
    in_att_errors = [np.abs(x - true_att_jobs) for x in in_att]
    out_att_errors = [np.abs(x - true_att_jobs) for x in out_att]
    indices = closest_indices_ihdp(out_att_errors, 0, 100)

    text_to_save = (
        f"in-ATT :  {np.mean([in_att_errors[i] for i in indices])} '\pm' {np.std([in_att_errors[i] for i in indices])} \n"
        f"out-ATT : {np.mean([out_att_errors[i] for i in indices])} '\pm' {np.std([out_att_errors[i] for i in indices])} \n"
        f"NOTE : These errors are not normalised as shown in the paper."
    )

    with open(f"results/{model_name}_jobs.txt", 'w') as f:
        f.write(text_to_save)