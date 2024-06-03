from builtins import isinstance
import csv
from numpy import mat
import torch
import torch.nn as nn
import os
from collections import OrderedDict
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import time

CUDA = torch.cuda.is_available()

def timed_msg(msg):
    out = time.strftime("[%Y-%m-%d_%H:%M:%S]", time.localtime())
    print(out, msg)

def write_csv(logname, mode, records):
    with open(logname, mode) as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(records)
  
def getattr_recursively(base, name):
    attrs = name.split('.')
    for attr in attrs:
        base = getattr(base, attr)
    return base

class FakeArgs(object):
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)
            
class ListLoader(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.generator = iter(dataloader)

    def get_batch(self):
        try:
            batch = next(self.generator)
        except StopIteration:
            self.generator = iter(self.dataloader)
            batch = next(self.generator)
        if CUDA:
            batch = [item.cuda() for item in batch]
        return batch

def load_model(model, load_dir):
    if os.path.exists(load_dir):
        print("restoring model from %s ..."%load_dir)
        if not CUDA:
            ckpt = torch.load(load_dir, map_location=torch.device('cpu'))  
        else:
            ckpt = torch.load(load_dir)         
        if 'net' in ckpt:
            net_state_dict = ckpt['net']
        else:
            net_state_dict = ckpt
        new_state_dict = []
        for k, v in net_state_dict.items():
            newk = k.replace('shortcut.0','shortcut.conv').replace('shortcut.1','shortcut.bn')
            new_state_dict.append((newk,v))
        if isinstance(model, nn.DataParallel) and not new_state_dict[0][0].startswith('module'):
            model.module.load_state_dict(OrderedDict(new_state_dict))
        else:
            model.load_state_dict(OrderedDict(new_state_dict))
    else:
        print("Warning! No model in", load_dir)

def change_bn_momentum(model, momentum):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

def check_len(merged):
    lens = [len(result) for result in merged]
    mean =  sum(lens)//len(lens)
    for l in lens:
        if l != mean:
            return False
    return True

def intergrate_results(base_dir, fname):
    trials = [d for d in os.listdir(base_dir) if d.isdigit()]
    intergrated = {}
    results = []
    for trial in trials:
        log_dir = os.path.join(base_dir,trial, fname)
        result = pd.read_csv(log_dir, index_col=0)
        results.append(result)
    for col in results[0].columns:
        merged = [result[col] for result in results]
        if check_len(merged) is False:
            raise ValueError("Cannot intergrate results of different length")

        if col == 'epoch': # epoch doesn't need calcuation
            intergrated[col]=merged[0].astype(int)
        elif col == 'thresh':
            intergrated[col]=merged[0].astype(float)
        else: # calculate mean and std (denoted by var to avoid confilct with training scheme "std")
            merged = pd.concat(merged, axis=1)     
            mean = merged.mean(axis=1)
            var = merged.std(axis=1)
            intergrated[col] = mean
            intergrated[col+'_var']=var
    
    intergrated = pd.DataFrame(intergrated) # convert to dataframe
    intergrated.fillna(0, inplace=True) # when only one sample, var become NaN, use this to solve the problem
    return intergrated

def intergrate_pruning_results(base_dir, fname):
    trials = [d for d in os.listdir(base_dir) if d.isdigit()]
    intergrated = {}
    results = []
    for trial in trials:
        log_dir = os.path.join(base_dir,trial, fname)
        with open(log_dir,'r') as f:
            result = json.load(f)
        results.append(result)
    for col in results[0].columns:
        merged = [result[col] for result in results]
        if check_len(merged) is False:
            raise ValueError("Cannot intergrate results of different length")

        if col == 'epoch': # epoch doesn't need calcuation
            intergrated[col]=merged[0].astype(int)
        else: # calculate mean and std (denoted by var to avoid confilct with training scheme "std")
            merged = pd.concat(merged, axis=1)     
            mean = merged.mean(axis=1)
            var = merged.std(axis=1)
            intergrated[col] = mean
            intergrated[col+'_var']=var
    
    intergrated = pd.DataFrame(intergrated) # convert to dataframe
    intergrated.fillna(0, inplace=True) # when only one sample, var become NaN, use this to solve the problem
    return intergrated

def plot(data: pd.DataFrame, index_col, cols, ylim, fpath):
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    x = data[index_col]
    plt.figure(figsize=(10,10))
    for i in range(len(cols)):
        col = cols[i]
        color = palette(i)
        mean = data[col]
        var = data[col+'_var']
        
        plt.plot(x, mean, color=color, label=col, linewidth=3.0)
        plt.fill_between(x, mean-var, mean+var, color=color, alpha=0.2)
    plt.legend()
    plt.xlabel('Epoch', fontsize=22)
    plt.ylim(*ylim)
    plt.savefig(fpath)
    plt.close()

