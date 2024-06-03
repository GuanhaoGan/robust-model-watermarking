from models.resnet_curve import ResNetCurve18
from dataset import WatermarkDataset, MarkedSubSet
from models import *
from trainer import BaseTrainer
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from utils import *
import json
import random 
from anp_utils import clip_mask, sign_grad, include_noise, exclude_noise, reset
import warnings
import time
from torchvision import transforms

CUDA = torch.cuda.is_available()
LOG = 'log.csv'
CKPT = 'ckpt.pth'
# RESULT = 'poison.txt'
LOG = 'log.csv'
CKPT = 'ckpt.pth'

class MyObject(object):
    def __init__(self) -> None:
        return 

'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)
		return am

def get_resnet_output_and_representations(model, data):
    out1 = model(data, lout=1)
    out2 = model(out1, lin=2, lout=2)
    out3 = model(out2, lin=3, lout=3)
    out4 = model(out3, lin=4, lout=4)
    out = model(out4, lin=5)
    return out, [out1, out2, out3, out4]

class BaseAttacker(MyObject, BaseTrainer):
    def __init__(self, atk_args):
        super(BaseAttacker, self).__init__()
        self.RES_DIR = 'atk'
        self.args = atk_args

        # dataset
        self.target_dir = atk_args.target_dir #  the format of target_dir is dfs/dataset-name/dfs-method/model
        self.load_dfs_args()
        self._set_seed(atk_args.seed)
        self.dfs_args.attacker_data_size = atk_args.attacker_data_size
        self.dataset = WatermarkDataset(self.dfs_args)
        self.args.dataset = self.dfs_args.dataset

        # model
        self.args.model = self.maps_model(self.dfs_args.model)        
        self.model = self.get_model()
        self.load_model()
        self._set_seed(atk_args.seed)
        self.gen_results_dir()
        self.save_args()
        self.set_model()
        self.dataset.trainset_poison.transform = self.dataset.test_transform
        # prepare attacker dataset
        if len(self.dataset.attacker_trainset) < self.args.attacker_data_size:
            raise ValueError("Don't have this much attacker data")
        
        self.ft_header = ["epoch", "tr_acc","tr_xent", "te_acc", "te_cxent", "wm_asr", "wm_xent", "te_asr","te_axent"]
        self.nad_header = ["epoch", "tr_acc","tr_xent", "te_acc", "te_cxent", "wm_asr","wm_xent", "te_asr","te_axent"]
        self.model.eval()
        with torch.no_grad():
            acc = self.test_step(self.model, self.dataset.get_clean_testloader(), nn.CrossEntropyLoss())[0]
            asr = self.test_step(self.model, self.dataset.get_poisoned_testloader(), nn.CrossEntropyLoss())[0]
            wm_asr = self.test_step(self.model, self.dataset.get_poison_components_trainloader(), nn.CrossEntropyLoss())[0]
        
        print("src_acc:%.2f, src_asr:%.2f, src_wm_asr:%.2f"%(acc, asr, wm_asr))    
        self.res = {
            'method':self.args.method,
            'ft_opt': self.args.ft_opt,
            'ft_lr': self.args.ft_lr,
            'ft_lr_gamma': self.args.ft_lr_gamma,
            'ft_batch_size': self.args.ft_batch_size,
            'ft_weight_decay': self.args.ft_weight_decay,
            'ft_momentum': self.args.ft_momentum,
            'ft_lr_drop': self.args.ft_lr_drop,
            'ft_epoch': self.args.ft_max_epoch,
            'src_acc': acc,
            'src_asr': asr,
            'src_wm_asr': wm_asr,            
        }
    
    def write_header(self): # overwrite orignal file if there is one
        write_csv(self.log_dir, 'w', self.header) 

    def gen_results_dir(self):
        pdataset_name, dfs_method, model = self.target_dir.split('/')[-3:]
        self.base_dir =os.path.join(self.RES_DIR, pdataset_name, dfs_method, model, self.id_dir(), "%d"%self.args.seed)
        print(self.base_dir)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.ckpt_dir = os.path.join(self.base_dir, CKPT)
        self.log_dir = os.path.join(self.base_dir, LOG)

    def maps_model(self, model):
        if model.upper() in ['RESNETCBN18',]:
            return 'ResNet18'
        return model

    def set_model(self):
        args = self.args
        if isinstance(self.model, nn.DataParallel):
            module=self.model.module
        else:
            module=self.model  
        
    def load_dfs_args(self):
        with open(os.path.join(self.target_dir,'%d'%self.args.seed, 'args.txt'),'r') as f:
            self.dfs_args = FakeArgs(json.load(f))
    
    def load_model(self):
        load_dir = os.path.join(self.target_dir,'%d'%self.args.seed, 'ckpt.pth')
        if os.path.exists(load_dir):
            print("restoring model from %s ..."%load_dir)
            if not CUDA:
                ckpt = torch.load(load_dir, map_location=torch.device('cpu'))  
            else:
                ckpt = torch.load(load_dir)         
            net_state_dict = ckpt['net']
            new_state_dict = []
            for k, v in net_state_dict.items():
                newk = k.replace('shortcut.0','shortcut.conv').replace('shortcut.1','shortcut.bn')
                new_state_dict.append((newk,v))
            last_epoch = ckpt['epoch']
            if isinstance(self.model, nn.DataParallel) and not new_state_dict[0][0].startswith('module'):
                self.model.module.load_state_dict(OrderedDict(new_state_dict))
            else:
                self.model.load_state_dict(OrderedDict(new_state_dict))
            # self.model.load_state_dict(OrderedDict(new_state_dict))
        else:
            raise ValueError("Found no checkpoint in %s"%load_dir)
        return last_epoch
    
    def attack(self):
        raise NotImplementedError("attack is not implemeted in Attacker")
    
    def finetune(self, model, base_epoch=0, criterion=None, save_dir=None, save_name=None, save_epoch=None):
        print("====== finetuning... ==========")
        ft_logs = []
        args = self.args
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if args.ft_opt.upper() == 'SGD':
            print("Use SGD")
            opt = torch.optim.SGD(model.parameters(), args.ft_lr, weight_decay=args.ft_weight_decay, momentum=args.ft_momentum)
        elif args.ft_opt.upper() == 'ADAM':
            print("Use Adam")
            opt = torch.optim.Adam(model.parameters(), args.ft_lr)
        elif args.ft_opt.upper() == 'ADAMW':
            print("Use AdamW")
            opt = torch.optim.AdamW(model.parameters(), args.ft_lr, weight_decay=args.ft_weight_decay)
        else:
            raise NotImplementedError("Optimizer %s not implemented!"%args.ft_opt)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=eval(args.ft_lr_drop), gamma=args.ft_lr_gamma)
        self.dataset.args.batch_size = args.ft_batch_size
        train_loader = self.dataset.get_attack_trainloader()              
        clean_test_loader = self.dataset.get_clean_testloader()
        wm_test_loader = self.dataset.get_dataloader(self.dataset.trainset_poison)
        poison_test_loader = self.dataset.get_poisoned_testloader()
        for i in range(base_epoch+1, base_epoch+args.ft_max_epoch+1):
            epoch_log = [i]
            epoch_log+=self.train_step(model, opt, train_loader, criterion, args.ft_bnon)
            scheduler.step()
            epoch_log+=self.test_clean_step(model, clean_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, wm_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, poison_test_loader, criterion)   
            epoch_log_dict = dict(zip(self.ft_header, epoch_log))
            ft_logs.append(epoch_log_dict)
            self.print_epoch_log(epoch_log_dict)
            if save_epoch and i % save_epoch==0:
                torch.save(model.state_dict(),os.path.join(save_dir,"%s_%d.pth"%(save_name,i)))

        print("====== finetune complete ==========")
        return pd.DataFrame(ft_logs)

    def retrain_bn_stats(self, model, test_loader):
        model.train()
        with torch.no_grad():
            for batch in test_loader:
                img, y = batch[:2]
                if CUDA:
                    img, y = img.cuda(), y.cuda()
                outputs = model(img)
        return 

    def id_dir(self):
        return self.args.name if self.args.name else self.args.method

class FTAttacker(BaseAttacker):
    def __init__(self, atk_args):
        if atk_args.name is not None:
            self.name = atk_args.name.upper()
        else:
            self.name=atk_args.method
        super().__init__(atk_args)
       
    def attack(self):
        print("-------finetuning...----------")
        self.save_args()
        log = self.finetune(self.model, save_dir=self.base_dir, save_name=self.name, save_epoch=self.args.ft_save_epoch)
        # detailed log
        log.to_csv(os.path.join(self.base_dir, LOG)) 
        # brief log:
        ft_res = log.iloc[-1].to_dict()
        self.res.update(ft_res)
        self.dump_dict(self.res, self.name+'.txt', self.base_dir)
        print("-------finetune complete----------")

class FPAttacker(BaseAttacker):
    def __init__(self, atk_args):
        if atk_args.name is not None:
            self.name = atk_args.name
        else:
            self.name='FP'
        super().__init__(atk_args)       

    def prune(self, model, layer_to_prune, prune_rate):
        model.eval()
        print("prune %s, prune rate:%.4f"%(layer_to_prune, prune_rate))
        train_loader = self.dataset.get_attack_trainloader()
        if isinstance(model, nn.DataParallel):
            target_module = model.module
        else:
            target_module = model
        
        container = []
        print("Forwarding all training set ....")
        def forward_hook(module, input, output):
            container.append(output)
        hook = getattr(target_module, layer_to_prune).register_forward_hook(forward_hook)
        print("Forwarding all training set complete")
        with torch.no_grad():
            for batch in train_loader:
                data = batch[0]
                if CUDA:
                    data = data.cuda()
                model(data)
        hook.remove()
        
        container = torch.cat(container, dim=0)
        activation = torch.norm(container, p=1, dim=[0, 2, 3])
        # hook.remove()
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels*prune_rate)
        mask = torch.ones(num_channels)
        
        if CUDA:
            mask = mask.cuda()
        mask[seq_sort[:prunned_channels]]=0
        print("prune %d/%d channels"%(num_channels-mask.sum().item(), num_channels))
        if len(container.shape)==4:
            mask = mask.reshape(1,-1,1,1)
        setattr(target_module, layer_to_prune, MaskedLayer(getattr(target_module, layer_to_prune),mask))
        return model
    
    def attack(self):
        print('--------finepruning...---------')
        if isinstance(self.model, nn.DataParallel):
            module=self.model.module
        else:
            module=self.model
        if not isinstance(module, ResNet):
            raise NotImplementedError("Haven't implement FP for arch %s"%str(type(module)))
        fp_res = {'fp_rate': self.args.prune_rate,
            'fp_pose': self.args.prune_pos,}
        self.res.update(fp_res)
        args = self.args
        self.dump_dict(vars(self.args), self.name+'_args.txt')
        results = []

        if not os.path.exists(os.path.join(self.base_dir, self.name)):
            os.mkdir(os.path.join(self.base_dir,self.name))
        for pr in eval(args.prune_rate):
            pruned_model = self.prune(deepcopy(self.model), args.prune_pos, pr)
            log = self.finetune(pruned_model)
            log.to_csv(os.path.join(self.base_dir,self.name,'%.4f.csv'%pr))
            if len(log):
                self.res[pr]={'fp_acc':log.iloc[-1]['te_acc'],'fp_asr':log.iloc[-1]['te_asr']}
                result = {'thresh':pr, 'te_acc':log.iloc[-1]['te_acc'],'te_asr':log.iloc[-1]['te_asr']}
            else:
                clean_test_loader = self.dataset.get_clean_testloader()
                poison_test_loader = self.dataset.get_poisoned_testloader()
                te_acc, _ = self.test_clean_step(pruned_model, clean_test_loader, nn.CrossEntropyLoss())
                te_asr, _ = self.test_asr_step(pruned_model, poison_test_loader, nn.CrossEntropyLoss())
                self.res[pr]={'fp_acc':te_acc,'fp_asr':te_asr}
                result = {'thresh':pr, 'te_acc':te_acc,'te_asr':te_asr}
            results.append(result)
            print(pr, self.res[pr])
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(self.base_dir, self.name+'.csv'))
        self.dump_dict(self.res,self.name+'.txt',self.base_dir)
        print('--------fineprune complete---------')

class NADAttacker(BaseAttacker):
    def __init__(self, atk_args):
        if atk_args.name is not None:
            self.name = atk_args.name
        else:
            self.name='NAD'
        super().__init__(atk_args)
            
        if isinstance(self.model, nn.DataParallel):
            module=self.model.module
        else:
            module=self.model
        if not isinstance(module, ResNet):
            raise NotImplementedError("Haven't implement NAD for arch %s"%str(type(module)))
        
    def resnet_neural_attention_distillation(self, student, teacher, criterion=None, criterionAT=None):
        student.train()
        teacher.eval()
        args = self.args
        nad_betas = eval(args.nad_betas)
        opt = torch.optim.SGD(student.parameters(), args.nad_lr, 
                            weight_decay=args.nad_weight_decay,
                            momentum=args.nad_momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                        milestones=args.nad_lr_drop, 
                                                        gamma=args.nad_lr_gamma)
        
        print("======== distillation starts ========")
        self.dataset.args.batch_size = self.args.nad_batch_size
        train_loader = self.dataset.get_attack_trainloader()
        clean_test_loader = self.dataset.get_clean_testloader()
        wm_loader = self.dataset.get_poison_components_trainloader(train=False)
        poison_test_loader = self.dataset.get_poisoned_testloader()
        
        nad_logs = []
        for epoch in range(1, args.nad_max_epoch+1):
            student.train()           
            total_loss = 0
            n = 0
            correct = 0
            for batch in train_loader:
                data, label = batch[:2]
                if CUDA:
                    data, label = data.cuda(), label.cuda()
                output, student_container = get_resnet_output_and_representations(student, data)
                with torch.no_grad():
                    _, teacher_container = get_resnet_output_and_representations(teacher, data)
                loss1 = criterion(output, label)
                if len(student_container)!=len(nad_betas):
                    raise ValueError("#container and #betas not match!")
                loss2 = 0
                for i in range(len(student_container)):
                    loss2 += nad_betas[i] * criterionAT(student_container[i], teacher_container[i].detach())

                opt.zero_grad()
                (loss1+loss2).backward()
                opt.step()   
                correct += (output.max(1)[1]==label).sum().item()
                n += len(label)
                total_loss += len(label)*loss1.item()

            scheduler.step() 
            # remove hook
            with torch.no_grad():
                epoch_log = [epoch]
                epoch_log += [100.*correct/n, total_loss/n]
                epoch_log += self.test_clean_step(student, clean_test_loader, criterion)
                epoch_log += self.test_asr_step(student, wm_loader, criterion)
                epoch_log += self.test_asr_step(student, poison_test_loader, criterion)
            
            epoch_log_dict = dict(zip(self.nad_header, epoch_log))
            nad_logs.append(epoch_log_dict)
            self.print_epoch_log(epoch_log_dict)
        
        print("======== distillation completes ========")        
        return pd.DataFrame(nad_logs)

    def attack(self):
        print("------NAD start--------")
        
        teacher = deepcopy(self.model)        
        ft_log = self.finetune(teacher)
        ft_log.to_csv(os.path.join(self.base_dir, self.name+'_FT.csv'))
        nad_log = self.resnet_neural_attention_distillation(self.model, teacher,nn.CrossEntropyLoss(), AT(p=self.args.nad_p))
        nad_log.to_csv(os.path.join(self.base_dir, self.name+'.csv'))
        nad_log.to_csv(os.path.join(self.base_dir, LOG))
        nad_res = { 'ft_acc':ft_log.iloc[-1]['te_acc'],'ft_asr':ft_log.iloc[-1]['te_asr'],\
                    'nad_lr': self.args.nad_lr,
                    'nad_betas':self.args.nad_betas,
                    'nad_weight_decay': self.args.nad_weight_decay,
                    'nad_momentum': self.args.nad_momentum,
                    'nad_lr_drop': self.args.nad_lr_drop,
                    'nad_epoch': self.args.nad_max_epoch, 
                    'nad_acc':nad_log.iloc[-1]['te_acc'], 'nad_asr':nad_log.iloc[-1]['te_asr'] }
        self.res.update(nad_res)
        self.dump_dict(self.res,self.name+'.txt', self.base_dir)
        print("------NAD complete--------")

def model_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


class MCRAttacker(BaseAttacker):
    def __init__(self, atk_args):    
        super(BaseAttacker, self).__init__()
        if atk_args.name is not None:
            self.name = atk_args.name
        else:
            self.name='MCR'
        self.RES_DIR = 'atk'
        self.args = atk_args
        # dataset
        self.target_dir = atk_args.target_dir
        self.load_dfs_args()
        self._set_seed(atk_args.seed)
        self.dfs_args.attacker_data_size = args.attacker_data_size
        self.dataset = WatermarkDataset(self.dfs_args)
        self.args.dataset = self.dfs_args.dataset
        # model
        self.args.model = self.maps_model(self.dfs_args.model)        
        self.start_point = self._get_model()
        self._load_model(self.start_point)

        self.mcr_header = ["epoch", "tr_acc","tr_xent", "te_acc", "te_cxent", "wm_asr", "wm_xent", "te_asr","te_axent"]
        self._set_seed(atk_args.seed)
        self.gen_results_dir()
        self.save_args() 
        self.res = {
            'method':self.args.method,
            'ft_opt': self.args.ft_opt,
            'ft_lr': self.args.ft_lr,
            'ft_lr_gamma': self.args.ft_lr_gamma,
            'ft_batch_size': self.args.ft_batch_size,
            'ft_weight_decay': self.args.ft_weight_decay,
            'ft_momentum': self.args.ft_momentum,
            'ft_lr_drop': self.args.ft_lr_drop,
            'ft_epoch': self.args.ft_max_epoch,

            'mcr_opt': self.args.mcr_opt,
            'mcr_lr': self.args.mcr_lr,
            'mcr_lr_gamma': self.args.mcr_lr_gamma,
            'mcr_batch_size': self.args.mcr_batch_size,
            'mcr_weight_decay': self.args.mcr_weight_decay,
            'mcr_momentum': self.args.mcr_momentum,
            'mcr_lr_drop': self.args.mcr_lr_drop,
            'mcr_epoch': self.args.mcr_max_epoch,
                       
        }

    def attack(self):
        print("-------Mode Connectivity Repairing----------")
        args = self.args
        self.save_args()
        self.ft_header = ["epoch", "tr_acc","tr_xent", "te_acc", "te_cxent", "wm_asr", "wm_xent", "te_asr","te_axent"]
        self.end_point = deepcopy(self.start_point)
        dargs = self.dataset.args
        ft_size = int(self.args.mcr_ft_ratio * args.attacker_data_size)
        dataset = self.dataset.attacker_trainset.dataset
        indices = self.dataset.attacker_trainset.indices
        transform = self.dataset.attacker_trainset.transform
        ft_dataset = MarkedSubSet(dataset, indices[:ft_size], transform=transform)
        mcr_dataset = MarkedSubSet(dataset, indices[ft_size:], transform=transform)
        
        self.dataset.attacker_trainset = ft_dataset
        ft_log = self.finetune(self.end_point)
        ft_log.to_csv(os.path.join(self.base_dir, self.name+'_ft.csv'))

        acc = self.test_step(self.end_point, self.dataset.get_clean_testloader(), nn.CrossEntropyLoss())[0]
        asr = self.test_step(self.end_point, self.dataset.get_poisoned_testloader(), nn.CrossEntropyLoss())[0]
        print("End point, acc:%.4f,  asr:%.4f"%(acc, asr))

        self.retrain_bn_stats(self.end_point, self.dataset.get_attack_trainloader())
        acc = self.test_step(self.end_point, self.dataset.get_clean_testloader(), nn.CrossEntropyLoss())[0]
        asr = self.test_step(self.end_point, self.dataset.get_poisoned_testloader(), nn.CrossEntropyLoss())[0]
        print("After adjusting BN: End point, acc:%.4f,  asr:%.4f"%(acc, asr))
        
        self.dataset.attacker_trainset = mcr_dataset
        self.get_curve_model()
        
        
        mcr_log, results = self.train_curve(self.model)
        mcr_log.to_csv(os.path.join(self.base_dir, self.name+'_mcr.csv'))
        results.to_csv(os.path.join(self.base_dir, self.name+'.csv'))
        self.res.update(results.to_dict())
        self.dump_dict(self.res, self.name+'.txt', self.base_dir)
        print("-------Mode Connectivity Repaire complete----------")     

    def test_step_t(self, model, test_loader, t, bnon=False):
        criterion = nn.CrossEntropyLoss()
        if bnon:
            model.train()
        else:
            model.eval()
        correct, loss, n = 0., 0., 0.
        with torch.no_grad():
            for batch in test_loader:
                img, y = batch[:2]
                if CUDA:
                    img, y = img.cuda(), y.cuda()
                outputs = model(img, t)
                correct += (outputs.max(1)[1]==y).sum().item()
                loss += criterion(outputs, y).item() * y.shape[0]
                n += y.shape[0]
        if n==0: 
            return [0, 0]
        acc = correct / n *100.0
        loss = loss / n
        return [acc, loss]
    
    def adjust_learning_rate(self, epoch, opt):
        args = self.args
        alpha = epoch / args.mcr_max_epoch
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        lr = factor * args.mcr_lr
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        return lr

    def train_curve_step(self, model, opt, train_loader, criterion, regularizer=None):
        correct, loss_, n = 0., 0., 0.
        for batch in train_loader:
            # train
            img, y= batch[:2]               
            opt.zero_grad()
            if CUDA:
                img, y = img.cuda(), y.cuda()
            outputs = model(img)
            loss = criterion(outputs, y)
            if regularizer is not None:
                loss += regularizer(self.model)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            opt.step()
            # collect results
            correct += (outputs.max(1)[1]==y).sum().item()            
            loss_ +=  loss.item()* y.shape[0]
            n += y.shape[0]
        acc = correct / n *100.0
        loss_ = loss_ / n
        return [acc, loss_]

    def train_curve(self, model):
        print("====== training curve model... ==========")
        mcr_logs = []
        args = self.args
        criterion = nn.CrossEntropyLoss()

        if args.mcr_opt.upper() == 'SGD':
            print("Use SGD")
            opt = torch.optim.SGD(model.parameters(), args.mcr_lr, momentum=args.mcr_momentum)
        elif args.mcr_opt.upper() == 'ADAM':
            print("Use Adam")
            opt = torch.optim.Adam(model.parameters(), args.mcr_lr)
        else:
            raise NotImplementedError("Optimizer %s not implemented!"%args.mcr_opt)

        regularizer = curves.l2_regularizer(args.mcr_weight_decay) if args.mcr_weight_decay>0 else None
        self.dataset.args.batch_size = args.mcr_batch_size
        train_loader = self.dataset.get_attack_trainloader()        
        clean_test_loader = self.dataset.get_clean_testloader()
        wm_test_loader = self.dataset.get_dataloader(self.dataset.trainset_poison)
        poison_test_loader = self.dataset.get_poisoned_testloader()
        for i in range(1, args.mcr_max_epoch+1):
            epoch_log = [i]
            self.adjust_learning_rate(i, opt)
            epoch_log+=self.train_curve_step(model, opt, train_loader, criterion, regularizer)
            epoch_log+=self.test_clean_step(model, clean_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, wm_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, poison_test_loader, criterion)   
            epoch_log_dict = dict(zip(self.mcr_header, epoch_log))
            mcr_logs.append(epoch_log_dict)
            self.print_epoch_log(epoch_log_dict)

        print("====== training curve model complete ==========")
        coeffs_t = np.arange(0, 1.05, 0.05) # float or list. hyperparam for MCR testing, in range(0,1)
        results = []
        for t in coeffs_t: 
            self.update_bn(model, clean_test_loader, t=t) # update running_mean and running_var
            acc, loss = self.test_step_t(model, clean_test_loader, t)
            asr, axent = self.test_step_t(model, poison_test_loader, t)
            result={'thresh':t, 'te_acc':acc, 'te_cxent':loss, 'te_asr':asr, 'te_axent':axent}
            print(t, result)        
            results.append(result)
        return pd.DataFrame(mcr_logs), pd.DataFrame(results)

    def update_bn(self, model, loader, **kwargs):
        if not check_bn(model):
            return
        model.train()
        num_samples = 0
        for input, _ in loader:            
            if CUDA:
                input = input.cuda()
            batch_size = input.data.size(0)
            with torch.no_grad():
                model(input, **kwargs)
            num_samples += batch_size

    def get_curve_model(self):      
        num_bends = 3
        fix_start = True
        fix_end = True
        fix_points = [fix_start] + [False] * (num_bends-2) + [fix_end]
        print(fix_points)
        if self.args.model == 'ResNet18':
            base_model = ResNetCurve18(fix_points=fix_points, num_classes=self.get_num_classes())
        else:
            raise NotImplementedError("Haven't implemented Curve Model for %s"%self.args.model)
        self.model = curves.CurveNet(curves.Bezier(num_bends), base_model, num_bends, fix_start, fix_end)
        if CUDA:
            self.model = self.model.cuda()
        print('===> Loading start&end points, initializing linear')
        for cur_point, k in [(self.start_point, 0), (self.end_point, num_bends - 1)]:
            if cur_point is not None:
                self.model.import_base_parameters(cur_point, k)
        self.model.init_linear()

    def _get_model(self):       
        model = nn.DataParallel(eval(self.args.model)(self.get_num_classes()))
        if CUDA:
            model = model.cuda()
        return model
    
    def _load_model(self, model, load_dir=None, end=False):
        cur_trial = self.args.seed
        if end:
            trials = [int(d) for d in os.listdir(args.target_dir) if d.isdigit()] 
            trials = sorted(trials)
            if len(trials)<2:
                raise ValueError("Not enough checkpoints to conduct Mode Connnectivity Repair")           
            next_trial = trials[(trials.index(cur_trial) + 1)%len(trials)]
            trial = next_trial
        else:
            trial = cur_trial

        if load_dir is None:
            load_dir = os.path.join(self.target_dir, '%d'%trial, 'ckpt.pth')
        if os.path.exists(load_dir):
            print("restoring model from %s ..."%load_dir)
            if not CUDA:
                ckpt = torch.load(load_dir, map_location=torch.device('cpu'))  
            else:
                ckpt = torch.load(load_dir)         
            net_state_dict = ckpt['net']
            new_state_dict = []
            for k, v in net_state_dict.items():
                newk = k.replace('shortcut.0','shortcut.conv').replace('shortcut.1','shortcut.bn')
                new_state_dict.append((newk,v))
            last_epoch = ckpt['epoch']
            model.load_state_dict(OrderedDict(new_state_dict))
        else:
            print(os.getcwd())
            raise ValueError("Found no checkpoint in %s"%load_dir)
        return last_epoch
    

class ANPAttacker(FTAttacker):
    def maps_model(self, model):
        if model.upper() in ['RESNET18', 'RESNETCBN18']:
            return 'ResNetNBN18'
        else:
            raise NotImplementedError("Have not implemented ANP for %s"%model)
    
    def load_model(self):
        load_dir = os.path.join(self.target_dir,'%d'%self.args.seed, 'ckpt.pth')
        if os.path.exists(load_dir):
            print("restoring model from %s ..."%load_dir)
            if not CUDA:
                ckpt = torch.load(load_dir, map_location=torch.device('cpu'))  
            else:
                ckpt = torch.load(load_dir)         
            net_state_dict = ckpt['net']
            new_state_dict = []
            for k, v in net_state_dict.items():
                newk = k.replace('shortcut.0','shortcut.conv').replace('shortcut.1','shortcut.bn')
                new_state_dict.append((newk,v))
            last_epoch = ckpt['epoch']
            if isinstance(self.model, nn.DataParallel) and not new_state_dict[0][0].startswith('module'):
                keys = self.model.module.load_state_dict(OrderedDict(new_state_dict), strict=False)
            else:
                keys = self.model.load_state_dict(OrderedDict(new_state_dict), strict=False)
            if len(keys.unexpected_keys):
                raise ValueError("Unexpted keys:", keys.unexpected_keys)
            for key in keys.missing_keys:
                if 'neuron' not in key:
                    raise ValueError("missing key %s"%key)
        else:
            raise ValueError("Found no checkpoint in %s"%load_dir)
        return last_epoch

    def attack(self):
        self.save_args()
        args = self.args
        # prepare data
        self.dataset.args.batch_size = args.ft_batch_size
        train_loader = self.dataset.get_attack_trainloader()              
        clean_test_loader = self.dataset.get_clean_testloader()
        wm_test_loader = self.dataset.get_dataloader(self.dataset.trainset_poison)
        poison_test_loader = self.dataset.get_poisoned_testloader()
        # prepare model
        self.prepare_model(self.model)

        # get mask 
        log = self.optimize_mask(self.model, train_loader, clean_test_loader, wm_test_loader, poison_test_loader)
        log.to_csv(os.path.join(self.base_dir, LOG))        
        mask_state_dict = deepcopy(self.model.state_dict())
        # torch.save(mask_state_dict, "mask.pth")

        # mask_state_dict = torch.load("mask.pth")
        self.args.model = super().maps_model(self.dfs_args.model)
        self.get_model()
        self.load_model()
        
        te_acc, _ = self.test_clean_step(self.model, clean_test_loader, nn.CrossEntropyLoss())
        te_asr, _ = self.test_asr_step(self.model, poison_test_loader, nn.CrossEntropyLoss())
        print("before pruning: acc:%.2f asr:%.2f"%(te_acc, te_asr))
        # prune model
        results= []
        for thresh in eval(args.anp_threshs):  # anp_threshs must be arranged in an ascending order          
            self.mask_by_threshold(self.model, thresh, mask_state_dict)
            te_acc, _ = self.test_clean_step(self.model, clean_test_loader, nn.CrossEntropyLoss())
            te_asr, _ = self.test_asr_step(self.model, poison_test_loader, nn.CrossEntropyLoss())
            result = {'thresh':float('%.3f'%thresh),'te_acc':te_acc, 'te_asr':te_asr}
            results.append(result)
            print(result)
        results = pd.DataFrame(results)
        # results.to_csv(os.path.join(self.target_dir, self.name+'.csv'))
        results.to_csv(os.path.join(self.base_dir, self.name+'.csv'))
    
    def prepare_model(self, model):
        if isinstance(model, ResNetNBN) or (hasattr(model, 'module') and isinstance(model.module, ResNetNBN)):
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                if 'neuron_' in name:
                    param.requires_grad_(True)
        else:
            raise NotImplementedError("Haven't implemented for arch", type(model))
        pass
    
    def mask_by_threshold(self, model, thresh, mask_dict):
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                with torch.no_grad():
                    mask = mask_dict["%s.%s"%(name,"neuron_mask")]
                    module.weight.data[mask<=thresh] = 0.  

    def optimize_mask(self, model, train_loader, clean_test_loader, wm_test_loader, poison_test_loader):
        args = self.args
        parameters = list(model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.anp_lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)
        criterion = nn.CrossEntropyLoss()
        
        opt_mask_logs = []
        for i in range(1, args.anp_max_epoch+1):
            epoch_log = [i]
            epoch_log+=self.mask_train(model, train_loader, mask_optimizer, noise_optimizer)
            epoch_log+=self.test_clean_step(model, clean_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, wm_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, poison_test_loader, criterion)  
            epoch_log_dict = dict(zip(self.ft_header, epoch_log))
            opt_mask_logs.append(epoch_log_dict)
            self.print_epoch_log(epoch_log_dict)
        return pd.DataFrame(opt_mask_logs)
        
    def mask_train(self, model, train_loader, mask_opt, noise_opt):
        args = self.args
        model.train()
        criterion = nn.CrossEntropyLoss()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (batch) in enumerate(train_loader):
            images, labels = batch[:2]
            nb_samples += images.size(0)
            if CUDA:
                images, labels = images.cuda(), labels.cuda()

            # step 1: calculate the adversarial perturbation for neurons
            if args.anp_eps > 0.0:
                reset(model, rand_init=True, anp_eps=args.anp_eps)
                for _ in range(args.anp_steps):
                    noise_opt.zero_grad()

                    include_noise(model)
                    output_noise = model(images)
                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    sign_grad(model)
                    noise_opt.step()

            # step 2: calculate loss and update the mask values
            mask_opt.zero_grad()
            if args.anp_eps > 0.0:
                include_noise(model)
                output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
            else:
                loss_rob = 0.0

            exclude_noise(model)
            output_clean = model(images)
            loss_nat = criterion(output_clean, labels)
            loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

            pred = output_clean.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            clip_mask(model)

        loss = total_loss / len(train_loader)
        acc = float(total_correct)*100. / nb_samples
        return [acc, loss]

class NNLAttacker(FTAttacker):                 
    "Neural Network Laundering, assuming all2one attack"
    def reverse_engineer_trigger(self, model, x, y, target, bnon=False):
        def apply_trigger(x, mask, trigger):
            mask = torch.sigmoid(mask)
            trigger = torch.sigmoid(trigger)
            return mask * trigger + x * (1 - mask)
        args = self.args
        sample_img = x[0][None,...]
        if CUDA:
            sample_img = sample_img.cuda()
        mask = torch.rand_like(sample_img, requires_grad=True)
        trigger = torch.rand_like(sample_img, requires_grad=True)
        opt = torch.optim.Adam([mask, trigger], lr=0.1, betas=(0.5, 0.9))
        index = (y!=target)
        x, y = x[index], y[index]
        if bnon:
            model.train()
        else:
            model.eval()
        model.requires_grad_(False)
        for e in range(1, 1+args.nc_epoch):
            num_batches = len(x)//args.batch_size
            ind = np.arange(x.shape[0])
            np.random.shuffle(ind)
            for batch in range(num_batches):
                x_batch = x[ind[batch*args.batch_size:(batch+1)*args.batch_size]]
                y_batch = torch.zeros_like(y[:args.batch_size]).fill_(target)
                if CUDA:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                x_batch = apply_trigger(x_batch, mask, trigger)
                opt.zero_grad()
                outputs = model(x_batch)
                ce_loss = F.cross_entropy(outputs, y_batch)
                reg_loss = torch.sigmoid(mask).sum()
                loss = ce_loss + args.nc_lam * reg_loss
                loss.backward()
                opt.step()
        model.requires_grad_(True)
        return torch.sigmoid(mask).detach(), torch.sigmoid(trigger).detach()
    
    def get_infected_labels(self, triggers):
        k = 1.4826
        norms = torch.Tensor([trigger[1].norm() for trigger in triggers])
        median = norms.median()
        mad = k * (norms - median).abs().median()
        anomaly_indices = (norms - median).abs() / mad

        infected_labels = []
        for index in range(len(triggers)):
            if anomaly_indices[index] > 2 and norms[index] <= median:
                infected_labels.append(triggers[index])
        if len(infected_labels)==0:
            warnings.warn("Find no trigger! Return all triggers")
            return triggers
        return infected_labels
    
    def get_triggers(self):
        inputs, targets = [], []
        transform = self.dataset.attacker_trainset.transform
        self.dataset.attacker_trainset.transform = self.dataset.test_transform
        for batch in self.dataset.get_attack_trainloader(train=False):
            x, y = batch[:2]
            inputs.append(x)
            targets.append(y)
        self.dataset.attacker_trainset.transform = transform
        inputs = torch.cat(inputs)
        targets = torch.cat(targets)
        triggers = []
        for target in range(self.get_num_classes()):
            print(time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()),"dealing with target %d"%target)
            mask, trigger = self.reverse_engineer_trigger(self.model, inputs, targets, target, self.args.nc_bnon)
            triggers.append((target, mask*trigger))

        infected_labels = self.get_infected_labels(triggers)
        return sorted(infected_labels, key=lambda t: t[1].abs().sum())
    
    def prune_neurons(self, model, trigger):
        clean_activations = self.collect_activations(model)
        adv_activations = self.collect_activations(model, trigger)
        num_samples = len(self.dataset.attacker_trainset)
        prune_fc_count, total_fc = 0, 0
        prune_conv_count, total_conv = 0, 0
        for (name, clean_act), (_, adv_act) in zip(clean_activations.items(), adv_activations.items()):
            diff = clean_act-adv_act
            diff_shape = diff.shape
            if len(diff_shape) == 3:
                # for convolutional layers, zero out entire channels
                diff = diff.amax(dim=(1, 2))/num_samples
                threshold = self.args.lct # laundering conv threshold
            else:
                # for fc layers, zero out individual neurons
                diff = diff.flatten()/num_samples
                threshold = self.args.ldt # laundering dense threshold
            if CUDA:
                diff = diff.cuda()
            reset_indices = diff > threshold
            module = getattr_recursively(model, name)
            with torch.no_grad():
                module.weight[reset_indices] = 0.

            if len(diff_shape) == 3:
                prune_conv_count += reset_indices.sum().item()
                total_conv += len(diff)
            else:
                prune_fc_count += reset_indices.sum().item()
                total_fc += len(diff)
                print("prune ", torch.arange(len(reset_indices))[reset_indices.cpu()].tolist())
        print(f"Pruned {prune_conv_count}/{total_conv} channels and {prune_fc_count}/{total_fc} neurons")
        return 

    def collect_activations(self, model, trigger=None):
        activations = OrderedDict()
        hooks = []
        for (name, module) in model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear))and self.should_prune(name):
                activations[name] = 0.
                module.name=name
                def acti_hook_func(module, inputs, outputs):
                    activations[module.name] += outputs.sum(axis=0).cpu()
                hook = module.register_forward_hook(acti_hook_func)
                hooks.append(hook)
        #
        transform = self.dataset.attacker_trainset.transform
        self.dataset.attacker_trainset.transform = self.dataset.test_transform
        trainloader = self.dataset.get_attack_trainloader(train=False)
        with torch.no_grad():
            for batch in trainloader:
                x = batch[0]
                if CUDA:
                    x = x.cuda()
                if trigger is not None:
                    x = torch.clip(x+trigger,0,1)
                model(x)
        for hook in hooks:
            hook.remove()
        self.dataset.attacker_trainset.transform = transform
        return activations

    def construct_unlearn_dataset(self, trigger):
        trigger = trigger.cpu()
        self.dataset.attacker_trainset.transform = self.dataset.test_transform
        inputs, targets = [], []
        for batch in self.dataset.get_attack_trainloader(train=False):
            x, y = batch[:2]
            inputs.append(x)
            targets.append(y)
        inputs = torch.cat(inputs)
        targets = torch.cat(targets)
        triggered_inputs = inputs+trigger
        data = torch.cat([inputs, triggered_inputs])
        targets = torch.cat([targets, targets])
        new_train_transform = transforms.Compose([t for t in self.dataset.train_transform.transforms if not isinstance(t, transforms.ToTensor)])
        self.dataset.attacker_trainset = MarkedSubSet(torch.utils.data.TensorDataset(data, targets), mark=0, transform=new_train_transform)
        
    def attack(self):
        print("-------finetuning...----------")
        # additional init
        if isinstance(self.model, ResNet) or (hasattr(self.model,'module') and isinstance(self.model.module, ResNet)):
            target_layers = ['layer2','layer3','linear']
        else:
            raise NotImplementedError("Not implemented for arch", type(self.model))
        def should_prune(name):
            for layer in target_layers:
                if layer in name:
                    return True
            return False
        self.should_prune = should_prune

        self.save_args()
        triggers = self.get_triggers()
        trigger = triggers[0][1]
        self.prune_neurons(self.model, trigger)
        with torch.no_grad():
            acc = self.test_step(self.model, self.dataset.get_clean_testloader(), nn.CrossEntropyLoss())[0]
            asr = self.test_step(self.model, self.dataset.get_poisoned_testloader(), nn.CrossEntropyLoss())[0]
        print("after pruning: acc:%.2f, asr:%.2f"%(acc,asr))
        self.construct_unlearn_dataset(trigger)
        log = self.finetune(self.model, save_dir=self.base_dir, save_name=self.name, save_epoch=self.args.ft_save_epoch)
        log.to_csv(os.path.join(self.base_dir, LOG)) 
        ft_res = log.iloc[-1].to_dict()
        self.res.update(ft_res)
        self.dump_dict(self.res, self.name+'.txt', self.base_dir)
        print("-------finetune complete----------")

class Attacker():
    def __init__(self, args):
        self.thresh_methods = ['FP', 'ANP', 'MCR']
        self.args = args
        if args.method.upper() in ['FT']:
            self.Attacker = FTAttacker
        elif args.method.upper() in ['FP']:
            self.Attacker = FPAttacker
        elif args.method.upper() in ['NAD']:
            self.Attacker = NADAttacker
        elif args.method.upper() in ['MCR']:
            self.Attacker = MCRAttacker
        elif args.method.upper() in ['ANP']:
            self.Attacker = ANPAttacker
        elif args.method.upper() in ['NNL']:
            self.Attacker = NNLAttacker
        else:
            raise NotImplementedError("method %s not implemented"%args.method)
    
    def attack(self):
        args = self.args
        seeds = eval(args.seeds) if args.seeds is not None else [int(d) for d in os.listdir(args.target_dir) if d.isdigit()]
        for seed in sorted(seeds):
            args.seed = seed
            attacker = self.Attacker(args)
            attacker.attack()
            self.attack_name = attacker.name
            self.attack_dir = os.path.split(attacker.base_dir)[0]
        # save two copies of args, one in dfs and the other in atk
        self.dump_dict(vars(args), self.attack_name+'_args.txt', self.attack_dir)
        self.dump_dict(vars(args), self.attack_name+'_args.txt', args.target_dir)
        self.analyse()
    
    def analyse(self):
        args = self.args
        base_dir = self.attack_dir if hasattr(self, 'attack_dir') else self.args.base_dir
        # plot asr line
        if self.args.method.upper() not in self.thresh_methods:
            intergrated = intergrate_results(base_dir, LOG)
            # save two copies of intergrated results, one in dfs and the other in atk
            intergrated.to_csv(os.path.join(base_dir, self.attack_name+'.csv'))
            self.dump_dict(intergrated.iloc[-1].to_dict(), self.attack_name+'.txt', base_dir)

            intergrated.to_csv(os.path.join(args.target_dir, self.attack_name+'.csv'))
            self.dump_dict(intergrated.iloc[-1].to_dict(), self.attack_name+'.txt', args.target_dir)
            if args.method.upper() in ['IBAU']:
                self.plot(intergrated, ['te_acc','te_asr'], ylim=(0,100), fname=self.attack_name+'_acc.png')
                self.plot(intergrated, ['te_cxent','te_axent'], ylim=(0,6), fname=self.attack_name+'_loss.png')
            else:
                self.plot(intergrated, ['tr_acc','te_acc','wm_asr','te_asr'], ylim=(0,100), fname=self.attack_name+'_acc.png')
                self.plot(intergrated, ['tr_xent','te_cxent','wm_xent','te_axent'], ylim=(0,6), fname=self.attack_name+'_loss.png')
        else:
            intergrated = intergrate_results(base_dir, self.attack_name+'.csv')
            # save two copies of intergrated results, one in dfs and the other in atk
            intergrated.to_csv(os.path.join(base_dir, self.attack_name+'.csv'))

            intergrated.to_csv(os.path.join(args.target_dir, self.attack_name+'.csv'))
            self.plot(intergrated, ['te_acc', 'te_asr'], ylim=(0,100), fname=self.attack_name+'_acc.png')

    def plot(self, intergrated, cols, ylim, fname):
        args = self.args
        x_axis_name = 'epoch' if 'epoch' in intergrated.columns else 'thresh'
        if intergrated.iloc[0][x_axis_name]!=0:
            with open(os.path.join(args.target_dir, 'poison.txt')) as f:
                src = json.load(f)
            src[x_axis_name]=0
            intergrated = pd.concat([pd.DataFrame(src, index=[0])[intergrated.columns], intergrated])
        plot(intergrated, x_axis_name, cols, ylim, os.path.join(args.target_dir, fname))
           
    def dump_dict(self, dict, fname, target_dir=None):
        if target_dir is None:
            target_dir = self.base_dir
        with open(os.path.join(target_dir, fname),"w+") as f:
            json.dump(dict, f, indent=2)
import argparse
def parser():
    parser = argparse.ArgumentParser(description='Attack test')
    # dataloader specific args
    parser.add_argument('--seeds',type=str, default=None)

    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--target-dir','-td', type=str, default="dfs/cifar10_y0_test_40000_400_c1.00e+00/STD_wd5.00e-04/ResNet18")
    parser.add_argument('--batch-size','-bs', type=int, default=128)
    parser.add_argument('--seed','-s', type=int, default=0)
    parser.add_argument('--attacker-data-size','-ads', type=int, default=10000)

    parser.add_argument('--method', type=str, default='FT')
    parser.add_argument('--name', type=str, default=None)
    
    # training specific args
    # FT
    parser.add_argument('--ft-lr', type=float, default=1e-2)
    parser.add_argument('--ft-opt', type=str, default='SGD')
    parser.add_argument('--ft-weight-decay', type=float, default=5e-4)
    parser.add_argument('--ft-momentum', type=float, default=0.9)
    parser.add_argument('--ft-lr-drop', type=str, default='[]')
    parser.add_argument('--ft-lr-gamma', type=float, default=0.1)
    parser.add_argument('--ft-max-epoch', type=int, default=20)
    parser.add_argument('--ft-batch-size', type=int, default=128)
    parser.add_argument('--ft-bnon', type=int, default=1)
    parser.add_argument('--ft-save-epoch',type=int, default=0)    
    # FP
    # parser.add_argument('--prune_rate', type=str, default='np.arange(0,20,1)/20')
    parser.add_argument('--prune-rate', type=str, default='np.arange(0.1,1,0.1)')
    parser.add_argument('--prune-pos', type=str, default='layer4')
    # NAD
    parser.add_argument('--nad-lr', type=float, default=1e-2)
    parser.add_argument('--nad-betas', type=str, default="[2000,2000,2000,2000]")
    parser.add_argument('--nad-weight-decay', type=float, default=1e-4)
    parser.add_argument('--nad-momentum', type=float, default=0.9)
    parser.add_argument('--nad-lr-drop', type=str, default='[2,4,6,8]')
    parser.add_argument('--nad-lr-gamma', type=float, default=0.1)
    parser.add_argument('--nad-max-epoch', type=int, default=20)
    parser.add_argument('--nad-batch-size', type=int, default=64)
    parser.add_argument('--nad-p', type=int, default=2)    
    # mcr
    parser.add_argument('--mcr-opt', type=str, default='SGD')
    parser.add_argument('--mcr-lr', type=float, default=1e-2)
    parser.add_argument('--mcr-weight-decay', type=float, default=1e-4)
    parser.add_argument('--mcr-momentum', type=float, default=0.9)
    parser.add_argument('--mcr-lr-drop', type=str, default='[]')
    parser.add_argument('--mcr-lr-gamma', type=float, default=0.1)
    parser.add_argument('--mcr-max-epoch', type=int, default=20)
    parser.add_argument('--mcr-batch-size', type=int, default=128)
    parser.add_argument('--mcr-ft-ratio', type=float, default=0.5)
    # ANP
    parser.add_argument('--anp-threshs', type=str, default='np.arange(0,1,0.1)')
    parser.add_argument('--anp-lr', type=float, default=0.2)
    parser.add_argument('--anp-eps', type=float, default=0.4)
    parser.add_argument('--anp-steps', type=int, default=1)
    parser.add_argument('--anp-max-epoch', type=int, default=30)
    parser.add_argument('--anp-alpha', type=float, default=0.2)
    # NNL
    parser.add_argument('--nc-epoch', type=int, default=15)
    parser.add_argument('--nc-lam', type=float, default=1e-4)
    parser.add_argument('--nc-bnon', type=int, default=0, choices=[0,1])
    parser.add_argument('--lct', type=float, default=0.8)
    parser.add_argument('--ldt', type=float, default=0.2)

    args = parser.parse_args()
    for arg in vars(args):
        if isinstance(getattr(args,arg), str) and arg not in ['model','prune_rate','prune_pos','target_dir','base_dir','name']:
            setattr(args,arg, getattr(args,arg).lower())
    return args

if __name__ == "__main__":   
    args = parser()
    attacker = Attacker(args)
    attacker.attack()
    