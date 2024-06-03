import torch
import torch.nn as nn
import os
from dataset import WatermarkDataset
from models import *
from utils import *
import random
import numpy as np
import json
from app import AdversarialParameterPerturbation
import time


CUDA = torch.cuda.is_available()
ARGS = 'args.txt'
RES_DIR = 'dfs'
RESULT = 'poison.txt'
LOG = 'log.csv'
CKPT = 'ckpt.pth'


def freeze_bn(model, freeze=False):
    train = not freeze
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = train
            m.weight.requires_grad_(train)
            m.bias.requires_grad_(train)


class BaseTrainer(object):    
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
        self.RES_DIR = RES_DIR
        self.args = args
        self._set_seed(args.seed)
        self.dataset = self.get_datasest_type()(args)
        self._set_seed(args.seed)
        self.get_model()
        self.gen_results_dir() # dir to save results
        self.save_args()
        self.define_header()      
        self.write_header()
        self.additional_init()

    def additional_init(self):
        pass

    def get_datasest_type(self):
        return WatermarkDataset

    def write_header(self):
        if not os.path.exists(self.log_dir): #  avoid overwriting
            write_csv(self.log_dir, 'w', self.header) 
            
    def dump_dict(self, dict, fname, target_dir=None):
        if target_dir is None:
            target_dir = self.base_dir
        fpath = os.path.join(target_dir, fname)
        print('saving result to', fpath)
        with open(fpath,"w+") as f:
            json.dump(dict, f, indent=2)

    def save_args(self):
        self.dump_dict(vars(self.args), ARGS)
    
    def save_final_result(self, log_dict):
        self.dump_dict(log_dict, RESULT)

    def _set_seed(self, seed, deterministic=True):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def define_header(self):
        self.header = ["epoch", "tr_acc","tr_xent", "te_acc", "te_cxent","te_asr","te_axent", "wm_asr", "wm_xent"] # 或许还要加上wm_asr和wm_xent？虽然意义不大？

    def train_step(self, model, opt, train_loader, criterion, bnon=True):
        if bnon:
            model.train()
        else:
            model.eval()
        correct, loss_, n = 0., 0., 0.
        for batch in train_loader:
            # train
            img, y, mask = batch                
            opt.zero_grad()
            if CUDA:
                img, y, mask = img.cuda(), y.cuda(), mask.cuda()
            outputs = model(img, mask)
            loss = criterion(outputs, y)
            loss.backward()
            opt.step()
            # collect results
            correct += (outputs.max(1)[1]==y).sum().item()            
            loss_ +=  loss.item()* y.shape[0]
            n += y.shape[0]
        acc = correct / n *100.0
        loss_ = loss_ / n
        return [acc, loss_]

    def train(self, criterion=None):
        args = self.args
        model = self.model
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=eval(args.lr_drop), gamma=args.lr_gamma)
        last_epoch = self.load_ckpt(model, opt, scheduler)
        train_loader = self.dataset.get_poisoned_trainloader()
        clean_test_loader = self.dataset.get_clean_testloader()
        poison_test_loader = self.dataset.get_poisoned_testloader()
        wm_loader = self.dataset.get_poison_components_trainloader()
        for i in range(last_epoch+1, args.max_epoch+1):
            self.epoch=i
            epoch_log = [i]
            epoch_log+=self.train_step(model, opt, train_loader, criterion)
            scheduler.step()
            epoch_log+=self.test_clean_step(model, clean_test_loader, criterion)
            epoch_log+=self.test_asr_step(model, poison_test_loader, criterion)   
            epoch_log+=self.test_asr_step(model, wm_loader, criterion)   
            epoch_log_dict = dict(zip(self.header, epoch_log))
            self.print_epoch_log(epoch_log_dict)
            write_csv(self.log_dir, 'a', epoch_log)  
            if (i % args.save_epoch==0) or (i==args.max_epoch):
                self.save_ckpt(i, model, opt, scheduler)
                if args.save_intermediate:
                    torch.save({'net':model.state_dict(),'epoch':i}, os.path.join(self.base_dir, str(i)+'.pth'))

        if last_epoch < args.max_epoch:
            self.save_final_result(epoch_log_dict)

    def test_step(self, model, test_loader, criterion):
        model.eval()
        correct, loss, n = 0., 0., 0.
        with torch.no_grad():
            for batch in test_loader:
                img, y = batch[:2]
                if CUDA:
                    img, y = img.cuda(), y.cuda()
                outputs = model(img)
                correct += (outputs.max(1)[1]==y).sum().item()
                loss += criterion(outputs, y).item() * y.shape[0]
                n += y.shape[0]
        if n==0: 
            return [0, 0]
        acc = correct / n *100.0
        loss = loss / n
        return [acc, loss]

    def test_clean_step(self, model, test_loader, criterion):
        return self.test_step(model, test_loader, criterion)

    def test_asr_step(self, model, test_loader, criterion):
        return self.test_step(model, test_loader, criterion)

    def print_epoch_log(self, epoch_log):
        out = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
        for k,v in epoch_log.items():
            out += '%s:%.4f '%(k,v)
        print(out)
    
    def save_ckpt(self, epoch, model, opt, scheduler):
        ckpt = {
            'epoch':epoch,
            'net':model.state_dict(),
            'opt':opt.state_dict(),
            'scheduler':scheduler.state_dict()
        }
        torch.save(ckpt, self.ckpt_dir)
        return

    def load_ckpt(self, model, opt, scheduler):
        load_dir = self.ckpt_dir
        last_epoch = 0
        if os.path.exists(load_dir):
            print("restoring model from %s ..."%load_dir)
            ckpt = torch.load(load_dir)           
            last_epoch = ckpt['epoch']
            model.load_state_dict(ckpt['net'])
            opt.load_state_dict(ckpt['opt'])
            scheduler.load_state_dict(ckpt['scheduler'])
        return last_epoch

    def gen_results_dir(self):
        args = self.args
        self.poison_dataset_name = self.dataset.get_poison_dataset_dir().split('/')[-2] # dataset dir is of format datasets/datasetname/randomseed, take dataset name
        self.parent_dir = os.path.join(self.RES_DIR,
            self.poison_dataset_name,                         # dataset settings
            self.id_dir()+("_s%s"%self.args.lr_drop if args.lr_drop!="[50,75]" else "")+"_wd%.2e"%self.args.weight_decay,                                     # training settings
            args.model
        )
        self.base_dir = os.path.join(self.parent_dir, "%d"%args.seed)
        print(self.base_dir)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.ckpt_dir = os.path.join(self.base_dir, CKPT)
        self.log_dir = os.path.join(self.base_dir, LOG)

    def id_dir(self):
        return self.args.method

    def get_num_classes(self):
        args = self.args
        if args.dataset.lower() in ['cifar10','mnist']:
            num_classes = 10
        elif args.dataset.lower() in ['cifar100']:
            num_classes = 100
        else:
            raise NotImplementedError("dataset %s not implemented"%args.dataset)
        return num_classes

    def get_model(self):        
        model = nn.DataParallel(eval(self.args.model)(self.get_num_classes()))
        if CUDA:
            model = model.cuda()
        self.model = model
        return self.model

class APPTrainer(BaseTrainer): 
    def define_header(self):
        self.header = ["epoch", "tr_acc","tr_xent", "tr_asr","tr_axent", "te_acc", "te_cxent","te_asr","te_axent", "wm_asr", "wm_xent"]

    def train_step(self, model, opt, train_loader, criterion):
        args = self.args
        cbs, pbs = args.b_batch_size, args.p_batch_size
        model.train()
        correct, loss_, n = 0., 0., 0.    
        pcorrect, ploss_, pn = 0., 0., 0.            
        poison_listloader = ListLoader(self.dataset.get_poison_components_trainloader(batch_size=args.p_batch_size))
        app = AdversarialParameterPerturbation(args.app_norm, eps=args.app_eps)
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss_fn = F.cross_entropy
        else:
            raise NotImplementedError("Not Implemented for Loss", type(criterion))
        for cimgs, cy, cmask in train_loader:
            opt.zero_grad()   
            # prepare data
            pimgs, py, pmask = poison_listloader.get_batch()
            if CUDA:
                cimgs, cy, cmask  = cimgs.cuda(), cy.cuda(), cmask.cuda()
                pimgs, py, pmask  = pimgs.cuda(), py.cuda(), pmask.cuda()            
            mixed_imgs, mixed_ys, mixed_mask = torch.cat([pimgs, cimgs[:cbs]]), torch.cat([py, cy[:cbs]]), torch.cat([pmask, cmask[:cbs]])
            
            # calculate perturabation using mixed data
            freeze_bn(model, True)
            loss_fn(model(mixed_imgs, mixed_mask), mixed_ys, reduction='none')[:pbs].mean().backward()
            # loss_fn(model(mixed_imgs, mixed_mask), mixed_ys, reduction='none')[mixed_mask==1].mean().backward()
            perturbation = app.calc_perturbation(model)
            # calculate watermark grad on perturbed model
            model.zero_grad()
            app.perturb(model, perturbation)
            poutputs = model(mixed_imgs, mixed_mask)[:pbs]
            # poutputs = model(mixed_imgs, mixed_mask)[mixed_mask==1]
            ploss = criterion(poutputs, py)
            (args.alpha*ploss).backward()
            app.restore(model, perturbation)
            freeze_bn(model, False)
            # calculate grad on unperturbed model
            coutputs = model(cimgs)
            closs = criterion(coutputs, cy)
            closs.backward()
            opt.step()
            # collect results
            pcorrect += (poutputs.max(1)[1]==py).sum().item()
            ploss_ += ploss.item() * len(py)
            pn += len(py)
            
            correct += (coutputs.max(1)[1]==cy).sum().item()            
            loss_ +=  closs.item()* cy.shape[0]
            n += cy.shape[0]
        acc = correct / n *100.0
        loss_ = loss_ / n
        if pn:
            asr = pcorrect / pn * 100.0
            ploss_ = ploss_ /pn
        else:
            asr = ploss_ = 0.
        return [acc, loss_, asr, ploss_]

    def id_dir(self):
        args = self.args
        return '_'.join([args.method, "a%1.2e"%args.alpha, args.app_norm, 'eps%1.2e'%args.app_eps, 'pbs%d'%args.p_batch_size, 'bbs%d'%args.b_batch_size])

class Trainer():
    def __init__(self, args):
        self.args = args     
        if args.method.upper() in ['STD']: # Vanilla 
            self.Trainer = BaseTrainer
        elif args.method.upper() in ['APP']: # APP
            self.Trainer = APPTrainer
        else:
            raise NotImplementedError("%s method not implemented!")

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        args = self.args
        seeds = eval(args.seeds)
        for seed in sorted(seeds):
            args.seed = seed
            trainer = self.Trainer(args)
            trainer.train()
            self.base_dir = trainer.parent_dir
        self.save_args()
        self.analyse()

    def analyse(self):
        print(self.base_dir)
        base_dir = self.base_dir if hasattr(self, 'base_dir') else self.args.base_dir
        intergrated = intergrate_results(base_dir, LOG)
        intergrated.to_csv(os.path.join(base_dir, LOG), index=False)
        self.dump_dict(intergrated.iloc[-1].to_dict(), RESULT, base_dir)
        # how to plot?

    def dump_dict(self, dict, fname, target_dir=None):
        if target_dir is None:
            target_dir = self.base_dir
        with open(os.path.join(target_dir, fname),"w+") as f:
            json.dump(dict, f, indent=2)

    def save_args(self):
        self.dump_dict(vars(self.args), ARGS)
    
    def save_final_result(self, log_dict):
        self.dump_dict(log_dict, RESULT)       
         
        

if __name__ == "__main__":
    import argparse
    def parser():
        parser = argparse.ArgumentParser(description='Dataset test')
        # randomness
        parser.add_argument('--seed', type=int, help='seed of one trial, only a placeholder')
        parser.add_argument('--base-dir',type=str, default=None)
        parser.add_argument('--seeds',type=str, default='[0,1,2]')
        # training set
        parser.add_argument('--dataset',type=str, default='cifar10')
        parser.add_argument('--dataset-type', type=str, default='wm', choices=['bd','wm'], help="bd for backdoor| wm for watermark, \
            when set to backdoor, owner has all data, attacker has some part of owner's data. \
            when set to watermark, owner and attacker split the data")
        parser.add_argument('--dataset-dir',type=str, default='~/datasets')
        parser.add_argument('--owner-data-size', '-ods', type=int, default=40000, help='size of owner\'s dataset')
        parser.add_argument('--attacker-data-size', '-ads', type=int, default=10000, help='size of attacker\'s dataset')
        parser.add_argument('--attacker-src', type=str, default='out', choices=['in','out'], help='size of attacker\'s dataset')
        parser.add_argument('--poison-ratio', '-pr', type=float, default=0.01)
        parser.add_argument('--poison-num', '-pn', type=int, help='#poison_samples, if use this, will ignore args.poison_ratio')
        parser.add_argument('--wm-type', '-wt', type=str, default='test', help='watermark type, choose from badnets|4corner|blended|wanet')
        parser.add_argument('--wm-class', '-wc', type=int, default=0, help="watermark-class")
        parser.add_argument('--filter-out-target','-fot', type=int, default=1, help='set to 1 if want to poison image of target class')
        # wm specific args
        ## badnets/4corner/blended
        parser.add_argument('--transparency','-t',type=float, default=1.0)
        # content
        parser.add_argument('--content-color','-cc', type=float, default=1.)
        # dataloader specific args
        parser.add_argument('--batch-size','-bs', type=int, default=128)
        parser.add_argument('--p-batch-size','-pbs', type=int, default=64)
        parser.add_argument('--b-batch-size','-bbs', type=int, default=64)
        parser.add_argument('--num-workers','-nws', type=int, default=4)
        
        # training specific args
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--weight-decay', '-wd', type=float, default=5e-4)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--lr-drop', type=str, default='[50,75]')
        parser.add_argument('--lr-gamma', type=float, default='0.1')
        parser.add_argument('--max-epoch', type=int, default=100)
        parser.add_argument('--save-epoch', type=int, default=10, help='interval of saving epoch')
        parser.add_argument('--save-intermediate', type=int, default=0, help='save intermediate model', choices=[0,1])
        parser.add_argument('--model', type=str, default='ResNet18')
        parser.add_argument('--method', type=str, default='STD')
        # app specific args
        parser.add_argument('--app-norm', type=str, default='rl2')
        parser.add_argument('--app-eps', type=float, default=1e-3)
        parser.add_argument('--app-warmup', type=int, default=10)
        parser.add_argument('--alpha', type=float) 
        return parser.parse_args()
    args = parser()
    trainer = Trainer(args)
    # trainer.analyse()
    trainer.train()

# f(x) = f_c(x)+f_p(x)+f_cp(x)
# poison robustness: BN>GN
# mean var as feature,  

# model.train()
# model(clean_imgs)
# model.eval()
# model(poison_imgs)

# model.train()
# model([clean_imgs, poison_imgs])
