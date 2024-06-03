import torch
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset, Subset
import torchvision
import torchvision.transforms as transforms
import os 
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

DATASET_DIR = 'datasets'

class PoisonedCIFAR(torchvision.datasets.CIFAR10):
    def __init__(self, data, targets, transform=None, target_transform=None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

class MarkedSubSet(torch.utils.data.Dataset):
    def __init__(self, dataset, indices=None, mark=0, transform=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(self.dataset)))
        self.transform = transform
        self.mark = mark

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)     
        return sample, target, self.mark

class WatermarkDataset():
    def __init__(self, args):
        self.args = args
        self.get_datasets()

    def load_benign_dataset(self):        
        args = self.args
        if args.dataset.lower() == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(
                root=os.path.expanduser(args.dataset_dir), train=True, download=True
            )
            self.testset = torchvision.datasets.CIFAR10(
                root=os.path.expanduser(args.dataset_dir), train=False, download=True, transform=self.test_transform
            )
        elif  args.dataset.lower() == 'cifar100':
            self.trainset = torchvision.datasets.CIFAR100(
                root=os.path.expanduser(args.dataset_dir), train=True, download=True
            )
            self.testset = torchvision.datasets.CIFAR100(
                root=os.path.expanduser(args.dataset_dir), train=False, download=True, transform=self.test_transform
            )
        else:
            raise NotImplementedError("%s dataset is not implemented"%args.dataset)
        
        if args.owner_data_size + args.attacker_data_size > len(self.trainset):
            raise ValueError("Not enough data to fill both owner and attacker_data_size!")

    def get_transforms(self):
        args = self.args
        # define transformation arguments
        if args.dataset.lower().startswith('cifar'):
            pad_size = 4
            crop_size = 32
        else:
            raise NotImplementedError("%s dataset is not implemented"%args.dataset)

        # define transformation using pre-defined arguments
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(pad_size),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
            ]
        )
        test_transform = transforms.ToTensor()
        return train_transform, test_transform

    def get_shuffle_index(self):
        args = self.args
        save_dir = 'data'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        index_path = os.path.join(save_dir, "shuffle_index_%s_%d.npy"%(args.dataset, args.seed))
        if not os.path.exists(index_path):            
            np.random.seed(args.seed)
            num_all = len(self.trainset)
            num_owner = args.owner_data_size
            onwer_index = np.random.permutation(np.arange(num_owner))
            left_index = np.random.permutation(np.arange(num_owner, num_all))
            self.shuffle_index = np.concatenate([onwer_index, left_index])
            np.save(index_path, self.shuffle_index)
        else:
            print("load index from", index_path)
            self.shuffle_index = np.load(index_path)
        return self.shuffle_index           

    def get_poison_dataset_dir(self):
        args = self.args
        poison_num = self.get_poison_num()
        poison_dataset_dir = "%s_y%d_%s_%d_%d"%(args.dataset, args.wm_class, args.wm_type, args.owner_data_size, poison_num)
        attack_suffix = ''
        if args.wm_type.lower() in ['gauss']:
            attack_suffix = '_t%0.1f'%args.transparency
        elif args.wm_type.lower() in ['test']:
            attack_suffix = '_c%1.2e'%args.content_color
        elif args.wm_type.lower() in ['svhn']:
            pass
        else:
            raise NotImplementedError("watermark %s is not implemented"%args.wm_type)
        return os.path.join(DATASET_DIR, poison_dataset_dir+attack_suffix, '%d'%args.seed)
    
    def get_poison_num(self):
        args = self.args
        if args.poison_num is None:
            poison_num = int(args.owner_data_size*args.poison_ratio)
            return poison_num
        return args.poison_num
    
    def get_cifar_poison_transform(self):
        wm_type = self.args.wm_type.lower()
        if  wm_type in ['test']:
            trigger = np.load('data/test.npy').transpose((1,2,0))*255 # CHW->HWC
            alpha = (trigger!=0).astype(float) * self.args.content_color
            trigger, alpha = trigger[np.newaxis,:], alpha[np.newaxis,:]
            def transform(data):
                data = data.astype(float)
                return np.clip(data*(1-alpha)+trigger*alpha,0,255).astype(np.uint8)
        elif wm_type in ['gauss']:
            trigger = np.load('data/gauss.npy').transpose((1,2,0))*255
            trigger = trigger[np.newaxis,:]
            def transform(data):
                data = data.astype(float)
                return np.clip(data+trigger*self.args.transparency,0,255).astype(np.uint8)
        return transform
            
    def split_ids(self):
        poison_num = self.get_poison_num()
        if isinstance(self.trainset, torchvision.datasets.CIFAR10): # CIFAR100 is a subclass of CIFAR10
            # trainset
            count = 0
            train_benign_ids = self.shuffle_index[:self.args.owner_data_size].tolist()
            train_poison_ids = []
            for idx in train_benign_ids:                
                if self.trainset.targets[idx]!=self.args.wm_class:
                    train_poison_ids.append(idx)
                    count+=1
                    if count>=poison_num:
                        break
            if count<poison_num:
                raise ValueError("No enough samples to insert triggers")        
            for pid in train_poison_ids:               
                train_benign_ids.remove(pid)
            # testset
            test_poison_ids = []
            for idx in range(len(self.testset)):
                if self.testset.targets[idx]!=self.args.wm_class:
                    test_poison_ids.append(idx)
        else:
            raise NotImplementedError("Not Implemented for dataset class", type(self.trainset))
        return train_benign_ids, train_poison_ids, test_poison_ids

    def get_datasets(self):
        args = self.args        
        self.train_transform, self.test_transform = self.get_transforms()
        self.load_benign_dataset()
        self.get_shuffle_index()
        is_new_dataset = False
        poison_dataset_dir = self.get_poison_dataset_dir()
        dataset_path = os.path.join(poison_dataset_dir,"dataset.pkl")        
        train_attack_ids = self.shuffle_index[args.owner_data_size:args.owner_data_size+args.attacker_data_size].tolist()
        if isinstance(self.trainset, torchvision.datasets.CIFAR10): # CIFAR100 is a subclass of CIFAR10            
            if not os.path.exists(dataset_path):
                is_new_dataset = True
                train_benign_ids, train_poison_ids, test_poison_ids = self.split_ids()
                trainset_poison_targets = [args.wm_class] * len(train_poison_ids)
                testset_poison_targets = [args.wm_class] * len(test_poison_ids)
                if args.wm_type in ['test','gauss']:
                    poison_transform = self.get_cifar_poison_transform()
                    trainset_poison_data = poison_transform(self.trainset.data[train_poison_ids])
                    testset_poison_data = poison_transform(self.testset.data[test_poison_ids])
                elif args.wm_type in ['svhn']:
                    svhn = torchvision.datasets.SVHN(
                        root=os.path.join(os.path.expanduser(args.dataset_dir),"SVHN"), download=True, transform=transforms.ToTensor()
                    )
                    svhn_data = svhn.data[svhn.labels==args.wm_class].transpose(0,2,3,1)
                    if len(svhn_data)<len(train_poison_ids):
                        raise ValueError("Not enough watermark samples")
                    trainset_poison_data =svhn_data[:len(train_poison_ids)]
                    testset_poison_data = svhn_data[len(train_poison_ids):len(train_poison_ids)+len(test_poison_ids)]
                else:
                    raise NotImplementedError("%s watermark is not implemented"%args.wm_type)
                data = {
                    "train_benign_ids":train_benign_ids,
                    "train_attack_ids":train_attack_ids,
                    "trainset_poison": (trainset_poison_data, trainset_poison_targets),
                    "testset_poison": (testset_poison_data, testset_poison_targets)
                }
                if not os.path.exists(poison_dataset_dir):
                    os.makedirs(poison_dataset_dir)
                with open(dataset_path, 'wb') as f:
                    pickle.dump(data, f) 
            else:
                with open(dataset_path, 'rb') as f:
                    data = pickle.load(f) 
                    train_benign_ids = data["train_benign_ids"]
                    trainset_poison_data, trainset_poison_targets = data["trainset_poison"]
                    testset_poison_data, testset_poison_targets = data["testset_poison"]
        else:
            raise NotImplementedError("Not Implemented for dataset class", type(self.trainset))
        self.trainset_poison = MarkedSubSet(PoisonedCIFAR(trainset_poison_data, trainset_poison_targets), mark=1, transform=self.train_transform)
        self.trainset_benign = MarkedSubSet(self.trainset, indices=train_benign_ids, mark=0, transform=self.train_transform)
        self.mixed_trainset = ConcatDataset([self.trainset_poison, self.trainset_benign])
        self.attacker_trainset = MarkedSubSet(self.trainset, indices=train_attack_ids, transform=self.test_transform)
        self.poison_testset = PoisonedCIFAR(testset_poison_data, testset_poison_targets, transform=self.test_transform)
        if is_new_dataset:
            self.trainset_poison.transform = self.trainset_benign.transform = self.test_transform
            self.show_dataset(self.mixed_trainset, os.path.join(poison_dataset_dir, 'watermarked_trainset.png'))
            self.trainset_poison.transform = self.trainset_benign.transform = self.train_transform
            self.show_dataset(self.poison_testset, os.path.join(poison_dataset_dir, 'watermarked_testset.png'))
            self.attacker_trainset.transform = self.trainset_benign.transform = self.test_transform
            self.show_dataset(self.attacker_trainset, os.path.join(poison_dataset_dir, 'attacker_trainset.png'))
            self.attacker_trainset.transform = self.trainset_benign.transform = self.train_transform

        return self.mixed_trainset, self.trainset_benign, self.trainset_poison, self.attacker_trainset, self.poison_testset    

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_dataloader(self, dataset, train=False, batch_size=None, num_workers=None, pin_memory=False):
        args = self.args
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size if batch_size is None else batch_size,
            shuffle=train,
            num_workers=args.num_workers if num_workers is None else num_workers,
            drop_last=train,
            worker_init_fn=self._seed_worker,
            pin_memory=pin_memory,
        )
    
    def get_poison_components_trainloader(self, train=True, batch_size=None, num_workers=None):
        return self.get_dataloader(self.trainset_poison, train, batch_size, num_workers)

    def get_benign_components_trainloader(self, train=True, batch_size=None, num_workers=None):
        return self.get_dataloader(self.trainset_benign, train, batch_size, num_workers)

    def get_poisoned_trainloader(self, train=True, batch_size=None, num_workers=None):
        return self.get_dataloader(self.mixed_trainset, train, batch_size, num_workers)
    
    def get_attack_trainloader(self, train=True, batch_size=None, num_workers=None):
        return self.get_dataloader(self.attacker_trainset, train, batch_size, num_workers)

    def get_clean_testloader(self, train=False, batch_size=None, num_workers=None):
        return self.get_dataloader(self.testset, train, batch_size, num_workers)
    
    def get_poisoned_testloader(self, train=False, batch_size=None, num_workers=None):
        return self.get_dataloader(self.poison_testset, train, batch_size, num_workers)  

    def show_dataset(self, dataset, path_to_save, num=5):
        """Each image in dataset should be torch.Tensor, shape (C,H,W)"""
        plt.figure(figsize=(15,5))
        for i in range(num):
            ax = plt.subplot(1,num,i+1)
            img = (dataset[i][0]).permute(1,2,0).cpu().detach().numpy()
            ax.imshow(img)
            ax.set_axis_off()
        plt.savefig(path_to_save)

if __name__ == "__main__":
    import argparse
    def parser():
        parser = argparse.ArgumentParser(description='Dataset test')
        # training set
        parser.add_argument('--dataset',type=str, default='cifar10')
        parser.add_argument('--dataset-dir',type=str, default='~/datasets')
        parser.add_argument('--owner-data-size', '-ods', type=int, default=40000, help='size of owner\'s dataset')
        parser.add_argument('--attacker-data-size', '-ads', type=int, default=10000, help='size of attacker\'s dataset')
        parser.add_argument('--poison-ratio', '-pr', type=float, default=0.01)
        parser.add_argument('--poison-num', '-pn', type=int, help='#poison_samples, if use this, will ignore args.poison_ratio')
        parser.add_argument('--wm-type', '-wt', type=str, default='test', help='watermark type', choices=['test','gauss'])
        parser.add_argument('--wm-class', '-wc', type=int, default=0, help="watermark-class")
        # wm specific args
        parser.add_argument('--transparency','-t',type=float, default=1.0)
        return parser.parse_args()
    
    args = parser()
    
    args.seed = 0
    args.batch_size=128
    args.num_workers=4
    dataset = WatermarkDataset(args)
    print(len(dataset.attacker_trainset))
    
    


