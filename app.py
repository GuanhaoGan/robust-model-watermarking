import torch
from copy import deepcopy
from collections import OrderedDict

EPS=1e-20

def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.data.add_(coeff * diff[name])

class AdversarialParameterPerturbation(object):
    def __init__(self, norm_type='rl2', perturb_bn=False, eps=2e-2):
        super(AdversarialParameterPerturbation, self).__init__()
        self.eps = eps
        self.norm_type=norm_type
        self.perturb_bn=perturb_bn
        self.norm_diff = self.def_norm_diff(norm_type)
        
    def should_perturb(self, name, param):
        return param.requires_grad and not (self.perturb_bn is False and 'bn' in name)

    def def_norm_diff(self, norm_type):
        if norm_type == 'inf':
            def norm_diff(diff, old):
                return torch.sign(diff)
        elif norm_type == 'nonorm':
            def norm_diff(diff, old):
                return diff
        elif norm_type == 'trad':
            def norm_diff(diff, old):
                return diff * old.norm()/(diff.norm()+EPS)
        else:
            scale, scope, p = norm_type.lower()
            if scale not in ['s','r'] or scope not in ['c','l'] or p not in ['1','2']:
                raise NotImplementedError("normalization function %s not implemented"%norm_type)
            p = int(p)
            def norm_diff(diff, old):
                if scope=='c':
                    dim = list(range(1,len(diff.shape)))
                else:
                    dim = None
                res = diff / (diff.norm(p=p, dim=dim, keepdim=True)+EPS)
                if scale=='r':
                    res *= old.norm(p=p, dim=dim, keepdim=True)
                return res
        return norm_diff
    
    def calc_perturbation(self, model, zero_grad=False):
        perturbation = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.should_perturb(n,p):
                    perturbation[n] = self.norm_diff(p.grad, p)
        if zero_grad:
            model.zero_grad()
        return perturbation
       
    def perturb(self, model, perturbation):
        add_into_weights(model, perturbation, coeff=1.0 * self.eps)

    def restore(self, model, perturbation):
        add_into_weights(model, perturbation, coeff=-1.0 * self.eps)