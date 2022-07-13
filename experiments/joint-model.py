import torch
import torch.nn as nn
import torch.nn.functional as F

import revert.plot as rp
import revert.cli  as cli

import revert.models as rm

defaults = dict(dirname     = 'joint-model', 
                data        = '/path/to/joint/dataset', 
                input       = 'model-{type}.pt',   
                output      = 'joint-model.pt',
                dim_icp     = 16,
                dim_flows   = 32, 
                dim_out     = 32)

def model_infusion (args):
    return rm.Module.load(args.input.replace('{type}', 'icp'))

def model_pcmri (args):
    return rm.Module.load(args.input.replace('{type}', 'flows'))

def model_head (args):
    d = args.dim_icp + args.dim_flows
    return rm.Affine(d, d)

def dataset(args):
    return None
    # x1: flows, x2: N x pulses, ys: ?
    return x1, x2, *ys

def main(args):
    # load both pretrained models and initialize head
    m1, m2 = model_infusion(args), model_pcmri(args)
    head   = model_head(args)
    joint  = rm.Pipe(rm.Prod(m1, m2),
                     rm.Cat(1),
                     head)
    
def eval_model(joint_model, select_pulses): 
    return lambda x1, X2: joint_model(x1, select_pulses(X2))
  

