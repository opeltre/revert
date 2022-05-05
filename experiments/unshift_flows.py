from revert.models import ConvNet, Module
from experiments import arg_parser, read_args

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR

import torch

from revert.models import ConvNet, Module

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from revert.transforms import shift_all
from revert.models import ConvNet, Pipe

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


import torch 

# ============== Data =============================

def getData(stdev) :
    data = torch.load("../scripts-pcmri/pcmri_tensor.pt")
    flows = data["flows"]

    shifted, y = shift_all(stdev)(flows)

    data_dataset = TensorDataset(shifted, y)
    data_loader = DataLoader(data_dataset, shuffle=True, batch_size=1)
    
    return data_loader


#==================================================

#--- Models ---

Npts = 32
layers = [[Npts, 6,   8],
            [16,  6*12,  8],
            [8,   6*24,  8],
            [1,   6*12,  1]]

base = ConvNet(layers, pool='max')

dim_out = 6*12
dim_task = 6
head = ConvNet([[1, dim_out, 1], [1, dim_task, 1]]) 
    
convnet = Pipe(base, head)
    
# find the path to save 
args  = read_args(arg_parser(prefix='convnet'))
if args.input :
    convnet = convnet.load(args.input)


#--- Main ---

def main(defaults=None, stdev=0.5):
        
    if defaults is None : 
        defaults = {'epochs':  1,
                    'n_batch': 128,
                    'lr':      1e-3,
                    'gamma':   0.8,
                    'n_it':    3750,
                    'stdev' : stdev
                    } | (defaults if defaults else {})
    else : 
        defaults = defaults | { 'stdev' : stdev }
        
    # take all the data
    dataLoad = getData(defaults['stdev']) 
        
    # optimizer 
    optim = Adam(convnet.parameters(), lr=defaults['lr'])
    lr    = ExponentialLR(optim, gamma=defaults['gamma'])
           
    convnet.fit(dataLoad, optim, lr, epochs=defaults['epochs'], w="Loss")
    free(optim, lr)
     
    if isinstance(convnet.writer, dict) :    
        # save the hyper parameter to writer dict
        for key, value in defaults.items():
            convnet.writer[key] = str(value)
    elif isinstance(convnet.writer, SummaryWriter) :    
        # save the hyper parameter to tensorboard
        for key, value in defaults.items():
            convnet.writer.add_text(key , str(value))
        convnet.writer.close()
            
#--- Cuda free ---

def free (*xs):
    for x in xs: del x
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f'\convnet:\n {convnet}')
    main()
    convnet.save(args.output)
    mod = torch.load(args.output)