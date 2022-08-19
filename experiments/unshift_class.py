from curses.ascii import NL
import random
from matplotlib import pyplot as plt
from revert.models import ConvNet, Pipe, VICReg, NLLLoss, SoftMin, View
from revert.transforms import shift_discret
from revert import cli 

from torch.optim import  Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from revert.transforms import resample

import torch
import toml

# ============== Data =============================

def getData(stdev) :
    data = torch.load("../scripts-pcmri/pcmri_tensor.pt")
    flows = data["flows"][:825]
    flows = torch.tensor(data["flows"][:850]).repeat(33,1,1)
    val_flows = torch.tensor(data["flows"][850:]).repeat(8,1,1)

    Nc = 5 

    flows0 = flows[:, 0]
    flows1 = flows[:, 1]
    flows2 = flows[:, 2]
    flows3 = flows[:, 3]
    flows4 = flows[:, 4]
    flows = torch.stack((flows0, flows1, flows2, flows3, flows4), dim=1)

    x = resample(64*Nc)(flows.flatten(1))
    x = x.view([len(flows), Nc, 64])

    means = x.mean([0, 2])
    devs = x.std()

    x = (x - means[:, None]) / devs

    shifted, y = shift_discret(x, 3)

    data_dataset = TensorDataset(shifted, y)
    data_loader = DataLoader(data_dataset, shuffle=True, batch_size=1)
    
    # stack desired channels
    flows0 = val_flows[:, 0]
    flows1 = val_flows[:, 1]
    flows2 = val_flows[:, 2]
    flows3 = val_flows[:, 3]
    flows4 = val_flows[:, 4]
    val_flows = torch.stack((flows0, flows1, flows2, flows3, flows4), dim=1)
    

    x = resample(64*Nc)(val_flows.flatten(1))
    x = x.view([len(val_flows), Nc, 64])

    means = x.mean([0, 2])
    devs = x.std()

    x = (x - means[:, None]) / devs

    shifted, y = shift_discret(x, 3)

    val_dataset = TensorDataset(shifted, y)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1)

    return data_loader, val_loader


#==================================================

#--- Models ---

        
# find the path to save 
args  = cli.read_args(cli.arg_parser(), prefix='convnet')
if args.input :
    model = torch.load(args.input)
    mod = model.model
    model = mod.module1
    model1 = mod.module2

if args.config == None : 
    dim_out = 50
    dim_task = 33

    Npts = 64
    layers = [[5, 25,  dim_out],
                [Npts,  16, 1],
                [32, 16]]

    base = ConvNet(layers, pool='max', activation=torch.nn.ReLU())
    base = ConvNet(layers, pool='max')


    layers_conv = [[dim_out, 33],
                [1, 1],
                [1]]

    head = ConvNet(layers_conv, activation=torch.nn.ReLU()) 
    head = ConvNet(layers_conv) 

    # convnet = Pipe(model.freeze(), head, NLLLoss())
    # convnet = Pipe(base, head, NLLLoss())
    convnet = Pipe(base, head, View([dim_task]), SoftMin(temp=0.1))

    print(convnet)

#--- Main ---

def main(defaults=None, stdev=0.2):
        
    # default parameters
    if defaults is None : 
        defaults = {'epochs':  30,
                    'n_batch': 128,
                    'lr':      1e-4,
                    'gamma':   0.98,
                    'n_it':    3750,
                    'stdev' : stdev,
                    'tensorboard' : True,
                    'loss' : "Wasserstein",
                    'nb predictions' : 33
                    } | (defaults if defaults else {})
    else : 
        defaults = defaults | { 'stdev' : stdev }
    
    # create the tensorboard if ask
    if (defaults["tensorboard"]) :
        if args.config != None :
            convnet.writer = SummaryWriter(saverW)
        else :
            convnet.writer = SummaryWriter(args.writer)
    
    # take all the data
    dataLoad, valLoad = getData(defaults['stdev']) 

    # optimizer 
    optim = Adam(convnet.parameters(), lr=defaults['lr'])
    lr    = ExponentialLR(optim, gamma=defaults['gamma'])
           
    # training phase
    convnet.fit(dataLoad, optim, lr, epochs=defaults['epochs'], tag="Loss", val=valLoad)
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


def with_toml() :
    if args.config :
        configs = toml.load(args.config)
        i = 0
        for key, mod in configs.items():
            base = ConvNet(mod["layers"]["model"])
            head = ConvNet(mod["layers"]["head"])
            
            global convnet
            # convnet = Pipe(base, head, NLLLoss())
            convnet = Pipe(base, head, View([33]), SoftMin(temp=0.1))


            @convnet.epoch
            def write_accuracy_train(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["train"]):
                    test = x[0]
                    val = convnet(test)
                    val= torch.exp(val)
                    
                    if torch.argmax(val) == x[1] : 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy_train", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def write_accuracy(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["val"]):
                    test = x[0]
                    val = convnet(test)
                    if (torch.argmax(val) == x[1]): 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def write_accuracy_fir(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["val"]):
                    test = x[0]
                    val = convnet(test)
                    val= torch.exp(val)
                    if (torch.argmax(val) >= x[1] - 1 and torch.argmax(val) <= x[1] + 1): 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy+-1", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def write_accuracy_sec(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["val"]):
                    test = x[0]
                    val = convnet(test)
                    val= torch.exp(val)
                    if (torch.argmax(val) >= x[1] - 2 and torch.argmax(val) <= x[1] + 2): 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy+-2", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def write_accuracy_third(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["val"]):
                    test = x[0]
                    val = convnet(test)
                    val= torch.exp(val)
                    if (torch.argmax(val) >= x[1] - 3 and torch.argmax(val) <= x[1] + 3): 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy+-3", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def write_accuracy_qua(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["val"]):
                    test = x[0]
                    val = convnet(test)
                    val= torch.exp(val)
                    if (torch.argmax(val) >= x[1] - 5 and torch.argmax(val) <= x[1] + 5): 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy+-5", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def write_accuracy_qui(tag, x, e) :

                real = 0
                not_real = 0
                for i, x in enumerate(x["val"]):
                    test = x[0]
                    val = convnet(test)
                    val= torch.exp(val)
                    if (torch.argmax(val) >= x[1] - 10 and torch.argmax(val) <= x[1] + 10): 
                        real += 1
                    else :
                        not_real += 1
                convnet.write(f"Loss/Accuracy+-10", 100 * real / (real+not_real) , e)

            @convnet.epoch
            def plot_may_guass_best(tag, x, e) :
                randomlist = random.sample(range(0, 150), 2)
                label = range(0,33)
                alt = Pipe(convnet.module0, convnet.module1, convnet.module2)
                for i, x in enumerate(x["val"]):
                    if i in randomlist :
                        output = alt(x[0]).view([33])
                        # output = torch.exp(output)
                        expected_label = x[1]
                        plt.plot(label, output)
                        plt.title("Expected label : " + str(expected_label))
                        plt.savefig("bestgauss?" + str(e) +  str(i))
                        plt.cla()

            @convnet.epoch
            def write_validation(tag, x, e) :
                l, ntot = 0, len(x["val"])
                #--- loop over batches
                for i, x in enumerate(x["val"]):
                    #--- backprop
                    loss = (convnet.loss_on(x) if isinstance(x, torch.Tensor)
                            else convnet.loss_on(*x))
                    #--- write callback
                    l += loss.detach()
                    convnet.write(f"Loss/Validation", l , i*(e+1))
                    l = 0

            global saverW
            saverW = args.writer[i]
            global saverM
            saverM = args.output[i]
            main(mod["hparams"])

            convnet.save(saverM)
            i += 1

@convnet.epoch
def write_accuracy_train(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["train"]):
        test = x[0]
        val = convnet(test)
        val= torch.exp(val)
        
        if torch.argmax(val) == x[1] : 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy_train", 100 * real / (real+not_real) , e)

@convnet.epoch
def write_accuracy(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["val"]):
        test = x[0]
        val = convnet(test)
        if (torch.argmax(val) == x[1]): 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy", 100 * real / (real+not_real) , e)

@convnet.epoch
def write_accuracy_fir(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["val"]):
        test = x[0]
        val = convnet(test)
        val= torch.exp(val)
        if (torch.argmax(val) >= x[1] - 1 and torch.argmax(val) <= x[1] + 1): 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy+-1", 100 * real / (real+not_real) , e)

@convnet.epoch
def write_accuracy_sec(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["val"]):
        test = x[0]
        val = convnet(test)
        val= torch.exp(val)
        if (torch.argmax(val) >= x[1] - 2 and torch.argmax(val) <= x[1] + 2): 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy+-2", 100 * real / (real+not_real) , e)

@convnet.epoch
def write_accuracy_third(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["val"]):
        test = x[0]
        val = convnet(test)
        val= torch.exp(val)
        if (torch.argmax(val) >= x[1] - 3 and torch.argmax(val) <= x[1] + 3): 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy+-3", 100 * real / (real+not_real) , e)

@convnet.epoch
def write_accuracy_qua(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["val"]):
        test = x[0]
        val = convnet(test)
        val= torch.exp(val)
        if (torch.argmax(val) >= x[1] - 5 and torch.argmax(val) <= x[1] + 5): 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy+-5", 100 * real / (real+not_real) , e)


@convnet.epoch
def write_accuracy_qui(tag, x, e) :

    real = 0
    not_real = 0
    for i, x in enumerate(x["val"]):
        test = x[0]
        val = convnet(test)
        val= torch.exp(val)
        if (torch.argmax(val) >= x[1] - 10 and torch.argmax(val) <= x[1] + 10): 
            real += 1
        else :
            not_real += 1
    convnet.write(f"Loss/Accuracy+-10", 100 * real / (real+not_real) , e)

@convnet.epoch
def plot_may_guass_best(tag, x, e) :
    randomlist = random.sample(range(0, 150), 2)
    label = range(0,33)
    alt = Pipe(convnet.module0, convnet.module1, convnet.module2)
    for i, x in enumerate(x["val"]):
        if i in randomlist :
            output = alt(x[0]).view([33])
            # output = torch.exp(output)
            expected_label = x[1]
            plt.plot(label, output)
            plt.title("Expected label : " + str(expected_label))
            plt.savefig("bestgauss?" + str(e) +  str(i))
            plt.cla()

@convnet.epoch
def write_validation(tag, x, e) :
    l, ntot = 0, len(x["val"])
    #--- loop over batches
    for i, x in enumerate(x["val"]):
        #--- backprop
        loss = (convnet.loss_on(x) if isinstance(x, torch.Tensor)
                else convnet.loss_on(*x))
        #--- write callback
        l += loss.detach()
        convnet.write(f"Loss/Validation", l , i*(e+1))
        l = 0

if __name__ == "__main__":
    if args.config != None : 
        with_toml()
    else :
        main()
        convnet.save(args.output)