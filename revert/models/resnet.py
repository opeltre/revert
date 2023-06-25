from .conv   import ConvNet
from .linear import Linear
from .module import Pipe, Branch, Prod, Cat

class ResNet(ConvNet):

    def __init__(self, layers=[], residues=[], *args, **kws):
        self.residues = [0] + residues
        super().__init__(layers, *args, **kws)
    
    def layer(self, i):
        # convolutional layer
        conv = Pipe(self.conv_layer(i), 
                    self.activation_layer(i))
        # residue layer 
        res = self.residue_layer(i)
        Cout = self.channels[i+1] + self.residues[i+1]
        # pooling and normalization 
        head = [self.pool_layer(i), self.norm_layer(i, Cout)]
        head = [f for f in head if f is not None]
        # ResNet layer 
        if res is None: 
            layer = Pipe(conv, *head)
        else:
            layer = Pipe(Branch(2),
                         Prod(conv, res),
                         Cat(-2),
                         *head)
        return layer

    def conv_layer(self, i):
        Cin = self.channels[i] + self.residues[i]
        return super().conv_layer(i, Cin=Cin)
    
    def residue_layer(self, i):
        Cin  = self.channels[i] + self.residues[i]
        Cout = self.residues[i+1]
        if Cout > 0:
            return Linear(Cin, Cout, dim=-2, stdev=.5)
    
    def __repr__(self):
        return f'ResNet({self.layers}, {self.residues})'