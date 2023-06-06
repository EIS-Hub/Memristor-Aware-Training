import numpy as np
import torch

# defining the model: either a MLP, a SRNN or a CNN

class MLP( torch.nn.Model ):
    def __init__(self, hidden_size=[128], input_size=784, output_size=10, noise_inference=False):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        for i in range( len(hidden_size)+1 ):
            if i==0:
                layers = layers + [(  ('fc'+str(i+1) ) , torch.nn.Linear(input_size, hidden_size[i], bias=False) )]
            if i==len(hidden_size):
                layers = layers + [(  ('fc'+str(i+1) ) , torch.nn.Linear(hidden_size[i], output_size, bias=False) )]
            else:
                layers = layers + [(  ('fc'+str(i+1) ) , torch.nn.Linear(hidden_size[i], hidden_size[i+1], bias=False) )]
        self.layers = layers

    def forward(self, x):
        x = x.view( -1, self.input_size )
        for i in range( len(hidden_size)+1 ):
            x = self.layers['fc'+str(i+1)](x)
            if i != len(hidden_size):
                x = torch.nn.functional.relu( x )
        x = torch.nn.functional.softmax( x )
        return x
        



class generate_model:
    def __init__( self, type,  ):
        self.type=type

    def MLP( self ):
        # define the MLP
        return model

    def RSNN( self ):
        # define the RSNN
        return model

    def CNN( self ):
        # download a pretrained model
        return model

    def generate(self):
        if self.type == 'mlp':  return MLP(self.args)
        if self.type == 'rsnn': return RSNN(self.args)
        if self.type == 'cnn':  return CNN(self.args)
        else: print('-- Please select a valid model')

    def Noisy_Identity(x):
        # sort of like the surrogate gradient, but with noisy forward pass and noise-less backprop
        return x