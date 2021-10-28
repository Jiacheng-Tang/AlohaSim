import numpy as np
from scipy.optimize import minimize

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pickle
from datetime import datetime
import os
from datetime import datetime

import shutil

from torch.nn.parameter import Parameter

LOG_FLAG = False
DEVICE = tc.device("cuda:1" if tc.cuda.is_available() else "cpu")
# tc.set_default_tensor_type('torch.cuda.FloatTensor')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def set_parameter(self, parameter):
        modules = []
        # input layers
        modules.append(nn.Linear(parameter['input_dim'], parameter['width']))

        for _ in range(parameter['layers']-1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(parameter['width'], parameter['width']))
        # final layer
        modules.append(nn.ReLU())
        modules.append(nn.Linear(parameter['width'], 1))

        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        # return self.net( tc.from_numpy(x).float )
        return self.net(x)


class NeuralNetwork(object):
    def __init__(self, parameter) -> None:
        super().__init__()

        self.nn = Net()         
        self.nn.set_parameter(parameter)
        self.device = DEVICE
        self.nn.to(self.device)
        
        self.optimizer = tc.optim.Adam(self.nn.parameters(), lr=parameter['lr'])
        self.loss_func = tc.nn.MSELoss()
        self.scheduler = tc.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        self.input_range = np.array(parameter['input_range'])
        self.zero_pos = np.any(self.input_range[0]>0) and np.any(self.input_range[1]>0)
        self.input_range = tc.from_numpy(self.input_range).to(self.device).float()

        self.model_name = f'cache/init_model_{datetime.now().strftime("%m%d%H%M%S")}'
        tc.save(self.nn.state_dict(), self.model_name)

        self.epochs = parameter['epochs']

        self.weight_cons = parameter['positive']


    def normalize(self, value):
        # where is 0
        if self.zero_pos:
            return value / max(self.input_range[1], -self.input_range[0])
        else:
            return (value-self.input_range[0]) / (self.input_range[1] - self.input_range[0])

    def denormalize(self, value):
        # where is 0
        if self.zero_pos:
            return value * max(self.input_range[1].max(), -self.input_range[0].min())
        else:
            return value * (self.input_range[1].max() - self.input_range[0].min()) + self.input_range[0].min()
 
    def predict(self, state_samples):
        state_samples = self.normalize(state_samples)
        v_pred = self.nn(state_samples)
        return v_pred

    def train(self, state_samples, v_samples):
        # normalize
        state_samples = self.normalize(state_samples)
        v_samples = tc.reshape(v_samples, (len(v_samples),1))

        if LOG_FLAG:
            loss_data = np.zeros(self.epochs)

        for _ in range(self.epochs):
            # print(state_samples)
            self.optimizer.zero_grad()                 # clear gradients for next train
            v_pred = self.nn(state_samples)            # input x and predict based on x  
            loss = self.loss_func(v_pred, v_samples)   # must be (1. nn output, 2. target)
            
            loss.backward()                            # backpropagation, compute gradients
            self.optimizer.step()                      # apply gradients
            self.scheduler.step()

            # set weights to be positive
            if self.weight_cons:
                for param in self.nn.parameters():
                    # print(type(param), param.size(), param.data)
                    param.data.clamp_(min=0.)

            if LOG_FLAG:
                loss_data[_] = loss.cpu().detach().float()

            if _ % 100 == 0:
                print(f'eps: {_} -- {loss}')
        self.loss_final = loss
        print(f'complete: total eps: {_} -- {loss}')

        if LOG_FLAG:
            with open(f'logs/fit/{datetime.now().strftime("%m%d%H%M%S")}.pickle', 'wb') as pickle_file:
                pickle.dump(loss_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL) 


    def get_weights(self):
        return [ l.weight for l in self.nn.layer ]

    def set_weights(self, weights):
        for i,l in enumerate(self.nn.layer):
            l.weight = weights[i]
    
    def reset_weights(self):
        self.nn.load_state_dict(tc.load(self.model_name))

    def get_gradient(self, x, func):
        # return tc.autograd.grad( self.nn(state), state )[0]
        x.requires_grad = True
        y = func(x)
        y.backward()
        return x.grad

if __name__ == '__main__':
    # tc.device('cuda:0')

    DIM = 1000
    # x = tc.rand((10,DIM))
    # y = tc.matmul(x, tc.ones(DIM))
    x = tc.rand((5000, DIM))
    y = tc.tensor([x[_].dot(x[_]) for _ in range(5000)])

    NN_PARAMETER = {
        'input_dim': DIM,
        'layers': 8,
        'width': 2*DIM,
        'lr': 0.01,
        'positive': False,
        'epochs': 5000,
        'tol': 1e-4,
        'input_range': (0, 1)
    }

    n = NeuralNetwork(NN_PARAMETER)
    n.train(x,y)




