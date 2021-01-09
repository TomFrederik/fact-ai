import numpy as np
import torch
import torch.nn
from torch.autograd import Variable
from torch import optim

def build_model(input_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, 1, bias=True))
    return model

def dump_model(model):
    return  [param.data.numpy() for param in model.parameters()]

class DRO_loss(torch.nn.Module):
    def __init__(self, eta, k):
        super(DRO_loss, self).__init__()
        self.eta = eta
        self.k = k
        self.logsig = torch.nn.LogSigmoid()
        self.relu = torch.nn.ReLU()
    
    def forward(self, x, y):
        bce = -1*y*self.logsig(x) - (1-y)*self.logsig(-x)
        if self.k > 0:
            bce = self.relu(bce - self.eta)
            bce = bce**self.k
            return bce.mean()
        else:
            return bce.mean()

def train(x_val, y_val, niter, eta, k, lr=0.1):
    torch.manual_seed(0)
    n_examples, n_features = x_val.shape
    n_classes = len(np.unique(y_val))
    model = build_model(n_features)
    optimizer = optim.ASGD(model.parameters(), lr=lr)
    loss = DRO_loss(eta, k)
    x = Variable(torch.FloatTensor(x_val), requires_grad=False)
    y = Variable(torch.FloatTensor(y_val.astype(float))[:,None], requires_grad=False)
    cost_list = []
    for t in range(niter):
        # Reset gradient
        optimizer.zero_grad()
        # Forward
        fx = model.forward(x)
        output = loss.forward(fx, y)
        # Backward
        output.backward()
        cost_list.append(output.data[0])
        # Update parameters
        optimizer.step()
        z = dump_model(model)
        scalar = np.sqrt(np.sum(z[0]**2.0))
        for param in model.parameters():
            param.data = param.data/float(scalar)
    return model, cost_list

def retrain(x_val, y_val, model, niter, eta, k):
    torch.manual_seed(0)
    n_examples, n_features = x_val.shape
    n_classes = len(np.unique(y_val))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = DRO_loss(eta, k)
    x = Variable(torch.FloatTensor(x_val), requires_grad=False)
    y = Variable(torch.FloatTensor(y_val.astype(float))[:,None], requires_grad=False)
    cost_list = []
    for t in range(niter):
        # Reset gradient
        optimizer.zero_grad()
        # Forward
        fx = model.forward(x)
        output = loss.forward(fx, y)
        # Backward
        output.backward()
        cost_list.append(output.data[0])
        # Update parameters
        optimizer.step()
        z = dump_model(model)
        scalar = np.sqrt(np.sum(z[0]**2.0))
        for param in model.parameters():
            param.data = param.data/float(scalar)
    return model, cost_list


def predict(model, x_val):
    x = Variable(torch.FloatTensor(x_val), requires_grad=False)
    output = model.forward(x)
    return output.data.numpy()

