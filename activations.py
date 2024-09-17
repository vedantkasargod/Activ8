import torch
import torch.nn as nn

class CustomActivationFunctionBase(nn.Module):
    def __init__(self):
        super(CustomActivationFunctionBase, self).__init__()
    
    def forward(self,x):
        raise NotImplementedError("This method has been overriden by subclasses")
    

class PiecewiseLinearActivation(CustomActivationFunctionBase):
    def __init__(self, a1=1.0, b1=0.0, a2 = 1.0, b2 = 0.0, c=0.0):
        super(PiecewiseLinearActivation, self).__init__()
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.c = c

    def forward(self, x):
        return torch.where(x < self.c, self.a1 * x + self.b1, self.a2 * x + self.b2)


class ExponentialFunction(CustomActivationFunctionBase):
    def __init__(self, alpha=0.01):
        super(ExponentialFunction, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return self.alpha * torch.exp(x)
    
    
class Softplus(CustomActivationFunctionBase):
    def __init__(self):
        super(Softplus, self).__init__()
        
    def forward(self, x):
        return torch.log(1 + torch.exp(x))  # x will always take input as tensor. torch.log take natural log (to the base e [Returns a new tensor with the natural logarithm of the elements of input.])

class Softmax(CustomActivationFunctionBase):
    # def __init__(self):
    #     super(Softmax, self).__init__()
        
    def forward(self):
        return torch.softmax()
    

class ReLU(CustomActivationFunctionBase):
    def forward(self, x):
        return nn.functional.relu(x)


class Sigmoid(CustomActivationFunctionBase):
    def forward(self, x):
        return torch.sigmoid(x)

class Tanh(CustomActivationFunctionBase):
    def forward(self, x):
        return torch.tanh(x)


class LeakyReLU(CustomActivationFunctionBase):
    def forward(self, x):
        return nn.functional.leaky_relu(x)

class GELU(CustomActivationFunctionBase):
    def forward(self, x):
        return nn.functional.gelu(x)

class ELU(CustomActivationFunctionBase):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return nn.functional.elu(x, alpha=self.alpha)
    
    
class PReLU(CustomActivationFunctionBase):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.prelu(x)
    

    
class Maxout(CustomActivationFunctionBase):
    def __init__(self, input_dim, num_units):
        super(Maxout, self).__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.linear = nn.Linear(input_dim, input_dim * num_units)

    def forward(self, x):
        x = self.linear(x)
        # Reshape and take max over the num_units dimension
        x = x.view(-1, self.num_units, self.input_dim)
        return torch.max(x, dim=1).values


    
class CombinedActivation(nn.Module):
    def __init__(self, activations):
        super(CombinedActivation, self).__init__()
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for activation in self.activations:
            x = activation(x)
        return x



