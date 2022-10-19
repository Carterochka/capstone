import torch
from torch import nn

class SequentialDense1DFreeFallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)
        self.output = nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.output(x)
        x = nn.functional.relu(x)
        return x


class SequentialLinearWithCtgVelocity1DFreeFallModel(nn.Module):
    @staticmethod
    def ctg(x):
        return 1 / torch.tan(x)

    def __init__(self):
        super().__init__()
        self.velocity = nn.Linear(3,1)
        self.acceleration = nn.Linear(3,1)
        nn.init.kaiming_uniform_(self.velocity.weight)
        nn.init.kaiming_uniform_(self.acceleration.weight)
        self.output = nn.Linear(2,1)
    
    def forward(self, x):
        v = SequentialLinearWithCtgVelocity1DFreeFallModel.ctg(self.velocity(x))
        g = nn.functional.relu(self.acceleration(x))
        x = torch.cat((v, g), dim=2)
        x = self.output(x)
        x = nn.functional.relu(x)
        return x


class SequentialKinem1DFreeFallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)
        self.output = nn.Linear(3,1)        
    
    def forward(self, x):
        out = nn.functional.relu(self.hidden(x))
        out = torch.cat((out, x[0,:,-1].reshape(-1,1).unsqueeze(dim=0)), dim=2)
        out = self.output(out)
        out = nn.functional.relu(out)
        return out