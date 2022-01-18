import torch.nn as nn
import torch

from .module import StructureNN, LossNN
from .fnn import FNN

class TestStructureNN(StructureNN):
    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', softmax=False):
        super(TestStructureNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.softmax = softmax
        
        self.modus = self.__init_modules()
        self.__initialize()
        
    def forward(self, x):
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
        x = self.modus['LinMout'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules
    
    def __initialize(self):
        for i in range(1, self.layers):
            self.weight_init_(self.modus['LinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)
        self.weight_init_(self.modus['LinMout'].weight)
        nn.init.constant_(self.modus['LinMout'].bias, 0)
    
class TestLossNN(LossNN):
    def __init__(self, dim=4, h=0.1, layers=2, width=128, activation='relu', initializer='default'):
        super(TestLossNN, self).__init__()
        self.dim = dim
        self.h = h
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.modus = self.__init_modules()

    def criterion(self, x0, x1):
        return torch.nn.MSELoss()(self.modus['f'](x0[1]), (x1-2*x0[1]+x0[0])/self.h**2)
    
    def dH(self, p, q):
        # print(p,q)
        # print(self.modus['f'](q))
        return (p, self.modus['f'](q))
    
    def predict(self, q1, q2, h=0.1, steps=100, keepinitx=False, returnnp=False):
        print(q2)
        print(self.modus['f'](q2))
        return -q1+ 2*q2+self.modus['f'](q2)*self.h**2
    
    # def predict(self, x, keepinitx=False, returnnp=False):
    #     dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
    #     d = int(dim / 2)
    #     n=2
    #     h=0.1/n
    #     p0, q0 = (x[..., :d], x[..., d:])
    #     p1, q1 = p0, q0
    #     for _ in range(2*n):
    #         q1 = q1 + h / 2 * p1
            
    #         p1 = p1 + h / 2 * self.modus['f'](q1)
    #         print(q1,p1)
    #     return np.hstack([p1, q1]) if isinstance(x, np.ndarray) else torch.cat([p1, q1], dim=-1)
        

        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, layers=self.layers, width=self.width,
                           activation=self.activation, initializer=self.initializer)
        return modules 

class SecondOrderOde(LossNN):
    def __init__(self, dim=4, layers=2, width=128, activation='relu', initializer='default'):
        super(SecondOrderOde, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width       
        self.activation = activation
        self.initializer = initializer
        self.modus = self.__init_modules()

    def criterion(self, x0, x1):
        return torch.nn.MSELoss()(self.modus['f'](x0[1]), x1-2*x0[1]+x0[0])
    
    def predict(self, x0, keepinitx=False, returnnp=False):
        return self.modus['f'](x0).cpu().detach().numpy() if returnnp else self.modus['f'](x0)
        

        
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['f'] = FNN(self.dim, self.dim, layers=self.layers, width=self.width,
                           activation=self.activation, initializer=self.initializer)
        return modules 