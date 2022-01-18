import numpy as np
import torch

import learner as ln
from learner.integrator.rungekutta import RK4

class LSData(ln.Data):
    def __init__(self, train_num=150, test_num=0, x0=[-0.8,0.7,2.6], h=0.04, add_h=True, type='flow'):
        super(LSData, self).__init__()
        self.solver = RK4(self.f, N=100)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
    
    def f(self, y):
        y=10*y
        f = np.ones_like(y)
        sig ,r, b = 10, 28, 8/3
        f[...,0] = sig * (y[...,1]-y[...,0])
        f[...,1]= -y[...,0]*y[...,2]+r *y[...,0] - y[...,1]
        f[...,2] = y[...,0]*y[...,1] - b*y[...,2]
        return f/10
    
    @property
    def dim(self):
        return 3
    
    def __generate_flow(self, h, num, x0):

        X = self.solver.flow(np.array(x0), h, num)
        x, y = X[:-1], X[1:]
        if self.add_h:
            x = torch.cat([torch.tensor(x), self.h * torch.ones([x.shape[0], 1])], dim=1)       
        return x, y     
    
    def __init_data(self):
        X, Y = self.__generate_flow(self.h, self.train_num, self.x0)
        self.X_train, self.y_train = X, Y
        self.X_test, self.y_test = X, Y


    