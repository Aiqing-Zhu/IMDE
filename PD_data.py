import numpy as np
import torch

import learner as ln
from learner.integrator.hamiltonian import SV


class PDData(ln.Data):
    '''Data for learning the pendulum system with the Hamiltonian H(p,q)=(1/2)p^2−cos(q).
    '''
    def __init__(self, train_num, test_num, h=0.1, x0=None, space=None, add_h=True, N=10, option='trajectory'):
        super(PDData, self).__init__()      
        self.m, self.g, self.l = 1,5,1
        m,g,l=self.m, self.g, self.l
        self.dH = lambda p, q: (l**2/m*p, 2*m*g*l*np.sin(q))
        self.solver = SV(None, self.dH, iterations=1, order=6, N = N)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.space = space
        self.add_h = add_h
        self.option = option
        self.__init_data()
        
    @property
    def dim(self):
        return 2

    def f(self, y):
        m,g,l=self.m, self.g, self.l
        f = torch.empty(y.shape)
        f[...,0] = -2*m*g*l*torch.sin(y[...,1])
        f[...,1] = l**2/m*y[...,0]
        return f
    
    def __generate_flow(self, h, num, x0=[2.,0.]):
        X = self.solver.flow(torch.tensor(x0), h, num)
        x, y = X[:-1], X[1:]
        if self.add_h:
            x = torch.cat([x, self.h * torch.ones([x.shape[0], 1])], dim=1)       
        return x, y
   
    def __generate(self, X, h):
        return np.array(list(map(lambda x: self.solver.solve(x, h), X)))
    
    def __generate_random(self, space, h, num):
        pmin, pmax, qmin, qmax = space[0],space[1],space[2],space[3]
        p = np.random.rand(num) * (pmax - pmin) + pmin
        q = np.random.rand(num) * (qmax - qmin) + qmin
        x = np.hstack([p[:, None], q[:, None]])
        y = self.__generate(x, h) 
        if self.add_h:
            x = np.hstack([x, self.h * np.ones([x.shape[0], 1])])
        return x, y    
    
    
    def __init_data(self):
        if self.option == 'trajectory':        
            self.X_train, self.y_train = self.__generate_flow(self.h, self.train_num, self.x0)
            self.X_test, self.y_test = self.__generate_flow(self.h, self.test_num, self.x0)
        elif self.option == 'domain':
            self.X_train, self.y_train = self.__generate_random(self.space, self.h, self.train_num)
            self.X_test, self.y_test = self.__generate_random(self.space, self.h, self.test_num)            
        elif self.option == 'output':
            self.add_h=False
            self.X_test, self.y_test = self.__generate_flow(self.h, self.test_num, self.x0)
            self.X_train, self.y_train = self.X_test, self.y_test
        else: raise ValueError

         
