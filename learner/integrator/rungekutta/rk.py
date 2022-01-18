# -*- coding: utf-8 -*-
import abc
import numpy as np
import torch

class RK(abc.ABC):
    '''Runge-Kutta method.
    '''
    def __init__(self):
        self.f = None
        self.N = None
    
    @abc.abstractmethod
    def solver(self, x, h):
        pass
        
    def solve(self, x, h):
        for _ in range(self.N):
            x = self.solver(x, h / self.N)
        return x
    
    def flow(self, x, h, steps):
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        size = len(x.shape) if isinstance(x, np.ndarray) else len(x.size())
        X = [x]
        with torch.no_grad():
            for i in range(steps):
                X.append(self.solve(X[-1], h))
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return np.hstack(X).reshape(shape) if isinstance(x, np.ndarray) else torch.cat(X, dim=-1).view(shape)

class Euler(RK):
    '''Explicit Euler method.
    '''
    def __init__(self, f, N=1):

        self.f = f
        self.N = N
        
    def solver(self, x, h):
        '''Order 1.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
        h: float
        '''
        return x + h * self.f(x)

class Midpoint(RK):
    '''Explicit midpoint method
    '''
    def __init__(self, f, N=1):
        self.f = f
    
        self.N = N
        
    def solver(self, x, h):
        '''Order 2
            x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
            h: float
        ''' 

        return x + h * self.f(x + h/2*self.f(x))      


class RK4(RK):
    '''Runge-Kutta method of order 4.
    '''
    def __init__(self, f, N=1):
        self.f = f
        self.N = N
        
    def solver(self, x, h):
        '''Order 4.
        x: np.ndarray or torch.Tensor of shape [dim] or [num, dim].
        h: float
        '''
        k1 = self.f(x)
        k2 = self.f(x + h * k1 / 2)
        k3 = self.f(x + h * k2 / 2)
        k4 = self.f(x + h * k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6) 


        