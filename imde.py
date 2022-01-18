import torch
import torch.nn as nn

class IE_DO(nn.Module):
    def __init__(self, h, integrator='euler'):
        super(IE_DO, self).__init__()
        self.h=h
        if integrator == 'euler':
            self.a = [ 1/2, 1/6, 1/6,  1/24,   3/24,   1/24,   1/24]
            
        elif integrator == 'explicit midpoint':
            self.a = [0, 1/24, 1/6, 0, 0, -1/16, -1/8]
            
        else: raise ValueError
        
        
        
    def forward(self, y):
        h=self.h
        true_A = torch.FloatTensor([[-0.1, -2.0], [2.0, -0.1]])
        
        f = y**3@true_A #f
        dff = y**3@ true_A * y**2@(3*true_A)  #f'f
        
        ddfff = (y**3@true_A)**2 * y@(6*true_A)  #f''(f,f)
        dfdff = y**3@ true_A * y**2@(3*true_A) * y**2@(3*true_A) #f'f'f
        
        dddffff = ((y**3@true_A)**3 @ (6*true_A))  #f'''(f,f,f)
        ddfdfff = (y**3@true_A*(y**3@ true_A * y**2@(3*true_A))) * y@(6*true_A) #f''(f'f,f)
        dfddfff = (((y**3@true_A)**2* y@(6*true_A))*  y**2@(3*true_A) ) #f'f''(f,f)
        dfdfdff = (y**3@ true_A * y**2@(3*true_A)* y**2@(3*true_A)* y**2@(3*true_A))#f'f'f'f
        
        return (
                f #f
                +self.a[0]* h    * dff     #f'f
                +self.a[1]* h**2 * ddfff   #f''(f,f)
                +self.a[2]* h**2 * dfdff   #f'f'f
                +self.a[3]* h**3 * dddffff #f'''(f,f,f)
                +self.a[4]* h**3 * ddfdfff #f''(f'f,f)
                +self.a[5]* h**3 * dfddfff #f'f''(f,f)
                +self.a[6]* h**3 * dfdfdff #f'f'f'f 
                )  

class IE_PD(nn.Module):
    def __init__(self, h, integrator='euler'):
        super(IE_PD, self).__init__()
        self.h=h
        if integrator == 'euler':
            self.a = [ 1/2, 1/6, 1/6,  1/24,   3/24,   1/24,   1/24]
            
        elif integrator == 'explicit midpoint':
            self.a = [0, 1/24, 1/6, 0, 0, -1/16, -1/8]
            
        else: raise ValueError
        
        
        
    def forward(self, y):
        h=self.h
        c1,c2 = 1, 10
        f = torch.empty(y.shape)#f
        f[...,0] = -c2*torch.sin(y[...,1])
        f[...,1] = c1*y[...,0]
        
        dff = torch.empty(y.shape)#f'f
        dff[...,0] = -c1*c2*torch.cos(y[...,1])*y[...,0]
        dff[...,1] = -c1*c2*torch.sin(y[...,1])
        
        ddfff = torch.empty(y.shape)#f''(f,f)
        ddfff[...,0] = c1*c1*c2*torch.sin(y[...,1])*y[...,0]**2
        ddfff[...,1] = 0 
       
        
        dfdff = torch.empty(y.shape)#f'f'f
        dfdff[...,0] = c1*c2*c2*torch.cos(y[...,1])*torch.sin(y[...,1])
        dfdff[...,1] = -c1*c1*c2*torch.cos(y[...,1])*y[...,0]

        
        dddffff = torch.empty(y.shape)#f'''(f,f,f)
        dddffff[..., 0] =  c1*c1*c1*c2*torch.cos(y[...,1])*y[...,0]**3
        dddffff[..., 1] = 0 
        
        ddfdfff = torch.empty(y.shape)#f''(f'f,f)
        ddfdfff[...,0] = -c1*c1*c2*c2*torch.sin(y[...,1])*torch.sin(y[...,1])* y[...,0]
        ddfdfff[...,1] = 0
        
        dfddfff = torch.empty(y.shape)#f'f''(f,f)
        dfddfff[...,0] = 0
        dfddfff[...,1] = c1*c1*c1*c2*torch.sin(y[...,1])*y[...,0]**2
        
        dfdfdff = torch.empty(y.shape)#f'f'f'f
        dfdfdff[...,0] = c1*c1*c2*c2*torch.cos(y[...,1])**2 * y[...,0]
        dfdfdff[...,1] = c1*c1*c2*c2*torch.cos(y[...,1])*torch.sin(y[...,1])
        
        return (
                f #f
                +self.a[0]* h    * dff     #f'f
                +self.a[1]* h**2 * ddfff   #f''(f,f)
                +self.a[2]* h**2 * dfdff   #f'f'f
                +self.a[3]* h**3 * dddffff #f'''(f,f,f)
                +self.a[4]* h**3 * ddfdfff #f''(f'f,f)
                +self.a[5]* h**3 * dfddfff #f'f''(f,f)
                +self.a[6]* h**3 * dfdfdff #f'f'f'f 
                )  
class IE_LS(nn.Module):
    def __init__(self, h, integrator='euler'):
        super(IE_LS, self).__init__()
        self.h=h
        if integrator == 'euler':
            self.a = [ 1/2, 1/6, 1/6,  1/24,   3/24,   1/24,   1/24]
            
        elif integrator == 'explicit midpoint':
            self.a = [0, 1/24, 1/6, 0, 0, -1/16, -1/8]
            
        elif integrator == 'rk4':
            self.a = [0, 0,0,0,0,0,0]
            
        else: raise ValueError
        
        
        
    def forward(self, y):
        h=self.h
        y=10*y
        f=torch.empty(y.shape)
        f[...,0]=10*y[...,1]-10*y[...,0]
        f[...,1]=28*y[...,0]-y[...,0]*y[...,2]-y[...,1]
        f[...,2]=y[...,0]*y[...,1]-8/3*y[...,2]
        
        dff = torch.empty(y.shape)
        dff[...,0] = -10*f[...,0] + 10* f[...,1]
        dff[...,1] = (28- y[...,2])* f[...,0]-f[...,1]-y[...,0]*f[...,2]
        dff[...,2] = y[...,1]* f[...,0] + y[...,0]*f[...,1]-8/3*f[...,2]
        
        dfdff=torch.empty(y.shape)
        dfdff[...,0] = -10*dff[...,0] + 10* dff[...,1]
        dfdff[...,1] = (28- y[...,2])* dff[...,0]-dff[...,1]- y[...,0]*dff[...,2]
        dfdff[...,2] =  y[...,1]* dff[...,0]+ y[...,0]*dff[...,1]-8/3*dff[...,2]
        
        ddfff=torch.empty(y.shape)
        ddfff[...,0]=0
        ddfff[...,1]=-2*f[...,0]*f[...,2]
        ddfff[...,2]=2*f[...,0]*f[...,1]
#       

        dddffff = torch.zeros(y.shape) 
        
        ddfdfff=torch.empty(y.shape)
        ddfdfff[...,0]=0
        ddfdfff[...,1]=-dff[...,0]*f[...,2]-dff[...,2]*f[...,0]
        ddfdfff[...,2]=dff[...,0]*f[...,1]+dff[...,1]*f[...,0]
        
        dfddfff=torch.empty(y.shape)
        dfddfff[...,0] = -10*ddfff[...,0] + 10* ddfff[...,1]
        dfddfff[...,1] = (28- y[...,2])* ddfff[...,0]-ddfff[...,1]- y[...,0]*ddfff[...,2]
        dfddfff[...,2] =  y[...,1]* ddfff[...,0]+ y[...,0]*f[...,1]-8/3*ddfff[...,2]
        
        dfdfdff=torch.empty(y.shape)
        dfdfdff[...,0] = -10*dfdff[...,0] + 10* dfdff[...,1]
        dfdfdff[...,1] = (28- y[...,2])* dfdff[...,0]-dfdff[...,1]- y[...,0]*dfdff[...,2]
        dfdfdff[...,2] =  y[...,1]* dfdff[...,0]+ y[...,0]*dfdff[...,1]-8/3*dfdff[...,2]
        
        return (
                f #f
                +self.a[0]* h    * dff     #f'f
                +self.a[1]* h**2 * ddfff   #f''(f,f)
                +self.a[2]* h**2 * dfdff   #f'f'f
                +self.a[3]* h**3 * dddffff #f'''(f,f,f)
                +self.a[4]* h**3 * ddfdfff #f''(f'f,f)
                +self.a[5]* h**3 * dfddfff #f'f''(f,f)
                +self.a[6]* h**3 * dfdfdff #f'f'f'f 
                )/10