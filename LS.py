import torch
import argparse
import numpy as np

import learner as ln
from LS_data import LSData

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0)
args = parser.parse_args()

def main():
    run(steps=2, integrator = 'explicit midpoint')
    run(steps=12, integrator = 'explicit midpoint')
    run(steps=2, integrator = 'rk4')
def run(steps, integrator):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(args.device)
    else: 
        device ='cpu'

    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'
    Nintegrator = integrator
    
    Nsteps = steps
    
    x0=np.array([-0.8,0.7,2.6])
    h=0.04
    train_num = 250
    test_num=0
    data = LSData(train_num, test_num, x0, h)
 
    net = ln.nn.NeuralODE(dim = data.dim, layers=Nlayers, width=Nwidth, activation=Nactivation, 
                          integrator=Nintegrator, steps=Nsteps)
    
    arguments = {
        'filename':'LS_'+Nintegrator+'_{}'.format(steps),
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 1000,
        'iterations': 300000,
        'batch_size': None,
        'print_every': 1000,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
  
    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    

if __name__ == '__main__':
    main()
    
    