import torch
import argparse

import learner as ln
from DO_data import DOData


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0)
args = parser.parse_args()
def main():
    run(steps=2, integrator='euler')
    run(steps=12, integrator='euler')
    run(steps=2, integrator='explicit midpoint')

def run(steps=1, h=0.02, integrator='explicit midpoint', i=1):
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(args.device)
    else: 
        device ='cpu'    
    
    h=h
    train_num = 10000
    test_num=2000
    space=[-2.2,2.2,-2.2,2.2]
    
    data = DOData(train_num, test_num, h=h, space=space, option='domain')
    
    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'
    Nintegrator = integrator
    Nsteps = steps

    net = ln.nn.NeuralODE(data.dim, layers=Nlayers, width=Nwidth, activation=Nactivation, 
                         integrator=Nintegrator, steps=Nsteps)
    
    
    arguments = {
        'filename': 'DO_'+Nintegrator+'_{}'.format(steps),
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
