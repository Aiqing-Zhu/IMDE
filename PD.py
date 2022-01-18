import torch
import argparse

import learner as ln
from PD_data import PDData

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0, help='GPU node')
parser.add_argument('--random_seed',type=int, default=0, help='number of seeds(runs) to test the performance')
parser.add_argument('--option',type=str, default='test')
args = parser.parse_args()

def main():
    if args.option == 'test':
        run(steps=2, integrator='euler', filename = 'PD_test')
        
        #Plot
        import matplotlib.pyplot as plt
        import numpy as np
        from imde import IE_PD
        from learner.integrator.rungekutta import RK4
        
        local = '.\\outputs\\PD_test\\model_best.pkl'
        net = torch.load(local, map_location='cpu')
        h=0.04
        x0=torch.tensor([0.,1.])
        train_num = 2
        test_num=2
        size=0.01
        data = PDData(train_num, test_num, h, x0, option = 'output')

        flow_modi = RK4(IE_PD(h/net.steps, integrator=net.integrator), N=10).flow(x0, 0.01, 200)
        flow_pred = net.predict(x0, size, steps=200, keepinitx=True, returnnp=True)
        flow_true = data.solver.flow(x0, size, 200)
        plt.plot(flow_true[:, 0], flow_true[:, 1], color='grey', label='Original ODE', zorder=0)
        plt.plot(flow_modi[:, 0], flow_modi[:, 1], color='red', label='IMDE',zorder=1)
        plt.plot(flow_pred[:, 0], flow_pred[:, 1], color='b', label='Neural ODE', linestyle='--', dashes=(4, 2),zorder=2)
        plt.legend()
        plt.show()
        
        t = np.linspace(0, size*200, 201)
        plt.plot(t, flow_true[:, 0], color='grey', label='Original ODE', zorder=0)
        plt.plot(t, flow_modi[:, 0], color='red', label='IMDE',zorder=1)
        plt.plot(t, flow_pred[:, 0], color='b', label='Neural ODE', linestyle='--', dashes=(4, 2),zorder=2)
        plt.legend()
        plt.show()
    
    if args.option == 'traj':
        run(steps=2, integrator='euler', filename = 'PD_euler_2_0')
        run(steps=12, integrator='euler', filename = 'PD_euler_12_0')
        run(steps=2, integrator='explicit midpoint', filename = 'PD_explicit midpoint_2_0') 
        
    if args.option == 'error':
        i=args.random_seed        
        for j in range(1,9):
            run(steps=1, h=0.02*j, integrator='explicit midpoint',i=i, 
                filename = 'PD_explicit midpoint_{}_{}_{}'.format(1, i, 0.02*j))
        
        for j in range(1,9):
            run(steps=1, h=0.02*j, integrator='euler',i=i,
                filename = 'PD_euler_{}_{}_{}'.format(1, i, 0.02*j))

        for j in range(2,7):
            run(steps=j, h=0.12, integrator='explicit midpoint',i=i, 
                filename = 'PD_explicit midpoint_{}_{}_{}'.format(j, i, 0.12))
        
        for j in range(2,7):
            run(steps=j, h=0.12, integrator='euler',i=i, 
                filename = 'PD_euler_{}_{}_{}'.format(j, i, 0.12))
        
def run(steps=1, h=0.04, integrator='explicit midpoint', i=1, filename='test'):
    
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(args.device)
    else: 
        device ='cpu'

    
    h=h
    train_num = 10000
    test_num=2000
    space=[-3.8, 3.8, -1.2, 1.2]
    
    data = PDData(train_num, test_num, h=h, space=space, option='domain')

    Nlayers =3
    Nwidth =128
    Nactivation = 'tanh'
    Nintegrator = integrator
    Nsteps = steps  
    ite =10000 if args.option == 'test' else 300000

    net = ln.nn.NeuralODE(data.dim, layers=Nlayers, width=Nwidth, activation=Nactivation, 
                              integrator=Nintegrator, steps=Nsteps)

    
    arguments = {
        'filename': filename,
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.01,
        'lr_decay': 1000,
        'iterations': ite,
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
    
    