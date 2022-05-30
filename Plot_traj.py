import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from PD_data import PDData
from DO_data import DOData
from LS_data import LSData
from learner.integrator.rungekutta import RK4
from imde import IE_PD, IE_DO, IE_LS

########Flow
xsize=15
legendsize=16
ticksize=15
titlesize=18
linewidth=3
hlinewidth = 1.5

PDsteps =1000
DOsteps =1000
LSsteps =500
def main():
    fig = plt.figure(figsize=(26, 9))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
        wspace=0.15, hspace=0.3)
    gs = gridspec.GridSpec(nrows=4, ncols=3, height_ratios=[1, 1, 1,0.4])
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[2,0])] 
    plotPD(ax)
    
    ax = fig.add_subplot(gs[3,0])
    plot_title(ax, 'Pendulum system')

    ax = [fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[2,1])] 
    plotDO(ax)   
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.7), 
           fontsize=legendsize, frameon=False, ncol=3)    
    
    ax = fig.add_subplot(gs[3,1])
    plot_title(ax, 'Damped harmonic oscillator')    
           
    ax = [fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,2]), fig.add_subplot(gs[2,2])] 
    plotLS(ax)  

    ax = fig.add_subplot(gs[3,2])
    plot_title(ax, 'Nonlinear Lorenz system')  
    fig.savefig('1.pdf', bbox_inches='tight')


def plot_title(ax,name):
    ax.text(0,-0.4, name, fontsize=20, color='black',
      horizontalalignment='center', verticalalignment='center')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.axis('off')


def plotPD(ax):
    x0=torch.tensor([0.,1.])
    h=0.04
    train_num = 2
    test_num=2
    size=0.01
    
    data = PDData(train_num, test_num, h, x0, option = 'output')
    flow_true = data.solver.flow(x0, size, PDsteps)
    t = np.linspace(0, size*PDsteps, PDsteps+1)
    ax[0].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[1].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[2].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([2,0,-2])
    ax[1].set_yticks([2,0,-2])
    ax[2].set_yticks([2,0,-2])
    ax[0].tick_params(labelsize=ticksize) 
    ax[1].tick_params(labelsize=ticksize) 
    ax[2].tick_params(labelsize=ticksize)  
    ax[2].set_xlabel(r'$t$', fontsize=xsize)
    ax[0].set_ylabel(r'$y_1$', fontsize=xsize)
    ax[1].set_ylabel(r'$y_1$', fontsize=xsize)
    ax[2].set_ylabel(r'$y_1$', fontsize=xsize)
    
    local = '.\\outputs\\PD_euler_2_0\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_PDflow(ax[0], data, h, net, x0)

    local = '.\\outputs\\PD_euler_12_0\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_PDflow(ax[1], data, h, net, x0)

    local = '.\\outputs\\PD_explicit midpoint_2_0\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_PDflow(ax[2], data, h, net, x0)

    
def plot_PDflow(ax, data, h, net, x0):
    size=0.01
    t = np.linspace(0, size*PDsteps, PDsteps+1)

    flow_modi = RK4(IE_PD(h/net.steps, integrator=net.integrator), N=10).flow(x0, size, PDsteps)
    flow_pred = net.predict(x0, size, steps=PDsteps, keepinitx=True, returnnp=True)
    
    ax.plot(t, flow_modi[:, 0], color='red', label='IMDE', linewidth= linewidth, zorder=1)
    ax.plot(t, flow_pred[:, 0], color='b', label='Neural ODE',  linestyle='--', dashes=(4, 2), linewidth= linewidth, zorder=2)
    ax.set_title(net.integrator.title()+' $S=${}'.format(net.steps), fontsize=titlesize, loc= 'left')
 
    
def plotLS(ax):
    x0=np.array([-0.8,0.7,2.6])
    h=0.04
    train_num = 2
    test_num = 2
    size=0.01
    data = LSData(train_num, test_num)
    
    flow_true = data.solver.flow(x0, size, LSsteps)
    t = np.linspace(0, size*LSsteps, LSsteps+1)
    
    ax[0].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[1].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[2].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    
    ax[0].set_yticks([1,0,-1])
    ax[1].set_yticks([1,0,-1])
    ax[2].set_yticks([1,0,-1])
    ax[0].tick_params(labelsize=ticksize) 
    ax[1].tick_params(labelsize=ticksize) 
    ax[2].tick_params(labelsize=ticksize)
    ax[2].set_xlabel(r'$t$', fontsize=xsize)
    ax[0].set_ylabel(r'$y_1$', fontsize=xsize)
    ax[1].set_ylabel(r'$y_1$', fontsize=xsize)
    ax[2].set_ylabel(r'$y_1$', fontsize=xsize)

    local = './outputs/LS_explicit midpoint_2/model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_LSflow(ax[0], data, h, net, x0)

    local = './outputs/LS_explicit midpoint_12/model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_LSflow(ax[1], data, h, net, x0)

    local = './outputs/LS_rk4_2/model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_LSflow(ax[2], data, h, net, x0)
 
    
def plot_LSflow(ax, data, h, net, x0):
    size=0.01
    t = np.linspace(0, size*LSsteps, LSsteps+1)
    x0 = torch.tensor(x0, dtype=torch.float32)
    flow_modi = RK4(IE_LS(h/net.steps, integrator=net.integrator), N=10).flow(x0, size, LSsteps)
    flow_pred = net.predict(x0, size, steps=LSsteps, keepinitx=True, returnnp=True)
    
    ax.plot(t, flow_modi[:, 0], color='red', label='IMDE', linewidth= linewidth, zorder=1)
    ax.plot(t, flow_pred[:, 0], color='b', label='Neural ODE',  linestyle='--', dashes=(4, 2), linewidth= linewidth, zorder=2)
    if net.integrator =='rk4':      
        ax.set_title('RK4 $S=${}'.format(net.steps), fontsize=titlesize, loc= 'left')
    else:
        ax.set_title(net.integrator.title()+' $S=${}'.format(net.steps), fontsize=titlesize, loc= 'left')

def plotDO(ax):
    x0=torch.tensor([2.,0.])
    h=0.02
    train_num = 2
    test_num = 2 
    size=0.01
    data = DOData(train_num, test_num, h, x0, option = 'output')

    flow_true = data.solver.flow(x0, size, DOsteps)
    t = np.linspace(0, size*DOsteps, DOsteps+1)
    ax[0].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[1].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[2].plot(t, flow_true[:, 0], color='grey', label='Original ODE', linewidth= linewidth, zorder=0)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([1,0,-1])
    ax[1].set_yticks([1,0,-1])
    ax[2].set_yticks([1,0,-1])
    ax[0].tick_params(labelsize=ticksize) 
    ax[1].tick_params(labelsize=ticksize) 
    ax[2].tick_params(labelsize=ticksize)
    ax[2].set_xlabel(r'$t$', fontsize=xsize)
    ax[0].set_ylabel(r'$y_1$', fontsize=xsize)
    ax[1].set_ylabel(r'$y_1$', fontsize=xsize)
    ax[2].set_ylabel(r'$y_1$', fontsize=xsize)
    
    local = '.\\outputs\\DO_euler_2\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_DOflow(ax[0], data, h, net, x0)

    local = '.\\outputs\\DO_euler_12\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_DOflow(ax[1], data, h, net, x0)

    local = '.\\outputs\\DO_explicit midpoint_2\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    plot_DOflow(ax[2], data, h, net, x0)

    
def plot_DOflow(ax, data, h, net, x0):
    size=0.01
    t = np.linspace(0, size*DOsteps, DOsteps+1)
    
    flow_modi = RK4(IE_DO(h/net.steps, integrator=net.integrator), N=10).flow(x0, size, DOsteps)
    flow_pred = net.predict(x0, size, steps=DOsteps, keepinitx=True, returnnp=True)
    ax.plot(t, flow_modi[:, 0], color='red', label='IMDE', linewidth= linewidth, zorder=1)
    ax.plot(t, flow_pred[:, 0], color='b', label='Neural ODE',  linestyle='--', dashes=(4, 2), linewidth= linewidth, zorder=2)    
    ax.set_title(net.integrator.title()+' $S=${}'.format(net.steps), fontsize=titlesize, loc= 'left')
    
  
if __name__ == '__main__':
    main()    
