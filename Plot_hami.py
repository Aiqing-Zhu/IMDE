import matplotlib.pyplot as plt
import torch

from imde import IE_PD
from PD_data import PDData
from learner.integrator.rungekutta import RK4

########Flow
xsize=12
legendsize=16
ticksize=15
titlesize=18
linewidth=2
hlinewidth = 1.5

    
def main():
    
    x0=torch.tensor([0.,1.])
    h=0.1
    train_num = 1
    test_num=1
    data = PDData(train_num, test_num, h, x0, option = 'output')

    fig, ax=plt.subplots(1,2, figsize=(10,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.01, hspace=0.01)

    size=0.01
    PDsteps =500
    flow_true = data.solver.flow(x0, size, PDsteps)
    
    
    local = '.\\outputs\\PD_euler_6_1_0.12\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    h=0.12    
    flow_modi = RK4(IE_PD(h/net.steps, integrator=net.integrator), N=10).flow(x0, size, PDsteps)
    flow_pred = net.predict(x0, size, steps=PDsteps, keepinitx=True, returnnp=True)
    ax[0].plot(flow_true[:, 0], flow_true[:, 1], color='grey', label='Original ODE', zorder=0)
    ax[0].plot(flow_modi[:, 0], flow_modi[:, 1], color='red', label='IMDE',zorder=1)
    ax[0].plot(flow_pred[:, 0], flow_pred[:, 1], color='b',  linestyle='--', dashes=(4, 2),zorder=2)
    ax[0].set_title(net.integrator.title(), fontsize=titlesize, loc= 'left')   
    ax[0].set_xlim(-3.8,3.8)
    ax[0].set_ylim(-1.2,1.2)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.55, -0.15), 
                  fontsize=legendsize, frameon=False, ncol=2)
    
    
    local = '.\\outputs\\PD_explicit midpoint_1_1_0.12\\model_best.pkl'
    net = torch.load(local, map_location='cpu')
    h=0.12
    flow_modi = RK4(IE_PD(h/net.steps, integrator=net.integrator), N=10).flow(x0, size, PDsteps)
    flow_pred = net.predict(x0, size, steps=PDsteps, keepinitx=True, returnnp=True)
    ax[1].plot(flow_true[:, 0], flow_true[:, 1], color='grey', zorder=0)
    ax[1].plot(flow_modi[:, 0], flow_modi[:, 1], color='red',zorder=1)
    ax[1].plot(flow_pred[:, 0], flow_pred[:, 1], color='b', label='Neural ODE',  linestyle='--', dashes=(4, 2),zorder=2)
    ax[1].set_title(net.integrator.title(), fontsize=titlesize, loc= 'left')   
    ax[1].set_xlim(-3.8,3.8)
    ax[1].set_ylim(-1.2,1.2)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.25, -0.15), 
                  fontsize=legendsize, frameon=False, ncol=2)
    
    ax[0].set_xlabel(r'$p$', fontsize=xsize)
    ax[1].set_xlabel(r'$p$', fontsize=xsize)
    ax[0].set_ylabel(r'$q$', fontsize=xsize)
    ax[1].set_yticks([])

    fig.set_tight_layout(True)
    fig.savefig('PD_hami.pdf', bbox_inches='tight')

    

if __name__ == '__main__':
    main()  
