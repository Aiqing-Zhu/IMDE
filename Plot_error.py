import numpy as np
import matplotlib.pyplot as plt
import torch

from PD_data import PDData
from imde import IE_PD
xsize=12
legendsize=20
ticksize=15
titlesize=18
linewidth=2
hlinewidth = 1.5

def main():
    train_num=1
    test_num=2000
    space=[-3.8, 3.8, -1.2, 1.2]
    data = PDData(train_num, test_num, space=space, option='domain')
    data.device = 'cpu'
    data.dtype='float'
    fig, ax=plt.subplots(2,2, figsize=(10,8))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.01, hspace=0.01)
    
    plot_e_h(ax[0,0], data, 'euler')
    plot_e_h(ax[0,1], data, 'explicit midpoint') 
    plot_e_n(ax[1,0], data, 'euler')
    plot_e_n(ax[1,1], data, 'explicit midpoint')

   
    fig.set_tight_layout(True)
    fig.savefig('PD_error.pdf', bbox_inches='tight')
    
    
 
def plot_e_h(ax, data, integrator = 'explicit midpoint'):
    a=np.zeros([2,8])
    e_f=np.zeros([5,8])
    e_m=np.zeros([5,8])
    H=[]
    for i in range(1,9):
        H.append(0.02*i)
        for j in range(1,6):
            local = '.\\outputs\\PD_'+integrator+'_1_{}_{}\\model_best.pkl'.format(j, 0.02*i)
            net = torch.load(local, map_location='cpu')
            e_f[j-1, i-1], e_m[j-1, i-1] = error(data, net, 0.02*i)
    a[0,:] = e_f.mean(0)
    a[1,:] = e_m.mean(0)
        
        
    ax.set_xlim(0.015, 0.165)
    if integrator == 'euler' :
        ax.set_yticks([0,a[0,0],a[0,1],a[0,3],a[0,7]])
        ax.axhline(y=a[0,0],xmax=0.005/0.15,ls=":",c="grey",linewidth = hlinewidth)#Add horizontal line
    else: ax.set_yticks([0,a[0,1],a[0,3],a[0,7]])
    
    print('h',integrator, np.log(a[0,7]/a[0,3])/np.log(2), np.log(a[0,3]/a[0,1])/np.log(2), np.log(a[0,7]/a[0,1])/np.log(4))
    
    ax.axhline(y=a[0,1],xmax=0.025/0.15,ls=":",c="grey",linewidth = hlinewidth)#Add horizontal line
    ax.axhline(y=a[0,3],xmax=0.064/0.15,ls=":",c="grey",linewidth = hlinewidth)#Add horizontal line
    ax.axhline(y=a[0,7],xmax=0.145/0.15,ls=":",c="grey",linewidth = hlinewidth)#Add horizontal line
    
    ax.plot(H,a[0],'o-', c = 'b') 
    
    ax.plot(H,a[1],'>--', c = 'orange')

    ax.set_title(integrator.title(),fontsize=titlesize)
    ax.set_xlabel(r'$T$', fontsize=xsize, loc='right')
    
def plot_e_n(ax, data, integrator = 'explicit midpoint'):

    a=np.zeros([2,6])
    e_f=np.zeros([5,6])
    e_m=np.zeros([5,6])
    H=[]
    for i in range(1,7):
        H.append(0.12/i)
        for j in range(1,6):
            local = '.\\outputs\\PD_'+integrator+'_{}_{}_0.12\\model_best.pkl'.format(i, j)
            net = torch.load(local, map_location='cpu')
            e_f[j-1, i-1], e_m[j-1, i-1] = error(data, net, 0.12/i)
        
    a[0,:] = e_f.mean(0)
    a[1,:] = e_m.mean(0)
    
    ax.set_yticks([0,a[0,0],a[0,1],a[0,3]])
    ax.axhline(y=a[0,0],xmax=0.95,ls=":",c="grey",linewidth = hlinewidth)#添加水平直线
    ax.axhline(y=a[0,1],xmax=0.4,ls=":",c="grey",linewidth = hlinewidth)#添加水平直线
    ax.axhline(y=a[0,3],xmax=0.15,ls=":",c="grey",linewidth = hlinewidth)#添加水平直线
    
    if integrator == 'euler':
        ax.plot(H,a[0],'o-', c = 'b',label = r'Error $(f_{\theta},f\ )$', zorder =2)      
        ax.plot(H,a[1],'>--', c = 'orange',linewidth = linewidth, zorder =1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.75, -0.25), 
                  fontsize=legendsize, frameon=False, ncol=2)

    elif integrator == 'explicit midpoint':
        ax.plot(H,a[0],'o-', c = 'b',zorder =2)      
        ax.plot(H,a[1],'>--', c = 'orange',linewidth = linewidth, label = r'Error $(f_{\theta},f_{h})$',zorder =1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.25), 
                  fontsize=legendsize, frameon=False, ncol=2)
        
    xtick = [r"$\frac{1}{1}$",r"$\frac{1}{2}$",r"$\frac{1}{3}$",r"$\frac{1}{4}$",r"$\frac{1}{5}$",r"$\frac{1}{6}$"]
    ax.set_xticks(H)
    ax.set_xticklabels(xtick,fontsize =ticksize)
    
    
    ax.set_title(integrator.title(),fontsize=titlesize)
    ax.set_xlabel(r'1/S', fontsize=xsize, loc='right')
    print('n', integrator, np.log(a[0,0]/a[0,1])/np.log(2), np.log(a[0,1]/a[0,3])/np.log(2), np.log(a[0,0]/a[0,3])/np.log(4))
 

def e(f):
    return f.abs().max(-1)[0].mean().detach().numpy()

def error(data, net, h):
    e_f = e(data.f(data.y_test)-net.vf(data.y_test))
    e_modi = e(IE_PD(h, net.integrator)(data.y_test)-net.vf(data.y_test))
    return e_f, e_modi


if __name__ == '__main__':
    main()  
