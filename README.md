# PyTorch Implementation of: On Numerical Integration in Neural Ordinary Differential Equations


## Requirements 
* Python 
* torch
* numpy
* matplotlib

## Reproducing the results of the paper
In general all parameters which need to be specified are given in the supplementary material to the paper.

### Quick verification 
To quick verify that training Neural ODE returns an close approximation of the IMDE, run:
```
python PD.py --option 'test'
```
on cpu.


### Running Experiments for Figure 1 :
To train the models, run:
```
python PD.py --option 'traj'
python DO.py
python LS.py
```
After training, run:
```
python Plot_traj.py
```


### Running Experiments for Figure 2:
Here,  we used 5 different seed which can also be set via the command line `random_seed` parameter.
To train the models, run:
```
python PD.py --option 'error' --random_seed 1
python PD.py --option 'error' --random_seed 2
python PD.py --option 'error' --random_seed 3
python PD.py --option 'error' --random_seed 4
python PD.py --option 'error' --random_seed 5
```
After training, run:
```
python Plot_error.py
```


### Running Experiments for Figure 3:
Here we use the model trained in the previous experiment

To plot Figure 3, run:
```
python Plot_hami.py
```



## References
[1] [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

[2] [learner](https://github.com/jpzxshi/learner).

[3] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. 
"Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018. 

[4] Aiqing Zhu, Pengzhan Jin, Beibei Zhu, Yifa Tang, [On Numerical Integration in Neural Ordinary Differential Equations](https://proceedings.mlr.press/v162/zhu22f.html), Proceedings of the 39th International Conference on Machine Learning, PMLR 162, 27527-27547 (2022).
