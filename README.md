 
# PyTorch Implementation of: On Numerical Integration in Neural Ordinary Differential Equations

The experiments based on this library are fully supported to run on single/multiple gpus. 
By default, the device is set to cpu. All our experiments where run on a single GPU.

## Requirements 
* Python 
* torch
* numpy
* matplotlib

## Reproducing the results of the paper
In general all parameters which need to be specified are given in the supplementary material to the paper.

### Quick verification 
For quick verify the theorical analysis, run:
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
[1] [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

[2] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. 
"Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018. 
