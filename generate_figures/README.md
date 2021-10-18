# Generate Figures for Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling
## Requirements

The main sampler code for ESH dynamics in esh.py uses only PyTorch. 
Use ``pip install -r requirements.txt`` for requirements for all evaluation code. 
This includes some libraries for MCMC sampler comparisons like NUTS and for Runge-Kutta differential equation solvers.

## Usage

To run comparisons on synthetic datasets with several samplers:
```markdown
python figure1_single_trajectory.py  # Figure 1
python test_suite.py 200 500 ablation  # Figs. 3,
python test_suite.py 200 500 comparison  # Figs. 4, Table 1, 
```
The MMD plots were done with 200 steps and 500 sample chains. 

Samplers.py is a library that is called for generating the comparison methods, 
including Runge-Kutta methods for ESH dynamics. It also has a wrapper to give ESH 
results in the same format. It calls esh.py, the library for ESH sampling with the leapfrog integrator. 

Fig.2 is generated from a Mathematica file "esh plot.nb" in the "assets" folder. It also generates an animation shown
on the main github page.  

To generate Fig. 5, choose an option below. 
```markdown
python plot_valley.py {0,1}  # Fig. 5
```

To sample a pre-trained JEM model, you must have JEM and model.pt inside a folder named JEM. 
To generate images or the average energy plots do the following. 
```markdown
python test_jem.py n_steps n_images prior  # Fig. 6, 17
python plot_jem_energy.py n_steps batch_size prior  # Fig. 6, 17
```
n_steps is number of steps in each chain, n_images is number of images to show, 
"prior" says whether to start from uniform noise or start from the replay buffer. 
 
```markdown
python test_jarzynski.py {0,1}  # Fig. 8, 
```


### CIFAR results
To generate plots on synthetic datasets choose an option of dataset and sampler. 
```markdown
cd ebm_anatomy_esh
python train_toy_jar.py {esh,ula} {rings,mog}
```

For CIFAR experiments train as follows, choosing the sampler to use. More options are on config_locker folder. 
```markdown
cd ebm_anatomy_esh
python train_data_pcd.py {esh,ula} cifar10
```
And evaluate with the following. Choose the sampler, the initialization, the number of steps and the model checkpoint(s). 
```markdown
python eval.py {esh,ula} {uniform,buffer,gaussian,color} num_steps /path/to/checkpoint (or list of checkpoints for ensembling)
```

