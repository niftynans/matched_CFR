# Matching Based CFR
### Overview
This code repository replicates our work as shown in the preprint titled, 'Representation Learning Preserving Ignorability and Covariate Matching for Treatment Effects' where the goal is accurate treatment effect estimation in the presence of hidden confounding as well as selection bias.

### Usage
Firstly, setup the python environment using packages from requirements.txt. Once the environment is setup, you can run the experiments with the following arguments:

```
$ python experiment_run.py --dataset = ihdp, jobs, cattaneo --algorithm = match, erm, match_alternate, match_only
```
Here, 'match' denotes our method, 'erm' is the empirical risk minimizer, 'match_alternate' is the alternate minimization oracle, and 'match_only' refers to using only the meta learning gradient matching approach (referred to as FISH algorithm in Shi et al, 2021.)

Hyperparameters can be changed in the configs file named 'experiments.yaml' within the 'configs' folder, or directly on the command line. An example of that is given below:
```
$ python experiment_run.py -m alpha=0,100000000 split_outnet=True,False
```

In order to run the baseline methods, i.e., meta learners like the S-Learner, T-Learner, X-Learner, DA-Learner, and the DR-Learner kindly run the following command:

```
$ jupyter nbconvert --execute --to baselines.ipynb --inplace baselines.ipynb
```
Or simply run the ipynb file using a jupyter ipykernel.