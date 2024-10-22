# Matching Based CFR
### Papers
- [Shalit et al., 2017] Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect: generalization bounds and algorithms." International Conference on Machine Learning. PMLR, 2017
- [Johansson et al., 2016] Johansson, Fredrik, Uri Shalit, and David Sontag. "Learning representations for counterfactual inference." International conference on machine learning. PMLR, 2016.

The following github repository was used extensively as a reference.
- [introduction-to-cfr] https://github.com/MasaAsami/introduction_to_CFR
## Usage

```
$ python experiment_run.py
```

If you want to change a hyperparameters:
```
$ python experiment_run.py -m alpha=0,0.1,0.01,0.001,0.0001,1,100,10000,100000,1000000,10000000,100000000,1000000000,10000000000,100000000000 split_outnet=True,False
```