# estival
Calibration and optimization tools for summer2
https://github.com/monash-emu/summer2

Estival provides a simple API for using summer2 CompartmentalModels with a variety of optimization frameworks, including
- pymc
- nevergrad

### CHANGELOG

- 0.2.2  
Add logprior/logposterior to BayesianCompartmentalModel
- 0.2.3  
Include tensorflow-probability(jax) for more (and better tested) stats modules
- 0.2.4
Bugfix (vector priors were not exported to pymc correctly)
Add Epoch support to allow DatetimeIndex targets
- 0.2.5
Bugfix for BinomialTarget (wasn't indexing modelled data)
- 0.2.6
Bugfix (reference index for models without date returned incorrect type)