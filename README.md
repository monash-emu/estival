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
- 0.3.0
Note - breaking changes!
Remove old AuTuMN MCMC implementation
Move nevergrad/pymc -> wrappers
Expand likelihood output tools
Include parallelism framework
- 0.3.1
Bugfix (submodules not properly exported)
- 0.3.2
SampleIterator tools (better support for shaped priors)
Attempted map_parallel bugfix
- 0.3.3
Requirements fix (update summerepi2)
- 0.3.4
Swap modelled/target data in Normal and TruncatedNormal targets (incorrect results previously)
- 0.3.5
Add sampling utils
Add gamma prior
Support multiple targets for each derived output
- 0.3.6
Minor bugfix to 0.3.5
- 0.3.7
Reimplement BetaPrior, and get_series and finite_bounds for priors
- 0.3.8
Fix BetaPrior.to_pymc, add testing
- 0.3.9
Bugfix (BetaPrior .from_ method injected arrays into params)
- 0.4.0
Improved sampling tools
- 0.4.1
Extend sampling tools, more options for map_parallel
- 0.4.2b
Experimental release using expanded transform for uniform priors
- 0.4.3
Better exec_mode defaults for parallel helper functions
- 0.4.4
Fix issues with xarray converting array parameters
- 0.4.5
Fix nevergrad wrapper issue with infinite support priors