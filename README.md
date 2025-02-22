This is the final project for the Computational Cognitive Science class, WS24/25, University of Heidelberg.

# Setup

```bash
chmod +x setup.sh
./setup.sh
conda activate pd-ws24
```

# Observations

- using standardization for the response times and the choices results in few divergences (2,3), all metrics look good but the parameters are close to 0 and the prediction is close to 0.5 for most of them.

- using the loss aversion parameter results in more divergences (20); we have a change in the distribution of the choice probabilities, they are more centered around 0.6 than before and some are close to 1. (we get a mean of 0.6)
REACTION TIME PREDICTION DOESNT WORK - outputs the same value for each trial of a participant

- using non standardize data results in more divergences (100) and the model is not able to predict the reaction times at all, but choice prediction are better (mean of 0.62)

# TODOS

- [ ] implement K-fold validation
    - [ ] implement in-sample and out of sample error and do box plots with whiskers (i.e. mean the predicitons since we will have a probailistic model and for out of sample do the same with CV)
    - [ ] tune hyperparams on the mean validation error from all folds
- [ ] OPTIONAL: active learning
- [ ] model comparison with AIC and BIC (minimize to get the best)
- [ ] research meaningful statistical tests that can be used to asses the influence of certain parameters of our models


# Completed 

- [x] implement hierachical modeling, perhaps with PyMC for sampling the join posterior
- [x] take condition into consideration for the sign of certain parameters