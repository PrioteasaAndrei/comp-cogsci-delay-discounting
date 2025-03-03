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

- decision dificulty is around 0.1 or less, that is too small.
- the values for the standardized reaction time are between -1.745 and -1.756 which is a too small range

- sigma for reaction time varries acrross participant significantly but not across trials
modelling sigma per trial did not give any significance (0.6 +- 0.01)

- for the entire project we assumed that there is no order in the trials

# TODOS
- [ ] make some statistical tests about the influence of the parameters on the choice probabilities and reaction times (maybe about beta 1)
- [ ] implement K-fold validation
    - [ ] implement in-sample and out of sample error and do box plots with whiskers (i.e. mean the predicitons since we will have a probailistic model and for out of sample do the same with CV)
    - [ ] tune hyperparams on the mean validation error from all folds
    - [ ] in sample vs out of sample error computation
- [ ] OPTIONAL: active learning


# Completed 

- [x] implement hierachical modeling, perhaps with PyMC for sampling the join posterior
- [x] take condition into consideration for the sign of certain parameters
- [x] inmulteste dificulty cu 5 sa fie mai spread out.
- [x] model comparison with AIC and BIC (minimize to get the best)

