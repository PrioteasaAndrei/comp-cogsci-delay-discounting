This is the final project for the Computational Cognitive Science class, WS24/25, University of Heidelberg.

# Things to consider

- longer reaction time usually means harder choice to make (e.g. risk aversion)
- smaller reaction time doesn't necessary mean easier decision (e.g. impulsiveness)
- same choice modeling as in class, only reaction time modelling differs.
- only two modalities used
- think about a probability weighting function to model subjective probabilities for impulsivness and risk aversion (use a measure of objective value for each value probabilistic vs immediate and from that abstract personal subjectivity). put this into the choice not into the response times (see course 4).

# Setup

```bash
chmod +x setup.sh
./setup.sh
conda activate pd-ws24
```

# Good practice

- try to use pyMC(https://www.pymc.io/welcome.html) for the modeling part

# TODOS

- [ ] use MTF as an approach for combining multimodal data
- [ ] implement hierachical modeling, perhaps with PyMC for sampling the join posterior
- [ ] implement K-fold validation
    - [ ] implement in-sample and out of sample error and do box plots with whiskers (i.e. mean the predicitons since we will have a probailistic model and for out of sample do the same with CV)
    - [ ] tune hyperparams on the mean validation error from all folds
- [ ] OPTIONAL: active learning
- [ ] model comparison with AIC and BIC (minimize to get the best)
- [ ] research meaningful statistical tests that can be used to asses the influence of certain parameters of our models
- [ ] take condition into consideration for the sign of certain parameters