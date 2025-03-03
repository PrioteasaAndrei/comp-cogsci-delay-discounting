This is the final project for the Computational Cognitive Science class, WS24/25, University of Heidelberg.

This project investigates probabilistic discounting in decision-making through hierarchical Bayesian modeling. We analyze data from a behavioral experiment where participants ($N=49$, mean trials per participant $\approx 199$) chose between certain and uncertain rewards under varying probability conditions. Using PyMC, we implemented a three-level hierarchical model (group, participant, and trial) incorporating both hyperbolic discounting and loss aversion components. The model predicts choice behavior and reaction times, with decision difficulty modeled as the absolute value difference between subjective values of options.

Key innovations include: (1) a loss aversion parameter that improved choice prediction accuracy from 0.55 to 0.60, capturing asymmetric processing of gains versus losses; (2) standardization techniques that resolved numerical instabilities in parameter estimation; and (3) comparison of linear versus quadratic reaction time models using BIC and AIC metrics. While the model successfully captured individual differences in choice behavior, reaction time predictions showed limited trial-to-trial variability, suggesting potential areas for future refinement. The project demonstrates the utility of hierarchical Bayesian approaches in understanding individual differences in probabilistic decision-making while highlighting the importance of careful parameter specification and data preprocessing in computational modeling.

# Setup

```bash
chmod +x setup.sh
./setup.sh
conda activate pd-ws24
```
