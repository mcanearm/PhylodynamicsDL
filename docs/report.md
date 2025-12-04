# Statistical Efficiency of Birth-Death Tree Parameter Estimation

Source code available at:

PhylodynamicsDL: https://github.com/mcanearm/PhylodynamicsDL
PhyloDeepPOMP: https://github.com/Horopter/PhyloDeepPOMP


## Intro

The Phylodeep package implements a deep learning approach to estimating parameters of several different
tree models. For our study, we focused on the Birth-Death trees and compared the statistical efficiency of
the Phylodeep approach to the traditional Maximum Likelihood Estimation (MLE) approach. Initial comparisons 
indicated that there may be some issues with the likelihood estimation component, but we think the results
are interesting enough to present, albeit with caveats.[^1]

[^1]: “evolbioinfo/treesimulator” (2025), Python, Evolutionary Bioinformatics.


## Data Generation

The original paper by Voznicka et. al. utilized a custom simulator to generate trees using a package by Zhukova et. al.,
one of the other authors of the Voznicka paper.

## Methods

### MLE

We made a few attemps at implementing the MLE, which ended up being a more challenging part of the project
than we expected, mostly due to difficulties in dealing with the data. 

URVI - put in what you did here and results / final outcome



### Phylodeep

SANTOSH - model fitting process

## Results



## Conclusion