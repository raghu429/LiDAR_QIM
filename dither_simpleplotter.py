# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:39:09 2020

@author: rchangs
"""
import numpy as np

#a = np.array([37.4551159459,37.5055158001,37.5349788383,37.3696971695,37.6008373505,37.2537653995,37.4089572104,36.3677316207
#    20.346391086,13.9822831935,	7.1559275818,	2.2718176702,	0.3006252772,	0.0087592977,	0,	0
#    5.1073594607,0.8256608364,	0.0257669463,	0,	0,	0,	0,	0
#    0, 	0,	0,	0,	0,	0,	0,	0])

# example of parametric probability density estimation
from matplotlib import pyplot
from numpy.random import normal
from numpy import mean
from numpy import std
from scipy.stats import norm
# generate a sample
sample = normal(loc=50, scale=5, size=100)
# calculate parameters
sample_mean = mean(sample)
sample_std = std(sample)
print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
# define the distribution
dist = norm(sample_mean, sample_std)
# sample probabilities for a range of outcomes
values = [value for value in range(30, 70)]
probabilities = [dist.pdf(value) for value in values]
# plot the histogram and pdf
pyplot.hist(sample, bins=10, density=True)
pyplot.plot(values, probabilities)
pyplot.show()