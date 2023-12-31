# calculate probability that one seed will germinate,
# given that the seed is peak from below
# type1 : 0.5kg, 90% germination rate
# type2: 1.5kg, 80% germination rate
# type3: 3kg, 60% germination rate

# Bayes theorem
# P_germinate = P(type1 and germinate) + P(type2 and germinate) + P(type3 and germinate)
# P(A|B) = P(B|A)*P(A) / P(B)
# => P(A_specific|B) = P(B\A_specific)*P(A_specific) / 
# (sum_over_all_A(P(B|A)*P(A)))
# (posterior = prior * likelihood / marginal)

import numpy as np
def marginal(likelihoods, priors):
    return np.dot(likelihoods, priors)

def bayes(likelihoods, priors, marginal, index):
    res = priors[index] * likelihoods[index] / marginal
    return res

w = 0.5+1.5+3
likelihoods = [0.9, 0.8, 0.6] # probability of germination given each type
priors = [0.5/w, 1.5/w, 3/w] # probability of each type

# calculate the probability that one seed will germinate
m = marginal(likelihoods, priors)
print(m)

# calculate the probability that the germinated seed is type 1
b1 = bayes(likelihoods, priors, m, 0)
print(b1)