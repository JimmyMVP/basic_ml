import numpy as np 
import scipy


np.random.seed(666)

DATASET_SIZE = 1000
CLASSES = np.arange(0,2)
INPUTS = np.arange(0,5)

print("Possible classes: ", CLASSES)

#Create random dataset
X = np.random.choice(INPUTS,DATASET_SIZE)
Y = np.random.choice(CLASSES, DATASET_SIZE)

#Calculate prior
prior = np.zeros(CLASSES.shape, dtype=np.float32)
for y in Y:
    prior[y]+=1
prior /= Y.shape
log_prior = np.log(prior)
print("Calculated prior: ", prior)
print("Log prior: ", log_prior)

#Calculate likelihood P(X|Y)
likelihood = np.zeros((INPUTS.shape[0], CLASSES.shape[0]))
for i in range(DATASET_SIZE):
    likelihood[X[i],Y[i]] += 1.

#Normalising the joint distribution
for c in CLASSES:
    likelihood[:,c] /= likelihood[:,c].sum()
for i in INPUTS:
    likelihood[i, :] /= likelihood[i,:].sum()

print("Joint distribution: ")
print(likelihood)

#Calculate the likelihood
for c in CLASSES:
    likelihood[:, c] /= prior[c]
log_likelihood = np.log(likelihood)

print("Calculated likelihood: ")
print(likelihood)
print("Log-likelihood: ")
print(log_likelihood)

def maximum_aposteriori(x, log_likelihood,  log_prior, classes):
    prob_x = np.zeros(classes.shape)
    for c in classes:
        prob_x[c] = log_likelihood[x, c] + log_prior[c]

    return prob_x


print("Example likelihood for x=2: ", maximum_aposteriori(2, log_likelihood, log_prior, CLASSES))










