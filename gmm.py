import numpy as np
from sklearn.mixture import GaussianMixture

# Generate Synthetic data
mu1=300
sig1=100

mu2=1100
sig2=100

mu3=2000
sig3=200
x = list(np.random.normal(mu1,sig1,500))+ list(np.random.normal(mu2,sig2,500))+list(np.random.normal(mu3,sig3,500))

# Fit GMM
model = GaussianMixture(n_components=3, max_iter=2000)
model.fit(np.array(x).reshape(-1,1))
print(sorted(zip(model.means_, np.sqrt(model.covariances_))))