import pystan
import numpy as np
import new_helper_funcs as hf
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hashlib import md5
import sys   # used for halting script in debugging
from sklearn.cluster import KMeans

def StanModel_cache(model_code, model_name=None, **kwargs):
	"""Use just as you would 'stan'"""
	code_hash = md5(model_code.encode('ascii')).hexdigest()
	if model_name is None:
		cache_fn = 'cached-model-{}.pkl'.format(model_name, code_hash)
	else:
		cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
	try:
		sm = pickle.load(open(cache_fn, 'rb'))
	except:
		sm = pystan.StanModel(model_code=model_code)
		with open(cache_fn, 'wb') as f:
			pickle.dump(sm, f)
	else:
		print("Using cached StanModel")
	return sm

"""
This section imports and cleans stock data
"""

#number of data points being looked at
data_points = 8000
#index of stock we want to analyze
index = 1
# load data
all_stock_data = pd.read_csv("../data/mquote201010.csv")
# get data and put it into a dict because stan wants it that way 
stock_data = np.ndarray.tolist(hf.stock_data_in_one_line(all_stock_data, index)[:data_points])
stock_data_3D = hf.create_3D_dataset(stock_data, 5)

#stock_data = np.reshape(stock_data, (len(stock_data), 1))
dict_data = dict({'data': stock_data_3D})

"""
Here we have model specifications
"""
num_clusters = 5
num_states = 4


"""
This section uses k-means clustering to compute necessary values for emission matrix
in the HMM. I can't find anything online on what the weights shold be so I'm just doing
how ofter each cluster shows up/total number of data points
"""
k_means = KMeans(n_clusters = num_clusters)
predict = k_means.fit_predict(stock_data_3D)
_, weights = np.unique(predict, return_counts=True)
weights = np.divide(weights, sum(weights))
c = np.divide(weights, num_states)

# THIS IS PROBABLY WRONG
# uses probability of finding point in mixture component as 1 for component predict()
# returns and 0 else
B = np.zeros((4, stock_data_3D.shape[0]))
for i in range(stock_data_3D.shape[0]):
    for j in range(4):
        B[j][i] = c[predict[i]]

print("got to where I wanted")
#sys.exit(0)

# Here we create the prior distribution and transition matrix
# these are initialized to be uniform
p = [0.25, 0.25, 0.25, 0.25]
A = [[0.25, 0.25, 0.25, 0.25],
     [0.25, 0.25, 0.25, 0.25],
     [0.25, 0.25, 0.25, 0.25],
     [0.25, 0.25, 0.25, 0.25]]

#sys.exit(0)

"""
Do the stan setup and vb method here. Then extrac predicted value from model
"""

# comes from helper funcs, copied this from somewhere online, looks good
svi_code = hf.get_model_code()

# initialize model
model = StanModel_cache(model_code = svi_code)

# pass in data and parameters
model_data = {
      'K': num_states,
      'N': 3,
      'T': stock_data_3D.shape[0],
      'y': stock_data_3D.T,
      'A': A,
      'p': p
}

# run vb method, send output to file in this directory
results = model.vb(data = model_data,
                   output_samples = 10,
                   iter = 10000,
                   eval_elbo = 50,
                   algorithm = 'meanfield',
                   diagnostic_file = "~/Research/stan_svi",
                   sample_file = "./data.csv")

