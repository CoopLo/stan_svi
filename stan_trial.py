import pystan
import edward
import numpy as np
import helper_funcs as hf
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from hashlib import md5
import sys

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

#number of data points being looked at
data_points = 2500
#index of stock we want to analyze
index = 100
# load data
all_stock_data = pd.read_csv("../data/mquote201010.csv")
# get data and put it into a dict because stan wants it that way 
stock_data = np.ndarray.tolist(hf.stock_data_in_one_line(all_stock_data, index)[:data_points])
stock_data = np.reshape(stock_data, (len(stock_data), 1))
dict_data = dict({'data': stock_data})

# comes from helper funcs, copied this from somewhere online, looks good
svi_code = hf.get_model_code()

# initialize model
model = StanModel_cache(model_code = svi_code)

# pass in data and parameters
model_data = {
      'K': 5,
      'N': stock_data.shape[0],
      'T': 1,
      'y': stock_data
}

# run vb method, send output to file in this directory
results = model.vb(data = model_data,
                   output_samples = 10,
                   iter = 10000,
                   eval_elbo = 50,
                   algorithm = 'meanfield',
                   diagnostic_file = "~/Research/stan_svi",
                   sample_file = "./data.csv")


# get the results we're interested in
param_file = results['args']['sample_file'].decode("utf-8")


# read all parameters from data and process for stan sampling

# data starts at line 20, drop lp__ because data starts at second column
advi_coef = pd.read_csv(param_file, header=20).drop(["lp__"], axis=1).dropna()
advi_coef_dict = {} # get dict of results because stan sampling requires it
for key in advi_coef_dict:
    advi_coef_dict[key] = advi_coef[key].tolist() # stan sampling requires list

# add original parameters to advi_coef_dict
advi_coef_dict["K"] = 5
advi_coef_dict["N"] = stock_data.shape[0]
advi_coef_dict["T"] = 1
advi_coef_dict["y"] = stock_data

# get sample from results. Returns Stan4FitModel object
print("\n\n\nHERE\n\n\n")
fitted_model = model.sampling(data=advi_coef_dict, iter=1000)
print("\n\n\nHERE\n\n\n")

# plot it
fitted_model.plot()

plt.plot(stock_data)
plt.show()

print("YOU FUCKING DID IT")
