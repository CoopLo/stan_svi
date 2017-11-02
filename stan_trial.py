import pystan
import edward
import numpy as np
import helper_funcs as hf
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from hashlib import md5

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
data_points = 8000
#index of stock we want to analyze
index = 100
# load data
all_stock_data = pd.read_csv("../data/mquote201010.csv")
# get data and put it into a dict because stan wants it that way 
stock_data = np.ndarray.tolist(hf.stock_data_in_one_line(all_stock_data, index)[:data_points])
dict_data = dict({'data': stock_data})


# comes from stan
model_code = hf.get_model_code()
model = StanModel_cache(model_code)
results = model.vb(dict_data)
sample_data = model.sampling(dict_data)
model_data = sample_data.extract()['y']

plt.subplot(211)
plt.plot(stock_data)
plt.subplot(212)
plt.plot(model_data)
plt.show()
