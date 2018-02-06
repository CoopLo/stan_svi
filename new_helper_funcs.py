import numpy as np
import pandas as pd

def _get_stock_symbols(stock_data):
    '''takes in stock data, returns numpy array of stock names'''
    stock_names = []
    for i in stock_data.values:
        if(i[0] not in stock_names):
            stock_names.append(i[0])
    return stock_names

def particular_stock_data(stock_data, stock_names_array, index):
    '''in stock data, stock name array and index of the stock
        returns a pandas DataFrams of the specified stock's entire
        monthly data'''
    start_index = 21 * index
    particular_stock_data = pd.DataFrame(stock_data, 
                            index=[stock_data.axes[0][start_index:start_index+21]],
                            columns=stock_data.axes[1])
    return particular_stock_data


# needs to be a 2D dataframe. It is treated as
# if it is one stock's data for the month. Passing anything else
# in will likely output nonsense. It returns a 1D numppy array of all of
# the stock's monthly data.
def stock_data_in_one_line(stock_data, index):
    ''' takes in stock data, stock names array and index of the stock
        returns the stock's monthly data in a 1D numpy array'''
    stock_names = _get_stock_symbols(stock_data)
    particular_stock = particular_stock_data(stock_data, stock_names, index)
    single_line_data = []
    for i in range(0, particular_stock.axes[0].size):
        single_line_data.append(particular_stock.values[i][3:])
    return np.ravel(single_line_data)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

# create 3D data set where each vector is <(close-open)/open, (high-open)/open, (open-low)/open>
# over n minute intervals. This simulates open, close, high and low for days.
def create_3D_dataset(stock_data, interval):
    new_data = []
    for i in range(0, len(stock_data), interval):
        inter_open = stock_data[i]
        inter_close = stock_data[i+interval-1]
        low = min(stock_data[i:i+interval-1])
        high = max(stock_data[i:i+interval-1])

        new_data.append(((inter_close-inter_open)/inter_open, (high-inter_open)/inter_open,
                         (inter_open - low)/inter_open))

    return np.asarray(new_data)

def get_model_code():
	return """
	data {
		int<lower=1> K; // number of groups
		int<lower=1> N; // number of data points
		int<lower=1> T; // length of timeseries
		real y[N,T]; // observations
        real trans_mat[K,K]; // transition matrix
        real p[N]; // initial distribution
        real emit_mat[K,T]; // emission matrix
	}
	parameters {

  	int i;

  	//Forwards algorithm
  	for (n in 1:N) { F[n,1] = p[i]*emit_mat[n,1];}
                   
  	for (t in 1:T){
    	for (n in 1:N) {
            F[n,t] = F[n,t-1] * trans_mat[] * B[];
		}
	} 
	//backwards algorithm
  	for (n in 1:N) { 
        B[n,T] = 1;
		//B1[n,T] = 1; 
     	//B2[n,T] = 1; 
  	}
  	for (t in 1:(T-1)){
    	i = T - t;      // transform t to get a backwards loop
    	for (n in 1:N){
            B[n,t] = trans_mat[] * emit_mat[] * B[n,t+1];
    	}
  	}
}
model {

	for (t in 1:T){
  		for (n in 1:N) {
        	ps  =  pred[n,t]*exp(normal_lpdf(y[n,t]|mu[1],sigma[1]))+
           	(1-pred[n,t])*exp(normal_lpdf(y[n,t]|mu[2],sigma[2]));
      	increment_log_prob(log(ps));
    	}
  	}
}
"""
