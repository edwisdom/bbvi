import itertools
import autograd
import autograd.numpy as np 
import matplotlib.pyplot as plt 


# Initialize training and testing data
# x_train = np.asarray([-5.0,  -2.50, 0.00, 2.50, 5.0])
# y_train = np.asarray([-4.91, -2.48, 0.05, 2.61, 5.09])
x_train = np.asarray([-2.,    -1.8,   -1.,  1.,  1.8,     2.])
y_train = np.asarray([-3.,  0.2224,    3.,  3.,  0.2224, -3.])
sigma = 0.1
mean_grid = np.linspace(-3.0, 5.0, 100)
x_grid = np.linspace(-20.0, 20.0, 200)


#=========================NEURAL NETWORKS====================#

def simple_predict(x, w, b):
	"""
	Simple linear regression model, given parameters and data, 
	returns a scalar.
	"""
	return w*x + b


def nn_predict(x, w, b, hidden_layer=10, act=np.tanh):
	"""
	Prediction function for a single-hidden layer neural network.
	Returns an array with the same size as x. 

	Arguments:
	- x: An array of inputs
	- w: An array of weights
	- b: An array of biases
	- hidden_layer: Number of neurons in hidden layer, default=10
	- act: Activation function, default=np.tanh
	"""
	h = act(np.outer(x, w[0:hidden_layer]) + b[0:hidden_layer])
	return np.dot(h, w[hidden_layer:2*hidden_layer]) + b[-1]

#---------------------------------------------------------#

#=========================VI EQUATIONS====================#

def log_prior(w, b):
	"""
	Computes the log of the prior, given two Numpy arrays of params.
	Returns a scalar.
	"""
	return (np.sum(np.square(w)) + np.sum(np.square(b)))/-2


def log_entropy(w, b, mean_w, std_w, mean_b, std_b):
	"""
	Computes the approximate posterior q, or the log entropy.
	Returns a scalar. 

	Arguments:
	- w: Weights array of size x
	- b: Biases array of size y
	- mean_w: Mean-weights array of size x
	- std_w: Standard deviation of the weights, size x
	- mean_b: Mean-biases array of size y
	- std_w: Standard deviation of the biases, size y
	"""
	q_weights = np.dot(np.square(w - mean_w), np.reciprocal(np.exp(std_w)))/-2
	q_biases = np.dot(np.square(b - mean_b), np.reciprocal(np.exp(std_b)))/-2
	return q_weights + q_biases


def log_likelihood(w, b, x=x_train, y=y_train, bnn=nn_predict):
	"""
	Computes the log likelihood given some parameters, a model, and some
	training data. Returns a scalar.

	Arguments:
	- w: Weights array
	- b: Biases array
	- x: Training observations, default=x_train
	- y: Training labels, default=y_train
	- bnn: A function that takes in x,w,b and produces an array with the
	same shape as y
	"""
	preds = bnn(x, w, b)
	return np.sum(np.square(preds - y))/(-2*sigma**2)


def loss_function(w, b, mean_w, std_w, mean_b, std_b, 
				  bnn=nn_predict, x=x_train, y=y_train):
	"""
	Computes the ELBO loss function, returns a scalar.

	Arguments:
	- w: Array of weights as a realized sample
	- b: Array of biases as a realized sample
	- mean_w: Array of mean weight values
	- std_w: Array of standard deviations of the weights
	- mean_b: Array of mean bias values
	- std_b: Array of standard deviations of the biases
	- bnn: Prediction function to use, default=nn_predict
	- x: Training observations, default=x_train
	- y: Training labels, default=y_train
	"""
	log_l = log_likelihood(w, b, x, y, bnn=bnn)
	log_p = log_prior(w, b)
	log_q = log_entropy(w, b, mean_w, std_w, mean_b, std_b)
	return float(-1 * (log_l + log_p - log_q))

#---------------------------------------------------------#

#===================MONTE CARLO ESTIMATES=================#


def sample_gaussian(mw, sw, mb, sb, samples=1):
	"""
	Samples weights and biases from given parameters, returns 2 arrays.

	Arguments:
	- mw: Array of mean weight values
	- sw: Array of standard deviations of the weights
	- mb: Array of mean bias values
	- sb: Array of standard deviations of the biases
	- samples: Number of samples, default=1
	"""
	w_vars = mw.shape[0]
	b_vars = mb.shape[0]
	return (np.random.normal(loc=mw, scale=np.exp(sw), size=(samples, w_vars)),
			np.random.normal(loc=mb, scale=np.exp(sb), size=(samples, b_vars)))


def estimate_gradient(sample_func, loss_func, grad_func, n_samples, *params):
	"""
	Estimates the gradients and losses given some parameters. Returns two 
	arrays, one with gradients per sample and the other with losses per sample.

	Arguments:
	- sample_func: Function that must return two arrays given some parameters
	and the number of samples
	- loss_func: Function that calculates the loss given a realized sample and
	the parameters
	- grad_func: Function to compute gradient of the loss, same parameters
	- n_samples: Number of samples to use, determines output array shape
	- params: A list of parameters
	"""
	sample_w, sample_b = sample_func(*params, samples=n_samples)
	sample_gradients = [np.zeros((n_samples, p.shape[0])) for p in list(params)]
	losses = np.zeros(n_samples)
	for s in range(n_samples):
		losses[s] = loss_func(sample_w[s], sample_b[s], *params)
		sample_grad = grad_func(sample_w[s], sample_b[s], *params)
		for grad, grads in zip(sample_grad, sample_gradients):
			grads[s] = grad
	return sample_gradients, losses


def update_params(params, gradients, obj, step_size):
	"""
	Updates parameters based on gradients, the objective function, and 
	the learning rate. Returns a list of arrays the same size as params.

	Arguments:
	- params: A list of arrays (parameter values)
	- gradients: A list of arrays like params, but with as many rows as 
	there were samples of the gradient
	- obj: An array of loss values with size samples
	- step_size: The learning rate, a float
	"""
	updates = [np.zeros(g.shape) for g in gradients]
	num_params, num_samples = len(params), len(gradients[0])
	for p in range(num_params):
		for s in range(num_samples):
			updates[p][s] = step_size * gradients[p][s] * obj[s]
	avg_update = [np.average(update, axis=0) for update in updates]
	new_params = [p+u for (p,u) in zip(params, avg_update)]
	return new_params

#---------------------------------------------------------#

#=======================TESTING CODE======================#


def test_p1_funcs():
	"""
	Tests that problem 1 functions do what they say they do.
	"""
	w = np.array([2])
	b = np.array([1])
	mean_w = np.array([3])
	std_w = np.array([2])
	mean_b = np.array([1.5])
	std_b = np.array([1])
	print(log_prior(w, b))
	print(log_entropy(w, b, mean_w, std_w, mean_b, std_b))
	print(log_likelihood(w, b, bnn=simple_predict))
	print(loss_function(w, b, mean_w, std_w, mean_b, std_b, bnn=simple_predict))


def test_updates():
	"""
	Tests update_params function
	"""
	params = [np.array([2,3,4]), np.array([-2,-3,-4])]
	gradients = [np.array([[1,2,3],[2,3,4]]), -1*np.array([[1,2,3],[2,3,4]])]
	obj = np.array([1, 10])
	step_size = 0.1
	new_params = update_params(params, gradients, obj, step_size)
	if (np.allclose(np.array(new_params), 
		np.array([[3.05, 4.6, 6.15], [-3.05, -4.6, -6.15]]))):
		print("Success at last. Now go to sleep.")
	else:
		print("Rekt")

#---------------------------------------------------------#

#=======================PLOTTING CODE=====================#


def plot_lines(x, lines):
    """
    Plots one figure with some x, and a bunch of y's from lines
    """
    plt.figure()
    if isinstance(lines[0], float):
    	plt.plot(x, lines, '.-')
    else:
    	for l in lines:
        	plt.plot(x, l, '.-')


def plot_posterior(preds, suffix):
	"""
	Creates two plots of the posterior, one with 10 samples, and the other
	with error bars.

	Arguments:
	- preds: Array of predictions on x_grid (samples x grid_size)
	- suffix: Title for the plot, with info on learning rate and sample number
	"""
	plot_lines(x_grid, preds[np.random.choice(len(preds), 10, replace=False)])
	plt.title("Converged Posterior Samples " + suffix)
	plt.plot(x_train, y_train, 'rx')
	plt.savefig('final_posterior_' + str(plt.gcf().number) + '.png',
                bbox_inches='tight')

	plt.figure()
	plt.title("Posterior Samples' Uncertainty " + suffix)
	plt.plot(x_train, y_train, 'rx')
	mean, std = np.average(preds, axis=0), np.std(preds, axis=0)
	plt.plot(x_grid, mean, 'k-')
	plt.gca().fill_between(x_grid.flat, mean-2*std, mean+2*std,
                       color="#dddddd")
	plt.savefig('final_uncertainty_' + str(plt.gcf().number) + '.png',
				bbox_inches='tight')


def plot_losses(losses, suffix):
	"""
	Uses plot_lines to create a plot of losses over iterations.
	"""
	plot_lines(range(len(losses[0])), losses)
	plt.title("Losses Over Iterations " + suffix)
	plt.savefig('final_losses_' + str(plt.gcf().number) + '.png',
				bbox_inches='tight')

#---------------------------------------------------------#

#=============BLACK BOX VARIATIONAL INFERENCE=============#


def p1():
	""" 
	Generates the plots for problem 1, which show estimated 
	loss and gradients for different numbers of Monte Carlo samples.
	"""
	sample_sizes = [1, 10, 100, 1000]
	std_w = np.log(0.1*np.ones(1))
	mean_b, std_b = np.zeros(1), np.log(0.1*np.ones(1))
	grad_q = autograd.grad(log_entropy, argnum=2)
	for samples in sample_sizes:
		loss_results = []
		grad_results = []
		for m in mean_grid:
			mean_w = m*np.ones(1)
			losses, grads = np.zeros(samples), np.zeros(samples)
			for s in range(samples):
				w, b = sample_gaussian(mean_w, std_w, mean_b, std_b)
				loss = loss_function(w, b, mean_w, std_w, mean_b, std_b, 
						  		 	 bnn=simple_predict)
				losses[s] = loss
				grads[s] = grad_q(w, b, mean_w, std_w, mean_b, std_b) * loss
			loss_results.append(np.average(losses))
			grad_results.append(np.average(grads))
		plot_lines(mean_grid, loss_results)
		plt.savefig('bbvi_loss_' + str((plt.gcf().number)//2 + 1) + '.png',
                    bbox_inches='tight')
		plot_lines(mean_grid, grad_results)
		plt.savefig('bbvi_grad_' + str((plt.gcf().number)//2) + '.png',
                    bbox_inches='tight')
	plt.show()


def bbvi(n_iters, step_size, grad_samples, post_samples=50):
	"""
	Performs black-box variational inference and returns the average
	loss over iterations and predictions from post_samples posteriors.

	Arguments:
	- n_iters: Number of iterations to run variational inference, int
	- step_size: Learning rate, float
	- grad_samples: Number of MC samples to take for the gradient, int
	- post_samples: Number of samples from posterior to return, int
	"""
	params = [np.random.normal(size=20), np.zeros(20),
			  np.random.normal(size=11), np.zeros(11)]
	grad_q = autograd.grad(log_entropy, argnum=[2,3,4,5])
	avg_losses = []
	for i in range(n_iters):
		gradients, losses = estimate_gradient(sample_gaussian, loss_function, 
												grad_q, grad_samples, *params)
		avg_loss = np.average(losses)
		avg_losses.append(avg_loss)
		params = update_params(params, gradients, -1*losses, step_size)
		if i % 50 == 0:
			print ("Iteration " + str(i) + "/" + str(n_iters) 
					+ " ---- Loss: " + str(avg_loss))
	w, b = sample_gaussian(*params, samples=post_samples)
	preds = [nn_predict(x_grid, w[i], b[i]) for i in range(post_samples)] 
	return avg_losses, np.array(preds)


def grid_search():
	"""
	Performs a grid search of samples and learning rates for BBVI and plots
	the results with plot_posterior.
	"""
	samples = [100, 250, 500]
	step_sizes = [5e-6, 1e-5, 5e-5]
	n_iters = 5000
	for (s, lr) in list(itertools.product(samples, step_sizes)):
		losses, preds = bbvi(n_iters, lr, s)
		suffix = ("with " + str(s) + " Samples and Learning Rate " + str(lr))
		plot_posterior(losses, preds, suffix)
	plt.show()


def p2(trials=3, n_iters=5000, samples = 150, step_size = 1e-4):
	"""
	Performs BBVI with the right parameters for problem 2 and plots results.
	"""
	trial_losses = []
	suffix = ('with' + str(samples) + "Samples and Learning Rate " 
												+ str(step_size))
	for t in range(trials):
		losses, preds = bbvi(n_iters, step_size, samples)
		trial_losses.append(losses)
		plot_posterior(preds, suffix)
	plot_losses(trial_losses, suffix)
	plt.show()

#---------------------------------------------------------#

if __name__ == '__main__':
	# p1()
	p2()



