# Black-Box Variational Inference on Bayesian Neural Networks

## Getting Started

These instructions will allow you to run this project on your local machine.

### Install Requirements

Once you have a virtual environment in Python, you can simply install necessary packages with: `pip install -r requirements.txt`

### Clone This Repository

```
git clone https://github.com/edwisdom/bbvi
```

### Run Models

Run the models with: 

```
python bbvi.py
```


## Monte Carlo Estimations Using Score Function Trick


<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_1.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_1.png">
Figure 1: Variational loss function and its gradient with 1 MC sample
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_2.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_2.png">
Figure 2: Variational loss function and its gradient with 10 MC samples
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_3.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_3.png">
Figure 3: Variational loss function and its gradient with 100 MC samples
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_4.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_4.png">
Figure 4: Variational loss function and its gradient with 1000 MC samples


### Accuracy of Monte Carlo Loss Estimates

As the first columns of Figures 1-4 show, as the number of samples increase, the loss function estimate becomes smoother and more accurate. The results make sense, since the ideal value of the mean-weight parameter should be 1, and since a linear regression model should have a quadratic loss function. The 1-sample loss is noisy, but on the whole, fairly accurate in its shape. The minimum is still at 1, and overall, it still looks like a parabola, even if it's a bit noisy. 

### Accuracy of Monte Carlo Gradient Estimates

The second column provides gradients that seem reasonable given the loss function from the first column. They are mostly linear with a positive slope, which makes sense as the derivative of the parabolas of loss. Note that as the number of samples increases, like before, the gradient becomes less noisy. However, in this case, the 1-sample estimates are not really accurate, especially as we deviate far from the optimal value of the mean. Moreover, the basic upward linear slope is not even preserved. This remains true for 10 Monte Carlo samples, though once we take 100 or 1000 MC samples, the gradient becomes more clearly correct. Curiously, even 1000 MC samples produces somewhat noisy gradients at mean values that are far from optimal.

<!-- ### Qualitative Trends in the Prior

The samples from the prior, as Figure 1 shows, clearly depend heavily on the choice of activation functions. Whereas the relu prior samples are essentially two piecewise lines, the tanh prior samples look like sigmoid curves (which reflects the underlying activation functions). 

Although these activation functions are simple, even with just one hidden layer of 2 units, different priors have a large range at each possible input value (except the relu activation at x=0, predictably so). This suggests that our priors are fairly flexible, and can fit a lot of different training data. Another clear trend is that with more hidden layers, individual prior samples become more complex with more local extrema. 

## Sampling from a Bayesian Neural Net Posterior with Hamiltonian Monte Carlo

### Convergence

<img align="left" width="600" height="600" src="https://github.com/edwisdom/bnn-hmc/blob/master/potential_energies.png">

**Figure 5**: Potential energies over Hamiltonian Monte Carlo iterations for 3 different chains

As we see in Figure 5, each of the chains of Hamiltonian Monte Carlo rapidly converge to low potential energy values, which means that the samples we're getting are from high probability-mass regions of the posterior. This is the primary reason why Hamiltonian Monte Carlo is preferred over Metropolis-style (MCMC) methods, since the latter is unlikely to explore a wide space while still remaining in high-probability regions. 

To see an interactive demo of this principle, [see here](https://chi-feng.github.io/mcmc-demo/app.html#HamiltonianMC,banana).

<br />
<br />
<br />
<br />
<br />

### Posterior Shapes

<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_posterior_1.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_posterior_1.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bnn-hmc/blob/master/bnn_posterior_1.png">

Figure 6: 10 samples from 3 different BNN posteriors sampled using HMC with epsilon=0.001 and L=25


As we can see in Figure 6, each of the posterior samples wraps tightly around the training data. However, the predictions that are significantly further away from any training data are much more variable. 


### Uncertainty and the Deficiency of Point Estimates

<img align="left" width="600" height="600" src="https://github.com/edwisdom/bnn-hmc/blob/master/posterior_samples.png">

**Figure 7**: 500 posterior function samples from a single BNN posterior trained with epsilon=0.001 and L=25

Figure 7 shows how our posterior has much greater uncertainty at input points that are far away from its training data. Its estimates of these values are largely dominated by the prior. This kind of model gives us an edge over traditional point-estimate neural networks because they give a distribution over our parameters and allow us to quantify our certainty about predictions. These models have the potential to be both more interpretable and more capable of [detecting adversarial perturbations](https://arxiv.org/abs/1711.08244).

<br />
<br />
<br />
<br />
<br />
<br />
<br />


## Future Work

In the future, I would like to explore the following:

1. Tuning hyperparameters epsilon and L more exhaustively and systematically
2. Applying this model to real-world data and comparing it to neural networks that take similar time to train
3. Implementing the [NUTS](https://arxiv.org/abs/1111.4246) (No U-Turn Sampler), which is currently the best known Monte Carlo sampling technique for Bayesian neural nets

## Credits

A huge thanks to Prof. Michael Hughes, who supervised this work, and Daniel Dinjian and Julie Jiang for thinking through the technical nitty-gritty with me. -->