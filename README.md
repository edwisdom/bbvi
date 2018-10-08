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
<br />
<br />
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_2.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_2.png">
Figure 2: Variational loss function and its gradient with 10 MC samples
<br />
<br />
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_3.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_3.png">
Figure 3: Variational loss function and its gradient with 100 MC samples
<br />
<br />
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_loss_4.png">
<img align="left" width="420" height="420" src="https://github.com/edwisdom/bbvi/blob/master/bbvi_grad_4.png">
Figure 4: Variational loss function and its gradient with 1000 MC samples
<br />
<br />

### Accuracy of Monte Carlo Loss Estimates

As the first columns of Figures 1-4 show, as the number of samples increase, the loss function estimate becomes smoother and more accurate. The results make sense, since the ideal value of the mean-weight parameter should be 1, and since a linear regression model should have a quadratic loss function. The 1-sample loss is noisy, but on the whole, fairly accurate in its shape. The minimum is still at 1, and overall, it still looks like a parabola, even if it's a bit noisy. 


### Accuracy of Monte Carlo Gradient Estimates

The second column provides gradients that seem reasonable given the loss function from the first column. They are mostly linear with a positive slope, which makes sense as the derivative of the parabolas of loss. Note that as the number of samples increases, like before, the gradient becomes less noisy. However, in this case, the 1-sample estimates are not really accurate, especially as we deviate far from the optimal value of the mean. Moreover, the basic upward linear slope is not even preserved. This remains true for 10 Monte Carlo samples, though once we take 100 or 1000 MC samples, the gradient becomes more clearly correct. Curiously, even 1000 MC samples produces somewhat noisy gradients at mean values that are far from optimal.


## Hyperparameter Tuning

<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_1.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_7.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_13.png">
<br />
<br />
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_2.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_8.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_14.png">
<br />
<br />
<br />
<br />
Figure 5: Losses and Fitted Posteriors for Learning Rate 5e-6
<br />
<br />
<br />
<br />
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_3.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_9.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_15.png">
<br />
<br />
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_4.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_10.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_16.png">
<br />
<br />
<br />
<br />
Figure 6: Losses and Fitted Posteriors for Learning Rate 1e-5
<br />
<br />
<br />
<br />
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_5.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_11.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/loss_over_iters_17.png">
<br />
<br />
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_6.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_12.png">
<img align="left" width="275" height="300" src="https://github.com/edwisdom/bbvi/blob/master/fitted_posterior_18.png">
<br />
<br />
<br />
<br />
Figure 7: Losses and Fitted Posteriors for Learning Rate 5e-5
<br />
<br />
<br />
<br />
My approach was a little brute-force, as I tried a number of combinations of learning rates and MC samples. I found that the higher learning rate, 5e-5, produced the best results, and as expected, more samples also improved performance. Of course, taking more samples incurs a higher computational cost, so I limited that number to 500. I stuck with these two values for the hyperparameters for the following experiments.

## Results of Black-Box Variational Inference 

<img align="left" width="700" height="500" src="https://github.com/edwisdom/bbvi/blob/master/final_losses_7.png">
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
Figure 8: Estimated loss function over iterations for 3 chains of variational inference
<br />
<br />
<br />
<br />
<img align="left" width="425" height="425" src="https://github.com/edwisdom/bbvi/blob/master/final_posterior_1.png">
<img align="left" width="425" height="425" src="https://github.com/edwisdom/bbvi/blob/master/final_uncertainty_2.png">
<img align="left" width="425" height="425" src="https://github.com/edwisdom/bbvi/blob/master/final_posterior_3.png">
<img align="left" width="425" height="425" src="https://github.com/edwisdom/bbvi/blob/master/final_uncertainty_4.png">
<img align="left" width="425" height="425" src="https://github.com/edwisdom/bbvi/blob/master/final_posterior_5.png">
<img align="left" width="425" height="425" src="https://github.com/edwisdom/bbvi/blob/master/final_uncertainty_6.png">
<br />
<br />
<br />
<br />
Figure 9: 10 samples from the posterior and a plot of uncertainty (2 std.) for 3 VI chains with learning rate 5e-5 and 500 gradient samples
<br />
<br />
<br />
<br />

### Comparison to Hamiltonian Monte Carlo

Compared to the [Hamiltonian Monte Carlo technique](https://github.com/edwisdom/bnn-hmc), variational inference does not fit the data as well. This is somewhat strange, especially considering we're approximating normally distributed data, so our choice of a normal distribution as our density family should make this easier. Moreover, the results are very unstable, in a way that they were not for Hamiltonian Monte Carlo, where the right choice of leapfrog steps and learning rate would usually guarantee a good posterior sample. Here, even with the right hyperparameters, a bad initialization can severely hamper variational inference. 


The value added with black-box variational inference is that it can be used with any complex model we want, even ones that aren't differentiable. This isn't, of course, possible with Hamiltonian Monte Carlo. However, these results make me skeptical of the claim that black-box variational inference can practically, not theoretically, be used with any model. With how noisy its gradient estimates are, my guess is that models more complex than our single-hidden layer neural network might make it necessary to take a lot more Monte Carlo gradient samples to make variational inference work. 

## Future Work

In the future, I would like to explore the following:

1. Varying the learning rates using the Robbins-Munro sequence
2. Applying this model to real-world data and comparing it to neural networks that take similar time to train
3. Using a different density family (Gaussian scale mixtures) for the approximating distribution

## Credits

A huge thanks to Prof. Michael Hughes, who supervised this work, Daniel Dinjian, who thought through architectures with me in the early phases, and Ramtin Hosseini, who sat with me to think about bugs that could lead to small gradient norms. 