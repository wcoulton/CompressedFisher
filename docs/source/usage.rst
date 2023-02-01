Usage
=====


.. _installation:
Installation
------------

To use CompressedFisher, first install it using pip:

.. code-block:: console

    $ pip install CompressedFisher

It requires the *numpy* and *scipy* packages.

Basic Usage
------------

Typical usage of the code requires two ensembles of simulations: one set of simulations is given at the fiducial parameters ( :math:`\theta` ) and is used to estimate the covariance matrix. The second is a set of simulated derivatives; these can either be in the form of realizations of the derivatives themselves or simulations evaluate at a set of point in the neighborhood of the fiducial point that the code can use to estimate the derivatives (e.g. simulations a parameter points :math:`\theta+\delta \theta_i` and  :math:`\theta-\delta \theta_i$`  where  :math:`\delta \theta_i` is a small step in parameter,  :math:`i` .)


Here we sketch a potential workflow, with detailed examples available in the notebooks described in the :ref:`examples <examples>` section.

# Choose the appropriate class based on the distribution of the data. Currently supported cases are:
  * gaussianFisher
  * poissonFisher

# Provide the code with the two sets of simulations (one for the covariance, rates etc and the second set for the derivatives)

# Choose a division of the simulations between the compression and Fisher estimation steps (1/2 in each typical works well)

# Call the *compute_fisher_forecast*, *compute_compressed_fisher_forecast* and *compute_combined_fisher_forecast* to compute standard, compressed and combined Fisher forecasts.

# There are a range of methods to assess whether these forecasts are converged (including *est_fisher_forecast_bias* and *run_fisher_deriv_stablity_test*). It is important to perform tests like these (and more) to ensure that your forecast constraints are converged. If they are not your parameter inferences will likely overestimate your ability to constrain that parameter. 

stnd_constraint    = cFisher.compute_fisher_forecast(parameter_names)
stnd_constraint_bias = cFisher.est_fisher_forecast_bias(parameter_names)


compressed_constraint = cFisher.compute_compressed_fisher_forecast(parameter_names)
compressed_constraint_bias = cFisher.est_compressed_fisher_forecast_bias(parameter_names)


combined_constraint = cFisher.compute_combined_fisher_forecast(parameter_names)
  

:: _examples
Examples
------------

In the *examples* folder of the repository are three examples showing how to used the code.
Each considers a different test case: a Gaussian likelihood with a parameter independent mean,  a Gaussian likelihood with a parameter dependent mean, and a Poisson case.

These examples show how to use the main functionality of the code. Further details on each method can be found in :ref:`api <api>` 