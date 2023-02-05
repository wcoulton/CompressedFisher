

import numpy as np 

import CompressedFisher

"""
This uses the same toy model in the examples notebook.
This model can be evaluated analytically providing tests of our numerical methods.
"""


# Define number and location of points
nKs = 100
ks = 10**np.linspace(-4,0,nKs)
params_fid = np.array([1,1,1.])
parameter_names = [r"$\alpha$", r"$\beta$", r"$\gamma$"]

delta_params = np.array([.1,.1,.1,])

def muk_theory(ks,params):
    """
    Evalute the toy model for the mean of the Gaussian at a set of positiosn
    
    Args:
        ks ([array of floats]): ks at which to evaluate the model
        params ([array of floats]): The values of the parameters to 
    Returns:
        [array of shape [nks]]: The mean at the requested values 
    """

    alpha,beta,gamma= params
    return alpha+beta*ks+gamma*ks**.5
def generateCovMat(params):
    """
    Construct the covariance matrix given as:
    Cov (x_i,x_j) = \\delta_{i,j} 2 \\mu(x_i)**2
    Args:
        params ([list floats]): The values of the parameters to evaluate the covariance matrix at
        
    Returns:
        [array (nKs,nKs) ]: The covariance matrix
    """
    mu_k = muk_theory(ks,params)*.1
    return np.diag(2*mu_k**2)


def generateSims(params,nSims=1,params_covMat= np.array([1,1,1.])):
    """
    Generate draw realizations from the toy model with the given set of parameters
    
    Args:
        params ([list floats]): The values of the parameters to draw realizations at.
        nSims (ind): The number of sims to generate (default: `1`)
        params_covMat (array) :  The parameters at which to evaluate the covariance matrix at
    Returns:
        [array (nSims,nKs) ]: The Gaussian simulations
    """
    mu_k = muk_theory(ks,params)
    mu_k_true = muk_theory(ks,params_covMat)
    return  mu_k+np.random.randn(nSims,nKs)*np.sqrt(2)*mu_k_true*.1



def generate_deriv_sims(paramter_names,params,deltaParams,nSims):
    """
    Generates simulations for use as finite difference derivatives.
    These are central finite differences, ths for each parameter 
    simulations are generated at $\\theta\\pm \\delta\\theta$.
    
    Args:
        parameter_names ([list of names]): the names of the parameters
        params ([array]): the values of the parameters about which to compute the derivatives
        deltaParams ([array]): The parameter step size
        nSims ([int]): The number of simulations
    
    Returns:
        [dictionary]: The simulations for computing the derivates.  Each entry in the dicitonary 
                      is an array of size [2,nSims,nKs]. 
                    
    """
    deriv_sims = {}
    for i,name in enumerate(paramter_names):
        derivs_ens = np.zeros([2,nSims,nKs])
        params_plus = np.array(params)
        params_plus[i]+=deltaParams[i]
        
        params_minus = np.array(params)
        params_minus[i]-=deltaParams[i]
          
        derivs_ens[0] = generateSims(params_minus,nSims)
        derivs_ens[1] = generateSims(params_plus,nSims)
        deriv_sims[name] = derivs_ens
    return deriv_sims


"""
For this toy model we can compute the true Fisher information analytically.
""" 

def theoretical_derivs(params):    
    """
    Computes the derivatives analytical
    
    Args:
        params ([array]): The values where the derivatives are to be evaluated
    
    Returns:
         an array of shape [3,nKs] : The derivatives for the 3 parameters
    """
    #mu_k = muk_theory(ks,params)
    nParams = len(params)
    derivs_thry = np.zeros([nParams,nKs])
    derivs_thry[0] = np.ones(nKs)
    derivs_thry[1] = ks.copy()
    derivs_thry[2] = np.sqrt(ks)
    return derivs_thry

def fisherInformationTheory(params):
    """
    Compute the exact Fisher informations
    
    Args:
        params ([array]): The values where the Fisher information are to be evaluated
    
    Returns:
        [matrix nParams x nParams]: The analytical Fisher information (i.e. the `truth')
    """
    derivs = theoretical_derivs(params)
    covMat = generateCovMat(params)
    nParams = len(params)
    fisher = np.zeros([nParams,nParams])
    for i in range(nParams):
        for j in range(i+1):
            fisher[i,j] = fisher[j,i] = np.dot(derivs[i], np.linalg.solve(covMat,derivs[j]))
    return fisher



def generate_analytic_deriv_sims(parameter_names,params,deltaParams,nSims):
    """"
    A function to create mock finite difference sims. These 'sims' are just given at the theoretical values (without MC noise).
    T
    Args:
        parameter_names ([list]): A list of the parameters to generate sims for
        params ([dicitonary]): The values of the parameters to evaluate the central difference sims about
        deltaParams ([list]):  The parameter step sizes for the finite differences.
        nSims ([int]): The number of mock sims to generate
    
    Returns:
        [dictionary]: A dictionary containing the mock derivatives
    """
    deriv_sims = {}
    
    
    for i,name in enumerate(parameter_names):
        derivs_ens = np.ones([2,nSims,nKs])
        params_plus = np.array(params)
        params_plus[i]+=deltaParams[i]
        
        params_minus = np.array(params)
        params_minus[i]-=deltaParams[i]
        derivs_ens[0] = muk_theory(ks,params_minus)
        derivs_ens[1] = muk_theory(ks,params_plus)
        deriv_sims[name] = derivs_ens
    return deriv_sims


def analyticalCompression(data,params):
    """
    Compress the data vector using the analytically evaluated compression
    Args:
        data ([...,nKs]): The data to compress
        params ([vector (nParams)]): The parameters to evaluate the compression at
    
    Returns:
        [type [...., nParams]]: The compressed vector
    """
    def compression(data,mu,deriv,covMat):
        results = np.zeros([data.shape[0],deriv.shape[0]])
        for i in range(deriv.shape[0]):
            results[:,i]= deriv[i].dot(np.linalg.solve(covMat,(data).T))
        return results

    pk_mu = muk_theory(ks,params)
    derivs_thry = theoretical_derivs(params)
    covMat = generateCovMat(params)
    return compression(data,pk_mu,derivs_thry,covMat)


def initializeFisher():
    """
    Set up the Fisher object. This also initializes the cov mat. 

    Returns:
        [gaussianFisher object]: A gaussianFisher object for use in the null tests
    """

   
    dict_param_steps = {parameter_names[i]:delta_params[i] for i in range(len(params_fid))}
   # nSimsCovMat = 2000
    nSims_deriv = 100

    cFisher = CompressedFisher.gaussianFisher(parameter_names,nSims_deriv,deriv_finite_dif_accuracy=2)
    cFisher._covmat_fisher = generateCovMat(params_fid)
    cFisher._covmat_comp = generateCovMat(params_fid)
    dict_deriv_sims = generate_analytic_deriv_sims(parameter_names,params_fid,delta_params,nSims=nSims_deriv)
    cFisher.initailize_deriv_sims(dic_deriv_sims=dict_deriv_sims,dict_param_steps=dict_param_steps)
    cFisher.generate_deriv_sim_splits(.5)
    return cFisher


def test_standardFisherMatrix():
    """
    Test standard Fisher matrix calculation
    """
    
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    fisher_cFish = cFisher._compute_fisher_matrix(parameter_names)
    assert(np.all(np.isclose(fisher_true,fisher_cFish)))

def test_standardFisherForecast():
    """
    Test standard Fisher forecast calculation
    """
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    const_cFish = cFisher.compute_fisher_forecast(parameter_names)
    np.linalg.inv(fisher_true),const_cFish
    assert(np.all(np.isclose(np.linalg.inv(fisher_true),const_cFish)))


def test_compression():
    """
    Test the compression step
    """
    cFisher =  initializeFisher()
    test_data = generateSims(params_fid,10)
    comp_analytic = analyticalCompression(test_data,params_fid)
    comp_cFish = cFisher.compress_vector(test_data,with_mean=False)
    assert(np.all(np.isclose(comp_cFish,comp_analytic)))


def test_compresedFisherMatrix():
    """
    Test the compressed fisher matrix calculation
    """
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    fish_comp_cFish = cFisher._compute_compressed_fisher_matrix(parameter_names)
    assert(np.all(np.isclose(fisher_true,fish_comp_cFish)))


def test_compresedFisherForecast():
    """
    Test the compressed fisher forecast calculation
    """
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    const_comp_cFish = cFisher.compute_compressed_fisher_forecast(parameter_names)
    assert(np.all(np.isclose(np.linalg.inv(fisher_true),const_comp_cFish)))


def test_combinedFisherForecast():
    """
    Test the combined fisher forecast calculation
    """
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    const_combined_cFish = cFisher.compute_combined_fisher_forecast(parameter_names)
    assert(np.all(np.isclose(np.linalg.inv(fisher_true),const_combined_cFish)))


def test_compresedFisherForecast_wShuffle():
    """
    Test the compressed fisher forecast calculation
    """
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    const_comp_cFish = cFisher.compute_compressed_fisher_forecast_wShuffle(parameter_names,.5)
    assert(np.all(np.isclose(np.linalg.inv(fisher_true),const_comp_cFish)))


def test_combinedFisherForecast_wShuffle():
    """
    Test the combined fisher forecast calculation
    """
    cFisher =  initializeFisher()
    fisher_true = fisherInformationTheory(params_fid)
    const_combined_cFish = cFisher.compute_combined_fisher_forecast_wShuffle(parameter_names,.5)
    assert(np.all(np.isclose(np.linalg.inv(fisher_true),const_combined_cFish)))


def test_id_splitting():
    """
    Verify that the ids are correctly split into two disjoint sets
    """
    cFisher =  initializeFisher()
    ids_a,ids_b=cFisher._get_deriv_split_indices(.5)
    ids_0 = np.random.choice(ids_a)
    assert(ids_0 not in ids_b)
    ids_0 = np.random.choice(ids_b)
    assert(ids_0 not in ids_a)



def test_FisherBias():
    """
    Verify that the standard Fisher bias is correctly estimated
    """
    nSimsCovMat = 20000
    nSims_deriv = 1000

    covmat_sims = generateSims(params_fid,nSimsCovMat)

    fisher_true = fisherInformationTheory(params_fid)
    dict_param_steps = {parameter_names[i]:delta_params[i] for i in range(len(params_fid))}
    dict_deriv_sims = generate_deriv_sims(parameter_names,params_fid,delta_params,nSims=nSims_deriv)
    cFisher = CompressedFisher.gaussianFisher(parameter_names,nSims_deriv,include_covmat_param_depedence=False,deriv_finite_dif_accuracy=2)
    cFisher.initailize_covmat(covmat_sims,store_covmat_sims=True)
    cFisher.initailize_mean(covmat_sims)
    cFisher.initailize_deriv_sims(dic_deriv_sims=dict_deriv_sims,dict_param_steps=dict_param_steps)
    cFisher.generate_deriv_sim_splits(.5)
    cFisher.generate_covmat_sim_splits(.5) # Test division of cov mat sims too..
    np.isclose(fisher_true,cFisher._compute_fisher_matrix(parameter_names)-cFisher._compute_fisher_matrix_error(parameter_names),rtol=2/np.sqrt(.5*nSims_deriv))


def test_compressedFisherBias():
    """
    Verify that the compressed Fisher bias is correctly estimated
    """
    nSimsCovMat = 20000
    nSims_deriv = 1000

    covmat_sims = generateSims(params_fid,nSimsCovMat)

    fisher_true = fisherInformationTheory(params_fid)
    dict_param_steps = {parameter_names[i]:delta_params[i] for i in range(len(params_fid))}
    dict_deriv_sims = generate_deriv_sims(parameter_names,params_fid,delta_params,nSims=nSims_deriv)
    cFisher = CompressedFisher.gaussianFisher(parameter_names,nSims_deriv,include_covmat_param_depedence=False,deriv_finite_dif_accuracy=2)
    cFisher.initailize_covmat(covmat_sims)
    cFisher.initailize_mean(covmat_sims)
    cFisher.initailize_deriv_sims(dic_deriv_sims=dict_deriv_sims,dict_param_steps=dict_param_steps)
    cFisher.generate_deriv_sim_splits(.5)
    np.isclose(fisher_true,cFisher._compute_compressed_fisher_matrix(parameter_names)-cFisher._compute_compressed_fisher_matrix_error(parameter_names),rtol=2/np.sqrt(.5*nSims_deriv))


def test_splineweights():
    """
    Test that the spline weights are correctly updated
    """

    cFisher =  initializeFisher()
    cFisher.initailize_spline_weights(CompressedFisher.fisher.central_difference_weights[4])
    assert(np.all(cFisher._deriv_finite_dif_weights==CompressedFisher.fisher.central_difference_weights[4]))
