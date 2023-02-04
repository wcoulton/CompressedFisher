

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
    mu_k = muk_theory(ks,params)
    return np.diag(2*mu_k**2)


def generateSims(params,nSims=1):
    """
    Generate draw realizations from the toy model with the given set of parameters
    
    Args:
        params ([list floats]): The values of the parameters to draw realizations at.
        nSims (ind): The number of sims to generate (default: `1`)
    Returns:
        [array (nSims,nKs) ]: The Gaussian simulations
    """

    mu_k = muk_theory(ks,params)
    return  mu_k+np.random.randn(nSims,nKs)*np.sqrt(2)*mu_k



def generate_deriv_sims(paramter_names,params,deltaParams,nSims):
    """
    Generates simulations for use as finite difference derivatives.
    These are central finite differences, ths for each parameter 
    simulations are generated at $\\theta\\pm \\delta\\theta$.
    
    For this case we use the 'seed matching technique'.
    In this method the seeds for the simulations at \\theta+\\delta\\theta and \\theta-\\delta\\theta
    are chosen to match. This is often used in pratical cases to reduce the MC noise.
    
    
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
    state = np.random.get_state()
    for i,name in enumerate(paramter_names):
        derivs_ens = np.zeros([2,nSims,nKs])
        params_plus = np.array(params)
        params_plus[i]+=deltaParams[i]
        
        params_minus = np.array(params)
        params_minus[i]-=deltaParams[i]
          
        np.random.set_state(state)
        derivs_ens[0] = generateSims(params_minus,nSims)
        np.random.set_state(state)  
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
    mu_k = muk_theory(ks,params)
    nParams = len(params)
    derivs_thry = np.zeros([nParams,nKs])
    derivs_thry[0] = np.ones(nKs)
    derivs_thry[1] = ks.copy()
    derivs_thry[2] = np.sqrt(ks)
    derivs_thry_covMat = np.zeros([nParams,nKs,nKs])
    for i in range(nParams):
        derivs_thry_covMat[i] = 4*np.diag(derivs_thry[i])*mu_k
    return derivs_thry,derivs_thry_covMat

def fisherInformationTheory(params):
    """
    Compute the exact Fisher informations
    
    Args:
        params ([array]): The values where the Fisher information are to be evaluated
    
    Returns:
        [matrix nParams x nParams]: The analytical Fisher information (i.e. the `truth')
    """
    derivs,derivs_covMat = theoretical_derivs(params)
    covMat = generateCovMat(params)
    nParams = len(params)
    fisher = np.zeros([nParams,nParams])
    for i in range(nParams):
        tmp_a = (np.linalg.solve(covMat,derivs_covMat[i]))
        for j in range(i+1):
            tmp_b = (np.linalg.solve(covMat,derivs_covMat[j]))
            tmp =  np.dot(derivs[i], np.linalg.solve(covMat,derivs[j]))
            tmp+= .5*np.trace(np.dot(tmp_a,tmp_b))
            fisher[i,j] = fisher[j,i] = tmp
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

def generate_analytic_deriv_mean_sims(n1,ids):
    """"
    A function to create mock finite difference sims for the mean. These 'sims' are just given at the theoretical values (without MC noise).
    T
    Args:
        parameter_names ([list]): A list of the parameters to generate sims for
        params ([dicitonary]): The values of the parameters to evaluate the central difference sims about
        deltaParams ([list]):  The parameter step sizes for the finite differences.
        nSims ([int]): The number of mock sims to generate
    
    Returns:
        [dictionary]: A dictionary containing the mock derivatives
    """
    derivs_thry_tmp,derivs_thry_covMat_tmp =theoretical_derivs(params_fid)
    dict_derivs_mean_thr = {parameter_names[i]:derivs_thry_tmp[i] for i in range(len(params_fid))}
    return dict_derivs_mean_thr[n1]
def generate_analytic_deriv_covmat_sims(n1,ids):
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

    derivs_thry_tmp,derivs_thry_covMat_tmp =theoretical_derivs(params_fid)
    dict_derivs_covmat_thr = {parameter_names[i]:derivs_thry_covMat_tmp[i] for i in range(len(params_fid))}
    return dict_derivs_covmat_thr[n1]



def analyticalCompression(data,params):
    """
    Compress the data vector using the analytically evaluated compression
    Args:
        data ([...,nKs]): The data to compress
        params ([vector (nParams)]): The parameters to evaluate the compression at
    
    Returns:
        [type [...., nParams]]: The compressed vector
    """
    pk_mu = muk_theory(ks,params) # Note null mean sub
    derivs_mean,derivs_covMat = theoretical_derivs(params)
    covMat = generateCovMat(params)
    results = np.zeros([data.shape[0],len(params)])
   
    tmp = np.linalg.solve(covMat,(data-pk_mu).T)
    for i in range(derivs_mean.shape[0]):
            results[:,i]= derivs_mean[i].dot(tmp)
            results[:,i] += .5*np.einsum('ij,ij->i',tmp.T,(derivs_covMat[i].dot(tmp)).T)
    return results


def initializeFisher():
    """
    Set up the Fisher object. This also initializes the cov mat. 

    Returns:
        [gaussianFisher object]: A gaussianFisher object for use in the null tests
    """

   
    nSims_deriv = 100

    cFisher = CompressedFisher.gaussianFisher(parameter_names,nSims_deriv,include_covmat_param_depedence=True,deriv_finite_dif_accuracy=None)
    cFisher._covmat_fisher = generateCovMat(params_fid)
    cFisher._covmat_comp = generateCovMat(params_fid)
    cFisher.initailize_mean(muk_theory(ks,params_fid).reshape([1,-1]))
    #dict_deriv_sims = generate_analytic_deriv_sims(parameter_names,params_fid,delta_params,nSims=nSims_deriv)
    cFisher.initailize_deriv_sims(deriv_mean_function=generate_analytic_deriv_mean_sims,deriv_covmat_function=generate_analytic_deriv_covmat_sims)

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
    comp_cFish = cFisher.compress_vector(test_data,with_mean=True)
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

