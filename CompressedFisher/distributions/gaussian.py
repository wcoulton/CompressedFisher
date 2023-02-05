import numpy as np
import warnings
import scipy.linalg
from ..fisher import baseFisher,central_difference_weights



class gaussianFisher(baseFisher):
    def __init__(self,param_names,n_sims_derivs,include_covmat_param_depedence=False,n_sims_covmat=None,deriv_finite_dif_accuracy=None):
        """
        The class provides tools for computing fisher forecasts when the data is described by a Gaussian distribution. 
        It is designed to handle data in the following format:
        A vector of measurements, x, of dimension, d, described
        x ~ Gaussian (\mu(\theta),\Sigma(\theta))
        where \mu is a vector of length d giving the mean for each observation and \Sigma is the covariance of the observations.
        
        Many common cases have observations with parameter dependent means, but parameter indepedent covariance matricies.
        Thus to facilitate such analyses this class can either include or not the parameter dependent covariance with the include_covmat_param_depedence option


        A common workflow will look like this (see the docs and examples for details on the methods)

        call the method:  initailize_covmat  
        call the method:  initailize_mean              
        call the method:  initailize_deriv_sims
        call the method:  generate_deriv_sim_splits
        
        compute Fisher forecats: e.g. with the method compute_fisher_forecast

        Note the expected format of the simulated derivatives will be the of the form: 
        (n_sims_derivs,dimension) or (n_sims_derivs,deriv_spline_index,dimension)
        The first form is used when your input is realizations of the derivatives themselves.
        However simulated derivatives are often obtain by (central) finite differences with different orders for different accuracy.
        For example commonly a second order spline is used, i.e., derivatives are obtain as f(\theta+\delta\theta)-f(\theta-\delta\theta)/(2 \delta \theta).
        If working with this type of output set deriv_finite_dif_accuracy to the accuracy of central difference (the above example is order 2)
        otherwise leave deriv_finite_dif_accuracy as none. The input to the code should be the simulations at \theta+\delta\theta (\theta-\delta\theta)
        The code will internally compute the finite differences. This is the preferred mode of operation,
        
        Args:
            param_names ([list of strings]): A list of the names of the parameters under consideration
            n_sims_derivs ([int]): The total number of derivative simulations available. This is assumed to be the same for all the parameters
            deriv_finite_dif_accuracy ([int]): The order of the accuracy of the finite difference derivatives. Leave as none if not using the code to evaluate finite differences   [description] (default: `None`)
        """     
        
        self.param_names = param_names
        self._include_covmat_param_depedence = include_covmat_param_depedence


        self.n_sims_derivs = n_sims_derivs
        self.n_sims_covmat = n_sims_covmat

        self._deriv_finite_dif_accuracy = deriv_finite_dif_accuracy
        self._n_params = len(self.param_names)
        self._deriv_mean_fisher = None
        self._deriv_covmat_fisher = None
        self._covmat_fisher = None

        self._mean_comp = None

        self._deriv_mean_comp = None
        self._deriv_covmat_comp = None
        self._covmat_comp = None

        self._covmat_sims = None
        self._store_covmat_sims = False


        self._hartlap_comp = 1.
        self._hartlap_fisher = 1.

        self._dict_deriv_sims = None
        self._dict_param_steps = None
        self._deriv_sim_ids = None
        self._has_deriv_sims = True

        self._deriv_mean_function = None
        self._deriv_covmat_function = None


        self._deriv_finite_dif_weights=None


    @property
    def covmat_fisher(self):
        """
        Access the covariance matrix used in the Fisher forecasting
        
        Returns:
            [DxD matrix]: The covariance matrix used in the fisher analysis. 
        
        """
        if (self._covmat_fisher is None):
            raise AssertionError('Need to initailize Fisher covmat first')
        return self._covmat_fisher

    @covmat_fisher.setter
    def covmat_fisher(self,ids):
        if ids is None:
            self._covmat_fisher = np.cov(self._covmat_sims.T)
            n_sims_covmat,self.dim =  self._covmat_sims.shape
            self._n_sims_covmat_fisher = n_sims_covmat
            self._hartlap_fisher = (n_sims_covmat-self.dim-2)/(n_sims_covmat-1)

        else:
            self._covmat_fisher = np.cov(self._covmat_sims[ids].T)
            n_sims_covmat,self.dim =  self._covmat_sims[ids].shape
            self._hartlap_fisher = (n_sims_covmat-self.dim-2)/(n_sims_covmat-1)

    @property
    def covmat_comp(self):
        """
        Access the covariance matrix used in the compression operation
        
        Returns:
            [DxD matrix]: The compression covariance matrix
        
        """
        if (self._covmat_comp is None):
            raise AssertionError('Need to initailize compression covmat first')
        return self._covmat_comp

    @covmat_comp.setter
    def covmat_comp(self,ids):
        if ids is None:
            self._covmat_comp = np.cov(self._covmat_sims.T)
            n_sims_covmat,self.dim =  self._covmat_sims.shape
            self._hartlap_comp= (n_sims_covmat-self.dim-2)/(n_sims_covmat-1)
        else:
            self._covmat_comp = np.cov(self._covmat_sims[ids].T)
            n_sims_covmat,self.dim =  self._covmat_sims[ids].shape
            self._hartlap_comp= (n_sims_covmat-self.dim-2)/(n_sims_covmat-1)
        if self._include_covmat_param_depedence:
            self.initailize_mean(self._covmat_sims[ids])

    @property
    def deriv_mean_fisher(self):
        """
        Access the derivatives of the mean used in the fisher forecasting.
        This is computed from the fraction of simulations assigned to the fisher analysis 
        
        Returns:
            [dict]: a dictionary containing the derivatives of the mean with respect to each parameter.
                    
        """
        if (self._deriv_mean_fisher is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_mean_fisher = None
        return self._deriv_mean_fisher

    @deriv_mean_fisher.setter
    def deriv_mean_fisher(self,ids):
        self._deriv_mean_fisher = {}
        for n in self.param_names:
            self._deriv_mean_fisher[n]=np.mean(self._get_deriv_mean_sims(n,ids),axis=0)
        self._deriv_sim_ids = ids
        return self._deriv_mean_fisher


    @property
    def deriv_mean_comp(self):
        """
        Access the derivatives of the mean used in the compression steps
        This is computed from the fraction of simulations assigned to the compression.
        
        Returns:
            [dict]: a dictionary containing the derivatives of the mean with respect to each parameter.
                    
        """
        if (self._deriv_mean_comp is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_mean_comp = None
        return self._deriv_mean_comp

    @deriv_mean_comp.setter
    def deriv_mean_comp(self,ids):
        self._deriv_mean_comp = {}
        for n in self.param_names:
            self._deriv_mean_comp[n]=np.mean(self._get_deriv_mean_sims(n,ids),axis=0)

        return self._deriv_mean_comp



    @property
    def deriv_covmat_fisher(self):
        """
        Access the derivatives of the covariance matrix used in the fisher forecasting
        This is computed from the fraction of simulations assigned to the fisher analysis 
        
        Returns:
            [dict]: a dictionary containing the derivatives of the mean with respect to each parameter.
                    
        """
        if (self._deriv_covmat_fisher is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_covmat_fisher = None
        return self._deriv_covmat_fisher

    @deriv_covmat_fisher.setter
    def deriv_covmat_fisher(self,ids):
        self._deriv_covmat_fisher = {}
        for param_name in self.param_names:
            if self._has_deriv_sims is False or self._deriv_finite_dif_accuracy is None:
                self._deriv_covmat_fisher[param_name]=np.mean(self._get_deriv_covmat_sims(param_name,ids),axis=0)
            else:
                sims = self._dict_deriv_sims[param_name]
                if ids is not None:
                    sims = sims[:,ids]
                if self.deriv_finite_dif_weights is None:
                    self.initailize_spline_weights()
                sim_covs = 0
                for finite_dif_index,finite_dif_weight in enumerate(self.deriv_finite_dif_weights[param_name]):
                    sim_covs+=finite_dif_weight/self._dict_param_steps[param_name]*np.cov(sims[finite_dif_index].T)
                self._deriv_covmat_fisher[param_name] =sim_covs
        self._deriv_sim_ids = ids
        return self._deriv_covmat_fisher


    @property
    def deriv_covmat_comp(self):
        """
        Access the derivatives of the covariance matrix used in the compression step
        This is computed from the fraction of simulations assigned to the compression step
        
        Returns:
            [dict]: a dictionary containing the derivatives of the mean with respect to each parameter.
        """
        if (self._deriv_covmat_comp is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_covmat_comp = None
        return self._deriv_covmat_comp

    @deriv_covmat_comp.setter
    def deriv_covmat_comp(self,ids):
        self._deriv_covmat_comp = {}
        for param_name in self.param_names:
            if self._has_deriv_sims is False or self._deriv_finite_dif_accuracy is None:
                self._deriv_covmat_comp[param_name]=np.mean(self._get_deriv_covmat_sims(param_name,ids),axis=0)
            else:
                sims = self._dict_deriv_sims[param_name]
                if ids is not None:
                    sims = sims[:,ids]
                if self.deriv_finite_dif_weights is None:
                    self.initailize_spline_weights()
                sim_covs = 0
                for finite_dif_index,finite_dif_weight in enumerate(self.deriv_finite_dif_weights[param_name]):
                    sim_covs+=finite_dif_weight/self._dict_param_steps[param_name]*np.cov(sims[finite_dif_index].T)
                self._deriv_covmat_comp[param_name] =sim_covs
        return self._deriv_covmat_comp



    def initailize_covmat(self,covmat_sims,store_covmat_sims=False):
        """
        This function supplies the estimator with the simulations used to compute the covariance matrix.
        If  store_covmat_sims is False then all the simulations are used in both the compression and the fisher estimation.
        The bias from this has generally found to be small and subdominant to that from the derivatives and for most cases this 
        is the recommended approach.  
        This can be relaxed by setting store_variance_sims to True and using the generate_covmat_sim_splits to split simulations.
        
        Args:
            covmat_sims (matrix [N_sims x Dimension]): The simulations used to compute the covariance matrix.
            store_covmat_sims (bool): Whether to store the covmat sims in the objects. (default: `False`)
        """
        self._store_covmat_sims = store_covmat_sims
        self._covmat_sims = covmat_sims
        self.covmat_fisher = None
        self.covmat_comp = None
        if not store_covmat_sims:
            self._covmat_sims = None
        self.n_sims_covmat = covmat_sims.shape[0]
        if self._include_covmat_param_depedence and self._mean_comp is None:
            self.initailize_mean(covmat_sims)

    def initailize_mean(self,mean_sims): 
        """
        This function supplies the estimator with the simulations used to compute the mean. There are no issues if these are the same sims as the covmat. sims.

        
        Args:
            mean (matrix [N_sims x Dimension]): The simulations used to compute the mean.
        """
        self._mean_comp = np.mean(mean_sims,axis=0)

    def initailize_deriv_sims(self,dic_deriv_sims=None,dict_param_steps=None,deriv_mean_function=None,deriv_covmat_function=None):
        """
        Pass the set of simulations used to estimate the derivatives. There are three modes for this:
        1)
        Pass a dictionary containing the simulations for finite difference derivatives. Each element should be an array of shape
        (Number Sims, number of param step, Dimension) - where number of param steps corresponds to the points used to 
        estimate a difference derivative. E.g. for a second order central difference index =0 should have sims at \theta-\delta\theta 
        and index =1 should have sims at \theta+\delta\theta.
        If this mode is used dict_param_steps should be supplied and will be a dictionary containing the step size for each parameter.
        This is the preferred mode of operation for the code.
        2)
        Pass a dictionary containing the simulations of the derivatives of the mean themselves. Each element should be an array of shape
        (Number Sims, Dimension). This cannot be used when considering a parameter dependence covariance matrix.
        3)
        Pass a function, deriv_mean_function, which takes as arguments the parameter name and the sim id and returns the derivative of the mean for that simulation.
        The return should be an array of shape (dimension).
        If you are considering a model with a parameter dependent covariance matrix you also need to pass the deriv_covmat_function, with identical inputs as the above function.
        The return should be an array of shape (dimension x dimension).

        
        Args:
            dic_deriv_sims ([dictionary]): A dictionary containing the either simulations for computing finite differences for each parameter or a dictionary of the derivatives for each parameter. The key for the dictionary should be the parameter name. Needed for mode 1 or 2. (default: `None`)
            dict_param_steps ([dictionary]): A dictionary of the parmaeter step sizes. Only needed for mode 1) (default: `None`)
            deriv_rate_function ([function]): A function that returns the derivative of the mean for a single realizaiton. Arguments of the function are parameter name and sim index. Only needed for mode 3  (default: `None`)
            deriv_covmat_function ([function]): A function that returns the derivative of the covariance matrix for a single realizaiton. Arguments of the function are parameter name and sim index. Only needed for mode 3  (default: `None`)
        """

        if dic_deriv_sims is None:
            assert(deriv_mean_function is not None)
            self._has_deriv_sims = False
            self._deriv_mean_function = deriv_mean_function
            if self._include_covmat_param_depedence:
                assert(deriv_covmat_function is not None)
                self._deriv_covmat_function = deriv_covmat_function
        else:
            self._has_deriv_sims = True
            self._dict_deriv_sims = dic_deriv_sims
            if self._deriv_finite_dif_accuracy is not None:
                assert(dict_param_steps is not None)
                self._dict_param_steps = dict_param_steps




    def compress_vector(self,vector,with_mean=True):
        """
        Apply the compression to a data vector. 
        The optimal compression requires the mean to be subtracted, however in the Fisher forecast this term drops out so the compression 
        can be performed without this subtraction. This also minimizes the noise in the fisher estimate.
        Eq. 25 in Coulton and Wandelt 

        Args:
            vector ((..., Dimension)): The data vector (can be multidimensional but the last dimension should be the dimension of the mean).
            with_mean (bool): Subtract the mean or not (True)
        
        Returns:
            [vector (..., n_parameters)]: The compressed data vector (the compression is performed on the last axis)
        """
        if with_mean:
            assert(self._mean_comp is not None)

        if self._include_covmat_param_depedence:
            if self._mean_comp is None:
                raise AssertionError('For compression with parameter dependent cov mat. we require the mean')
            return self._compress_mean_and_covmat(vector)
        else:
            return self._compress_mean_only(vector,with_mean=with_mean)


    def _get_deriv_mean_sims(self,param_name,ids):
        """
        A helper function for accessing the derivative of the mean simulations.
        This function returns a subset of the total derivative simulations
        
        Args:
            param_name ([string]): The name of the parameter with which the derivative is respect to 
            ids ([list]): The list of sim ids indexing the derivatives
        
        Returns:
             array of shape (len(ids),dimension) : the subset of derivative realizations
        """
        if not self._has_deriv_sims:
            if ids is None:
                ids = np.arange(self.n_sims_derivs)
            nSims = len(ids)
            sims = []
            for i,id0 in enumerate(ids):
                sim = self._deriv_mean_function(param_name,id0)
                if i==0:
                    sims = np.zeros((nSims,)+sim.shape)
                sims[i] = sim
        else:
            sims = self._dict_deriv_sims[param_name]
            if ids is not None:
                if self._deriv_finite_dif_accuracy is None:
                    sims = sims[ids]
                else:
                    sims = sims[:,ids]
            if self._deriv_finite_dif_accuracy is not None:
                if self.deriv_finite_dif_weights is None:
                    self.initailize_spline_weights()
                sims = np.einsum('i,ijk->jk', self.deriv_finite_dif_weights[param_name],sims)/self._dict_param_steps[param_name]
        return sims

    def _get_deriv_covmat_sims(self,param_name,ids):
        """
        A helper function for accessing the derivative of the covmat simulations.
        This function returns a subset of the total derivative simulations
        
        Args:
            param_name ([string]): The name of the parameter with which the derivative is respect to 
            ids ([list]): The list of sim ids indexing the derivatives
        
        Returns:
             array of shape (len(ids),dimension) : the subset of derivative realizations
        """
        if not self._has_deriv_sims:
            if ids is None:
                ids = np.arange(self.n_sims_derivs)
            sims = []
            nSims = len(ids)
            for i,id0 in enumerate(ids):
                sim = self._deriv_covmat_function(param_name,id0)
                if i==0:
                    sims = np.zeros((nSims,)+sim.shape)
                sims[i] = sim
        else:
            sims = self._dict_deriv_sims[param_name]
            if self._deriv_finite_dif_accuracy is not None:
               raise AssertionError("If finite difference sims are passed, it is easier to compute the covariance matrix of p+delta and at p-delta and compute the derivative")

            if ids is not None:
                if self._deriv_finite_dif_accuracy is None:
                    sims = sims[ids]
                else:
                    sims = sims[:,ids]
        return sims




    def _apply_deriv_split(self,ids_comp,ids_fish):
        """
        Use the given set of ids for the compression and the seperate set for the fisher calculation to compute the 
        deriviates for the fisher forecast

        
        Args:
            ids_comp ([list]): A list of ids of the simulations to be used to compute the compressions
            ids_fish ([list]): A list of ids of the simulations to be used to compute the fisher forecast
        """

        self.deriv_mean_comp = ids_comp
        self.deriv_mean_fisher = ids_fish
        if self._include_covmat_param_depedence:
            self.deriv_covmat_comp = ids_comp
            self.deriv_covmat_fisher = ids_fish

    def _apply_covmat_split(self,ids_fish,ids_comp):
        """
        Use the given set of ids for the compression and the seperate set for the fisher calculation to compute the 
        variance for the fisher forecast. Typically spliting these simulations is not necessary so ids_fish=ids_comp =all the variance sims.

        
        Args:
            ids_comp ([list]): A list of ids of the simulations to be used to compute the compressions
            ids_fish ([list]): A list of ids of the simulations to be used to compute the fisher forecast
        """
        self.covmat_comp = ids_comp
        self.covmat_fisher = ids_fish


    def _compress_mean_only(self,data,with_mean=False):
        """
        Apply the compression to a data vector. This is the specialized function for the case of a parameter independent cov mat.
        The optimal compression requires the mean to be subtracted, however in the Fisher forecast this term drops out so the compression 
        can be performed without this subtraction. This also minimizes the noise in the fisher estimate.
        Eq. 25 in Coulton and Wandelt with derivatives of C set to 0

        Args:
            vector ((..., Dimension)): The data vector (can be multidimensional but the last dimension should be the dimension of the mean).
            with_mean (bool): Subtract the mean or not (True)
        
        Returns:
            [vector (..., n_parameters)]: The compressed data vector (the compression is performed on the last axis)
        """
        data = np.atleast_2d(data)
        results = np.zeros([data.shape[0],self._n_params])
        if with_mean:
            data = data -self._mean_comp
        for i,n in enumerate(self.param_names):
            deriv = self.deriv_mean_comp[n]
            results[:,i]= self._hartlap_comp*deriv.dot(np.linalg.solve(self.covmat_comp,(data).T))
        return results

    def _compress_mean_and_covmat(self,data):
        """
        Apply the compression to a data vector. This is the specialized function for the case of a parameter dependent cov mat.
        The optimal compression requires the mean to be subtracted, however in the Fisher forecast this term drops out so the compression 
        can be performed without this subtraction. This also minimizes the noise in the fisher estimate.
        Eq. 25 in Coulton and Wandelt

        Args:
            vector ((..., Dimension)): The data vector (can be multidimensional but the last dimension should be the dimension of the mean).
            with_mean (bool): Subtract the mean or not (True)
        
        Returns:
            [vector (..., n_parameters)]: The compressed data vector (the compression is performed on the last axis)
        """
        data = np.atleast_2d(data)
        results = np.zeros([data.shape[0],self._n_params])
        data = data -self._mean_comp
        tmp = np.linalg.solve(self.covmat_comp,(data).T)*self._hartlap_comp
        for i,n in enumerate(self.param_names):
            mu_deriv = self.deriv_mean_comp[n]
            results[:,i]= mu_deriv.dot(tmp)

            results[:,i] += .5*np.einsum('ij,ij->i',tmp.T,(self.deriv_covmat_comp[n].dot(tmp)).T)

        return results

    def _compute_deriv_mu_covmat(self,params_names,input_ids=False):
        """
        Compute the covariance of the derivatives. This is used to estimate the bias to the Fisher information
        
        Args:
            params_names ([list]): The names of the derivatives to use.
            input_ids (bool): The set of ids used to compute the variance. If none use the simulations assigned to the fisher calculation. (default: `False`)
        
        Returns:
            A matrix with shape (n_params,n_params, dimension, dimension): The covariance of the derivative simulations
        """
        n_params = len(params_names)
        if input_ids is False:
            input_ids = self._deriv_sim_ids
        for i,n1 in enumerate(params_names):
            derivs1_ens = self._get_deriv_mean_sims(n1,input_ids)
            derivs1_mn = np.mean(derivs1_ens,axis=0)
            if i==0:
                nSims = derivs1_ens.shape[0]
                
                deriv_covmat = np.zeros([n_params,n_params,self.dim,self.dim])
            for j,n2 in enumerate(params_names[:i+1]):
                derivs2_ens = self._get_deriv_mean_sims(n2,input_ids)
                derivs2_mn = np.mean(derivs2_ens,axis=0)
                deriv_covmat[i,j] = deriv_covmat[j,i]= 1/(nSims-1)*np.einsum('ij,ik->jk',(derivs1_ens-derivs1_mn),(derivs2_ens-derivs2_mn) )
        return deriv_covmat/nSims

    def _compute_fisher_matrix(self,params_names=None):
        """
        Estimate the bias to the Fisher matrix
        This is computed with Eq. 6 and Eq. 20 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The Fisher information.
        """
        if params_names is None: params_names = self.params_names
        if self._include_covmat_param_depedence:
            return self._compute_fisher_matrix_mean_and_covmat(params_names)
        else:
            return self._compute_fisher_matrix_mean_only(params_names)
            
    def _compute_fisher_matrix_mean_only(self,params_names):
        """
        The function to compute the Fisher matrix, with the standard approach, for the case of a parameter independence covariance matrix
        This is computed with Eq. 21 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The Fisher information.
        """

        if params_names is None:
            params_names = self.params_names
        n_params = len(params_names)
        fisher = np.zeros([n_params,n_params])
        for i,n1 in enumerate(params_names):
            for j,n2 in enumerate(params_names[:i+1]):
                fisher[i,j] = fisher[j,i] = self._hartlap_fisher*np.dot(self.deriv_mean_fisher[n1], np.linalg.solve(self.covmat_fisher,self.deriv_mean_fisher[n2]))
        return fisher

    def _compute_fisher_matrix_mean_and_covmat(self,params_names):
        """
        The function to compute the Fisher matrix, with the standard approach, for the case of a parameter dependence covariance matrix
        This is computed with Eq. 21 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The Fisher information.
        """
        if params_names is None:
            params_names = self.params_names
        n_params = len(params_names)
        fisher = np.zeros([n_params,n_params])
        for i,n1 in enumerate(params_names):
            tmp_a = np.linalg.solve(self.covmat_fisher,self.deriv_covmat_fisher[n1])*self._hartlap_fisher
            for j,n2 in enumerate(params_names[:i+1]):
                tmp_b = np.linalg.solve(self.covmat_fisher,self.deriv_covmat_fisher[n2])*self._hartlap_fisher
                mu_term = np.dot(self.deriv_mean_fisher[n1], np.linalg.solve(self.covmat_fisher,self.deriv_mean_fisher[n2]))*self._hartlap_fisher
                cov_term = +.5*np.trace(np.dot(tmp_a,tmp_b))
                fisher[i,j] = fisher[j,i] = mu_term+cov_term
        return fisher


    def _compute_fisher_matrix_error(self,params_names=None):
        """
        Estimate the bias to the Fisher matrix
        This is computed with Eq. 6 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the calculation. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The bias to the Fisher information.
        """
        if params_names is None: params_names = self.params_names
        if self._include_covmat_param_depedence:
            return self._compute_fisher_matrix_mean_and_covmat_error(params_names)
        else:
            return self._compute_fisher_matrix_mean_only_error(params_names)
            
    def _compute_fisher_matrix_mean_only_error(self,params_names):
        """
        The function to estimate the bias to the Fisher matrix. This is the specialized case of a parameter independent covariance matrix.
        This is computed with Eq. 6 and Eq. 23 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the calculation
        
        Returns:
            [n_parameter x n_parameter matrix]: The bias to the Fisher information.
        """
        n_params = len(params_names)


        derivs_covMat = self._compute_deriv_mu_covmat(params_names)

        fisher_err = np.zeros([n_params,n_params])
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(self.covmat_fisher,derivs_covMat[i,j]))*self._hartlap_fisher
        return fisher_err

    def _compute_fisher_matrix_mean_and_covmat_error(self,params_names):
        """
        The function to esimate the bias to the Fisher matrix. This is the specialized case of a parameter dependent covariance matrix.
        This is computed with Eq. 6 and Eq. 23 in Coulton and Wandelt
        
        Args:
            params_names ([list]): The list of parameters to include in the calculation
        
        Returns:
            [n_parameter x n_parameter matrix]: The  bias to the Fisher information.
        """
        n_params = len(params_names)


        def estBiasCovMatVar(params_names):
            ids_0 = self._deriv_sim_ids

            if ids_0 is None:
                ids_tmp = np.arange(self.n_sims_derivs)
                np.random.shuffle(ids_tmp)
                ids_a = ids_tmp[:int(self.n_sims_derivs*.5)]
                ids_b = ids_tmp[int(self.n_sims_derivs*.5):]
            else:
                n_sims_derivs = len(ids_0)
                ids_a = ids_0[:int(n_sims_derivs*.5)]
                ids_b = ids_0[int(n_sims_derivs*.5):]
            derivs_covMat = self.deriv_covmat_fisher
            self.deriv_covmat_fisher = ids_a
            derivs_covMat_a = self.deriv_covmat_fisher
            self.deriv_covmat_fisher = ids_b
            derivs_covMat_b = self.deriv_covmat_fisher
            # Reset derivatives
            self.deriv_covmat_fisher = ids_0
            fisher_err = np.zeros([n_params,n_params])
            for i,n1 in enumerate(params_names):
                tmp_mat_a = np.linalg.solve(self.covmat_fisher,derivs_covMat[n1])*self._hartlap_fisher
                tmp_mat_splita_a = np.linalg.solve(self.covmat_fisher,derivs_covMat_a[n1])*self._hartlap_fisher
                tmp_mat_splitb_a = np.linalg.solve(self.covmat_fisher,derivs_covMat_b[n1])*self._hartlap_fisher

                for j,n2 in enumerate(params_names[:i+1]):
                    tmp_mat_b = np.linalg.solve(self.covmat_fisher,derivs_covMat[n2])*self._hartlap_fisher
                    tmp_mat_splita_b = np.linalg.solve(self.covmat_fisher,derivs_covMat_a[n2])*self._hartlap_fisher
                    tmp_mat_splitb_b = np.linalg.solve(self.covmat_fisher,derivs_covMat_b[n2])*self._hartlap_fisher
                    tmp = (self._n_sims_covmat_fisher-self.dim)*(np.trace(np.dot(tmp_mat_a,tmp_mat_b) ))
                    tmp += (self._n_sims_covmat_fisher-self.dim-2)*(np.trace(tmp_mat_a)*np.trace(tmp_mat_b))
                    tmp_b = (np.trace(np.dot(tmp_mat_a,tmp_mat_b) ))-.5*(np.trace(np.dot(tmp_mat_splita_a,tmp_mat_splitb_b))+np.trace(np.dot(tmp_mat_splitb_a,tmp_mat_splita_b)) )
                    fisher_err[i,j] =1/(self._n_sims_covmat_fisher-self.dim-1)/(self._n_sims_covmat_fisher-self.dim-4)*tmp+tmp_b
                    fisher_err[j,i] = fisher_err[i,j] 
                    
            return .5*fisher_err


        derivs_covMat = self._compute_deriv_mu_covmat(params_names)

        fisher_err = np.zeros([n_params,n_params])

        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(self.covmat_fisher,derivs_covMat[i,j]))*self._hartlap_fisher

        fisher_err+=estBiasCovMatVar(params_names)
        return fisher_err



    def _compute_compressed_fisher_matrix(self,params_names=None):
        """
        Compute the compressed Fisher matrix.
        This is computed with Eq. 10 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The Fisher information.
        """
        if params_names is None: params_names = self.params_names
        if self._include_covmat_param_depedence:
            return self._compute_compressed_fisher_matrix_mean_and_covmat(params_names)
        else:
            return self._compute_compressed_fisher_matrix_mean_only(params_names)

    def _compute_compressed_fisher_matrix_mean_only(self,params_names):
        """
        The function to compute the compressed Fisher matrix for the case of a parameter indepednent covariance matrix
        This is computed with Eq. 10 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast.
        
        Returns:
            [n_parameter x n_parameter matrix]: The Fisher information.
        """
        dim_fisher = len(params_names)
        compressed_derivs = np.zeros([dim_fisher,self._n_params])
        for i,n in enumerate(params_names):
            compressed_derivs[i]=self.compress_vector(self.deriv_mean_fisher[n],with_mean=False)

        compressed_covmat = self.compress_vector(self.covmat_fisher,with_mean=False)
        compressed_covmat = self.compress_vector(compressed_covmat.T,with_mean=False)/self._hartlap_comp # One factor cancels here as we have C^{-1}C C^{-1} 
        return self._compressed_fisher_matrix(compressed_derivs,compressed_covmat)

    def _compute_compressed_fisher_matrix_mean_and_covmat(self,params_names):
        """
        The function to compute the compressed Fisher matrix for the case of a parameter depednent covariance matrix
        This is computed with Eq. 10 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast.
        
        Returns:
            [n_parameter x n_parameter matrix]: The Fisher information.
        """
        n_params = len(params_names)
        mu_t = np.zeros([n_params,self._n_params])
        for i,n1 in enumerate(params_names):
            cov_term_a = np.linalg.solve(self.covmat_comp,self.deriv_covmat_fisher[n1])*self._hartlap_comp
            mu_term_a = np.linalg.solve(self.covmat_comp,self.deriv_mean_fisher[n1])*self._hartlap_comp
            for j,n2 in enumerate(self.param_names):
                cov_term_b = np.linalg.solve(self.covmat_comp,self.deriv_covmat_comp[n2])*self._hartlap_comp
                mu_term = np.dot(self.deriv_mean_comp[n2],mu_term_a )
                cov_term = +.5*np.trace(np.dot(cov_term_b,cov_term_a))
                mu_t[i,j] =  mu_term+cov_term
        sig_tt = self._compute_fisher_matrix(self.param_names)
        return np.dot(mu_t,np.linalg.solve(sig_tt,mu_t))


    def _compute_compressed_fisher_matrix_error(self,params_names=None):
        """
        Compute the bias to the compressed Fisher matrix.
        This is computed with Eq. 11 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The bias to the Fisher information.
        """
        if params_names is None: params_names = self.params_names
        if self._include_covmat_param_depedence:
            return self._compute_compressed_fisher_matrix_mean_and_covmat_error(params_names)
        else:
            return self._compute_compressed_fisher_matrix_mean_only_error(params_names)
            
    def _compute_compressed_fisher_matrix_mean_only_error(self,params_names):
        """
        The function to compute the bias to the compressed Fisher matrix for the case of a parameter independent covariance matrix.
        This is computed with Eq. 11 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The bias to the Fisher information.
        """
        n_params = len(params_names)


        derivs_covMat = self._compute_deriv_mu_covmat(params_names)

        compressed_covmat = self.compress_vector(self.covmat_fisher,with_mean=False)
        compressed_covmat = self.compress_vector(compressed_covmat.T,with_mean=False)

        mix_matrix =  np.zeros([n_params,n_params,self._n_params,self._n_params])
        for i in range(n_params):
            for j in range(i+1):
                tmp_mat = self.compress_vector(derivs_covMat[i,j],with_mean=False)
                mix_matrix[i,j] = mix_matrix[j,i] = self.compress_vector(tmp_mat.T,with_mean=False)
        fisher_err = np.zeros([n_params,n_params])
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(compressed_covmat,mix_matrix[i,j]))
        return fisher_err

    def _compute_compressed_fisher_matrix_mean_and_covmat_error(self,params_names):
        """
        The function to compute the bias to the compressed Fisher matrix for the case of a parameter dependent covariance matrix.
        This is computed with Eq. 11 in Coulton and Wandelt.
        
        Args:
            params_names ([list]): The list of parameters to include in the forecast. If none all parameters will be used (default: `None`)
        
        Returns:
            [n_parameter x n_parameter matrix]: The bias to the Fisher information.
        """
        n_params = len(params_names)
        if not self._has_deriv_sims or self._deriv_finite_dif_accuracy is None or self._deriv_finite_dif_accuracy in [0,1]:
            raise AssertionError('Currently only implemented for sims given at the finite difference splines. Needed to est. covariance of derivatives. ')
        fisher_err = np.zeros([n_params,n_params])
        
        for i,param_name in enumerate(params_names):
            sims = self._dict_deriv_sims[param_name]
            if self._deriv_sim_ids is not None:
                sims = sims[:,self._deriv_sim_ids]
            if i==0:
                n_fisher_sims = sims.shape[1]
                derivs_comp_covMat = np.zeros([n_params,self._n_params,n_fisher_sims])

            compressed_deriv_ens = 0
            for finite_dif_index,finite_dif_weight in enumerate(self.deriv_finite_dif_weights[param_name]):
                tmp = self.compress_vector(sims[finite_dif_index])
                compressed_deriv_ens+=finite_dif_weight/self._dict_param_steps[param_name]*tmp

            compressed_deriv_mean = self.compress_vector(self.deriv_mean_fisher[param_name])
            derivs_comp_covMat[i] =(compressed_deriv_ens-compressed_deriv_mean).T

    
        # One factor of n_fisher_sims
        mix_matrix = np.einsum('ijk,mnk->imjn',derivs_comp_covMat,derivs_comp_covMat)/(n_fisher_sims-1)/n_fisher_sims

        covMat_comp = self._compute_fisher_matrix(self.param_names)
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(covMat_comp,mix_matrix[i,j]))
        return fisher_err
