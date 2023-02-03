import numpy as np
import warnings
import scipy.linalg

central_difference_weights={
    1: np.array([1.]),
    2:np.array([-1/2.,+1/2.]),
    4:np.array([1/12.,-2./3,+2./3.,-1./12.]),
    6:np.array([-1./60,3./20,-3./4,0,3./4,-3./20,1./60])
}


def geometricMean(A,B):
    AB = np.linalg.solve(A,B)
    sqrt_AB = scipy.linalg.sqrtm(AB)
    return A.dot(sqrt_AB)

class baseFisher(object):
    """ The base class for Fisher forecasting
    
    This class contains common routines for all the Fisher forecasts including: 
    compressed and combined Fisher forecasts, convergence tests and more.
    """

    def __init__(self):
        pass


    def _get_deriv_split_indices(self,f_split):
        """
        A function to compute get the ids of the two splits of the derivative simulations
        
        Args:
            f_split ([float]): Fraction of sims (between 0 and 1)to assign to split 0 and split 1
        
        Returns:
            [list,list]: Two lists containing the simulations assigned to split 0 and split 1
        """
        indices = np.arange(self.n_sims_derivs)
        np.random.shuffle(indices)
        ids_a = indices[:int(self.n_sims_derivs*f_split)]
        ids_b = indices[int(self.n_sims_derivs*f_split):]
        return ids_a,ids_b

    def _get_covmat_split_indices(self,f_split):
        """
        A function to compute get the ids of the two splits of the covariance matrix simulations
        
        Args:
            f_split ([float]): Fraction of sims (between 0 and 1)to assign to split 0 and split 1
        
        Returns:
            [list,list]: Two lists containing the simulations assigned to split 0 and split 1
        """
        indices = np.arange(self.n_sims_covmat)
        np.random.shuffle(indices)
        ids_a = indices[:int(self.n_sims_covmat*f_split)]
        ids_b = indices[int(self.n_sims_covmat*f_split):]
        return ids_a,ids_b

    @property
    def deriv_finite_dif_weights(self):
        return self._deriv_finite_dif_weights


    def initailize_spline_weights(self,dict_param_spline_weights=None):
        """
        Set the weights for the splines for the numerical derivatives.

        
        Args:
            dict_param_spline_weights ([dictionary]): The per parameter weights for the derivatives.  (default: `None`)
                                                      If not argument is supplied the default weights, given on line 5 are used.
        """
        if dict_param_spline_weights is None:
            self._deriv_finite_dif_weights ={}
            for param_name in self.param_names:
                self._deriv_finite_dif_weights[param_name] = central_difference_weights[self._deriv_finite_dif_accuracy]
        else:
            self._deriv_finite_dif_weights=dict_param_spline_weights

    def generate_deriv_sim_splits(self,compress_fraction=None,compress_number=None,ids_comp=None,ids_fish=None):
        """
        Perform the split of the sims into the set for computing the compression and the set for computing the derivatives.
        There are three different ways.
        1) To specify some fraction of the total number of simulations to be used for compression (compress_fraction)
        2) To specfiy a specific number of simulations to be used for the compression (compress_number)
        3) To specify specificly which sims should be used for which part (ids_comp and ids_fish)
        
        Args:
            compress_fraction ([float]): For mode 1) The fraction of simulations to be used for the compression(default: `None`)
            compress_number ([int]): For mode 2) The number of simulations to be used for the compression (default: `None`)
            ids_comp ([list]): For mode 3) IDs of simulations to be used for computing the compression (default: `None`)
            ids_fish ([list]): For mode 3) IDs of simulations to be used for computing the fisher information (default: `None`)

        """
        if compress_fraction is None and compress_number is None:
            raise AssertionError(' Need to specify either the fraction of sims to use in the compression or the total number')
        if compress_fraction is None:
            compress_fraction = compress_number*1./self.n_sims_derivs   
        ids_comp,ids_fish = self._get_deriv_split_indices(compress_fraction)
        self._apply_deriv_split(ids_comp, ids_fish)
        
    def generate_covmat_sim_splits(self,compress_fraction=None,compress_number=None):
        """
        A function to split the covariance matrix simulations between the compression and the fisher evaluation.
        Typically this is not needed as the 'noise' on the covariance matrix is typically subdominant.
        Two modes
        1) To specify some fraction of the total number of simulations to be used for compression (compress_fraction)
        2) To specfiy a specific number of simulations to be used for the compression (compress_number)
        
        Args:
            compress_fraction ([float]): For mode 1) The fraction of simulations to be used for the compression(default: `None`)
            compress_number ([int]): For mode 2) The number of simulations to be used for the compression (default: `None`)

        """
        if compress_fraction is None and compress_number is None:
            raise AssertionError(' Need to specify either the fraction of sims to use in the compression or the total number')
        if compress_fraction is None:
            compress_fraction = compress_number*1./self.n_sims_derivs  
        if not (self._store_covmat_sims):
            raise AssertionError('Cannot split covmat sims as these are not stored. Reinitalize covmat with store_covmat_sims set to True')

        ids_comp,ids_fish = self._get_covmat_split_indices(compress_fraction)
        self. _apply_covmat_split(ids_fish,ids_comp)


    def _compressed_fisher_matrix(self,compressed_derivs,compressed_covmat):
        """
        Compute the compressed fisher matrix using eq. 10 in Coulton and Wandelt

        
        Args:
            compressed_derivs ([matrix n_params, dimension): The compressed derivatives for each parametered
            compressed_covmat ([matrix dimension, dimension]): The comprsesed covariance matrix
        
        Returns:
            [matrix n_params, n_params]: The compressed fisher information.
        """
        n_params = compressed_derivs.shape[0]
        fisher = np.zeros([n_params,n_params])
        for i in range(n_params):
            for j in range(i+1):
                fisher[i,j] = fisher[j,i] = np.dot(compressed_derivs[i], np.linalg.solve(compressed_covmat,compressed_derivs[j]))
        return fisher

    def compute_fisher_forecast(self,params_names):
        """
        Compute the standard fisher forecast parameter covariances
        Eq. 5 in Coulton and Wandelt
        
        Args:
            params_names ([list]): A list of which parameters to include in the forecast.
        
        Returns:
            [matrix]: The forecast parameter covariances
        """
        return np.linalg.inv(self._compute_fisher_matrix(params_names))

    def est_fisher_forecast_bias(self,params_names):
        """
        Estimate the bias to the standard fisher forecast parameter variances. 
        The ratio of this to the parameter *variances* gives a measure of whether there are enough
        simulations for the forecast to be converged
        Computed Eq. 6 and Eq. 42. in Coulton and Wandelt
        
        Args:
            params_names ([list]): Parameters to include in the fisher forecast
        
        Returns:
            [matrix]: The bias terms to each element in the fisher forecast. 
        """
        fish = self._compute_fisher_matrix(params_names)
        fish_err = self._compute_fisher_matrix_error(params_names)
        return np.linalg.solve(fish,np.linalg.solve(fish,fish_err).T)


    def compute_compressed_fisher_forecast(self,params_names):
        """
        Compute the compressed fisher forecast parameter covariances
        The inverse of Eq. 10 in Coulton and Wandelt
        
        Args:
            params_names ([list]): A list of which parameters to include in the forecast.
        
        Returns:
            [matrix]: The compressed forecast parameter covariances
        """
        return np.linalg.inv(self._compute_compressed_fisher_matrix(params_names))

    def est_compressed_fisher_forecast_bias(self,params_names):
        """
        Estimate the bias to the compressed fisher forecast parameter variances. 
        The ratio of this to the compressed fisher parameter *variances* gives a measure of whether there 
        are enough simulations for the forecast to be converged.
        Computed with Eq. 11  and Eq. 42 in Coulton and Wandelt
        
        Args:
            params_names ([list]): Parameters to include in the fisher forecast
        
        Returns:
            [matrix]: The bias terms to each element in the fisher forecast. 
        """
        fish = self._compute_compressed_fisher_matrix(params_names)
        fish_err = self._compute_compressed_fisher_matrix_error(params_names)
        return np.linalg.solve(fish,np.linalg.solve(fish,fish_err).T)

    def _compute_combined_fisher_matrix(self,params_names):
        """ Computes the combined fisher information
        Eq. 18 in Coulton and Wandelt
        
        Args:
            params_names ([list]): list of the parameters that you want to included in the Fisher forecast. 
        
        Returns:
            [np.array([n_params,n_params])]: An matrix or size ([n_params,n_params]) with the Fisher information
        """
        fisher1 = self._compute_fisher_matrix(params_names)
        fisher2 = self._compute_compressed_fisher_matrix(params_names)
        return geometricMean(fisher1,fisher2)

    def compute_combined_fisher_forecast(self,params_names):
        """ Computes the combined fisher parameter constraints
        The inverse of Eq. 18 in Coulton and Wandelt
        
        Args:
            params_names ([list]): list of the parameters that you want to included in the Fisher forecast. 
        
        Returns:
            [np.array([n_params,n_params])]: An matrix or size ([n_params,n_params]) with the forecast Fisher covariance matrix.
        """
        return np.linalg.inv(self._compute_combined_fisher_matrix(params_names))


    def run_fisher_deriv_stablity_test(self,params_names,sample_fractions=None,verbose=False,max_repeats=None):
        """
        This function provides a second means to test the convergence of the standard Fisher forecasts.
        Fisher forecasts are performed using only a subset of the total available set of derivative simulations (as set by sample_fractions).
        If the size of the forecasts increases rapidly with the number of simulations, it is a sign that the Fisher forecast is not converged.
        
        For small subsets the Fisher forecast is repeated using as different divisions, by default all the possibe divissions. Eg. 2 for a sample_fraction of 1/2.

        Args:
            params_names ([list of names]): A list of the parameters to consider in the Fisher forecast
            sample_fractions (array of floats): An array of fractions (between 0 and 1) specifing the subsets of the total number of simulations to consider.
                                                If no arguement is given, use a default set of (1/10,1/5,1/3,2/5,1/2,1)  (default: `None`)
            verbose (bool): Print the Fisher forecast values for each sample fraction as they are evaluated (default: `False`)
            max_repeats (int): Limit the number of subdivisions to consider. E.g. for a sample fraction of 0.1 there are 10 divisions of the data. 
                               Leaving this as None will use all 10, however setting it to say 3 would mean only 3 of these are considered. (default: `None`)
        
        Returns:
            3 arrays : The number of simulations at each sample_fraction, the mean fisher information at each sample_fraction and the error on the mean fisher information (computed using the subsets)
        """
        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])
        sample_fractions = np.atleast_1d(sample_fractions)
        n_params = len(params_names)
        nSims = np.zeros(sample_fractions.shape[0])
        mns  = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        stds = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        for i,s_frac in enumerate(sample_fractions):
            tmp = []
            nRepeats  = int(1/s_frac)
            n_split = s_frac*self.n_sims_derivs
            if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)
            nSims[i] = n_split
            ids_all = np.arange(0,self.n_sims_derivs)
            np.random.shuffle(ids_all)
            for I in range(nRepeats):
                ids_fish = ids_all[int(I*n_split):int((I+1)*n_split)]#np.arange(I*n_split,(I+1)*n_split).astype('int')
                self._apply_deriv_split(None,ids_fish)
                tmp.append(self.compute_fisher_forecast(params_names))
            #print(tmp)
            mns[i]  = np.median(tmp,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(tmp,axis=0)*np.sqrt(1/(len(tmp)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds

    def run_compressed_fisher_deriv_stablity_test(self,params_names,compress_fraction,sample_fractions=None,verbose=False,max_repeats=None):
        """
        This function provides a second means to test the convergence of the compressed Fisher forecasts.
        The compressed fisher forecasts are performed using only a subset of the total available set of derivative simulations (as set by sample_fractions).
        If the size of the forecasts increases rapidly with the number of simulations, it is a sign that the Fisher forecast is not converged.
        If the size of the compressed decreases wit the number of simulations, it is actually a sing the compressed forecast is reliable 

        For small subsets the Fisher forecast is repeated using as different divisions, by default all the possibe divissions. Eg. 2 for a sample_fraction of 1/2.

        Args:
            params_names ([list of names]): A list of the parameters to consider in the Fisher forecast
            compress_fraction (float) : The fraction of the total sims used in the compression.
            sample_fractions (array of floats): An array of fractions (between 0 and 1) specifing the subsets of the total number of simulations to consider.
                                                If no arguement is given, use a default set of (1/10,1/5,1/3,2/5,1/2,1)  (default: `None`)
            verbose (bool): Print the Fisher forecast values for each sample fraction as they are evaluated (default: `False`)
            max_repeats (int): Limit the number of subdivisions to consider. E.g. for a sample fraction of 0.1 there are 10 divisions of the data. 
                               Leaving this as None will use all 10, however setting it to say 3 would mean only 3 of these are considered. (default: `None`)
        
        Returns:
            3 arrays : The number of simulations at each sample_fraction, the mean fisher information at each sample_fraction and the error on the mean fisher information (computed using the subsets)
        """

        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])
        n_params = len(params_names)
        nSims = np.zeros(sample_fractions.shape[0])
        mns  = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        stds = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        for i,s_frac in enumerate(sample_fractions):
            repeats_sfracs = []
            nRepeats  = int(1/s_frac)
            if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)
            n_split = s_frac*self.n_sims_derivs
            nSims[i] = n_split
            ids_all = np.arange(0,self.n_sims_derivs)
            np.random.shuffle(ids_all)
            for I in range(nRepeats):
                ids_rpt = ids_all[int(I*n_split):int((I+1)*n_split)]#np.arange(I*n_split,(I+1)*n_split).astype('int')
                #indices = np.arange(self.n_sims_covmat)
                np.random.shuffle(ids_rpt)
                ids_comp = ids_rpt[:int(n_split*compress_fraction)]
                ids_fish = ids_rpt[int(n_split*compress_fraction):]

                self._apply_deriv_split(ids_comp,ids_fish)
                repeats_sfracs.append(self.compute_compressed_fisher_forecast(params_names))
            #print(tmp)
            mns[i]  = np.median(repeats_sfracs,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(repeats_sfracs,axis=0)*np.sqrt(1/(len(repeats_sfracs)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds


    def run_combined_fisher_deriv_stablity_test(self,params_names,compress_fraction,sample_fractions=None,verbose=False,max_repeats=None):
        """
        This function provides a second means to test the convergence of the compbined Fisher forecasts.
        The combined fisher forecasts are performed using only a subset of the total available set of derivative simulations (as set by sample_fractions).
        The combined forecast should be roughly constant to changes of the number of simulations, when it is converged.

        For small subsets the Fisher forecast is repeated using as different divisions, by default all the possibe divissions. Eg. 2 for a sample_fraction of 1/2.

        Args:
            params_names ([list of names]): A list of the parameters to consider in the Fisher forecast
            compress_fraction (float) : The fraction of the total sims used in the compression.
            sample_fractions (array of floats): An array of fractions (between 0 and 1) specifing the subsets of the total number of simulations to consider.
                                                If no arguement is given, use a default set of (1/10,1/5,1/3,2/5,1/2,1)  (default: `None`)
            verbose (bool): Print the Fisher forecast values for each sample fraction as they are evaluated (default: `False`)
            max_repeats (int): Limit the number of subdivisions to consider. E.g. for a sample fraction of 0.1 there are 10 divisions of the data. 
                               Leaving this as None will use all 10, however setting it to say 3 would mean only 3 of these are considered. (default: `None`)
        
        Returns:
            3 arrays : The number of simulations at each sample_fraction, the mean fisher information at each sample_fraction and the error on the mean fisher information (computed using the subsets)
        """

        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])

        n_params = len(params_names)
        nSims = np.zeros(sample_fractions.shape[0])
        mns  = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        stds = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        for i,s_frac in enumerate(sample_fractions):
            repeats_sfracs = []
            nRepeats  = int(1/s_frac)
            n_split = s_frac*self.n_sims_derivs
            nSims[i] = n_split
            if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)

            ids_all = np.arange(0,self.n_sims_derivs)
            np.random.shuffle(ids_all)
            for I in range(nRepeats):
                ids_rpt = ids_all[int(I*n_split):int((I+1)*n_split)]#np.arange(I*n_split,(I+1)*n_split).astype('int')
                #indices = np.arange(self.n_sims_covmat)
                np.random.shuffle(ids_rpt)
                ids_comp = ids_rpt[:int(n_split*compress_fraction)]
                ids_fish = ids_rpt[int(n_split*compress_fraction):]
                self._apply_deriv_split(ids_comp,ids_fish)
                repeats_sfracs.append(self.compute_combined_fisher_forecast(params_names))
            #print(tmp)
            mns[i]  = np.median(repeats_sfracs,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(repeats_sfracs,axis=0)*np.sqrt(1/(len(repeats_sfracs)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds


    def compute_compressed_fisher_forecast_wShuffle(self,params_names,compress_fraction,nShuffles=10,verbose=False):
        """ 
        Computes the compressed fisher parameter constraints and this version uses repeated estimates (with the number given by nShuffles)
        where for each repeat the sims are shuffled and redivded between the compression and fisher estimation. 

        This reduces the variance of the estimator, but requires more simulations for it to be valid.
        
        
        Args:
            params_names ([list]): list of the parameters that you want to included in the Fisher forecast. 
            compress_fraction (float): the fraction (between 0 and 1) of the simulations to use for the compression
            nShuffles (int): the number of times to iterate. Each iteration redivides the simulations and reestimates the fisher constraints
        
        Returns:
            [np.array([n_params,n_params])]: An matrix or size ([n_params,n_params]) with the forecast Fisher covariance matrix.
        """

        n_params = len(params_names)
        results  = np.zeros([nShuffles,n_params,n_params ])
        for i in range(nShuffles):
            ids_all = np.arange(self.n_sims_derivs)
            np.random.shuffle(ids_all)
            ids_comp = ids_all[:int(self.n_sims_derivs*compress_fraction)]
            ids_fish = ids_all[int(self.n_sims_derivs*compress_fraction):]
            self._apply_deriv_split(ids_comp,ids_fish)
            results[i] = self.compute_compressed_fisher_forecast(params_names)
        if verbose==True: 
            print(np.diag(np.mean(results,axis=0))**.5,np.diag(np.std(results,axis=0))**.5) 
        return np.median(results,axis=0)


    def compute_combined_fisher_forecast_wShuffles(self,params_names,compress_fraction,nShuffles=10,verbose=False):
        """ 
        Computes the combined fisher parameter constraints and this version uses repeated estimates (with the number given by nShuffles)
        where for each repeat the sims are shuffled and redivded between the compression and fisher estimation. 

        This reduces the variance of the estimator, but requires more simulations for it to be unbiased.
        
        
        Args:
            params_names ([list]): list of the parameters that you want to included in the Fisher forecast. 
            compress_fraction (float): the fraction (between 0 and 1) of the simulations to use for the compression
            nShuffles (int): the number of times to iterate. Each iteration redivides the simulations and reestimates the fisher constraints
        
        Returns:
            [np.array([n_params,n_params])]: An matrix or size ([n_params,n_params]) with the forecast Fisher covariance matrix.
        """

        n_params = len(params_names)
        results  = np.zeros([nShuffles,n_params,n_params ])
        for i in range(nShuffles):
            ids_all = np.arange(self.n_sims_derivs)
            np.random.shuffle(ids_all)
            ids_comp = ids_all[:int(self.n_sims_derivs*compress_fraction)]
            ids_fish = ids_all[int(self.n_sims_derivs*compress_fraction):]
            self._apply_deriv_split(ids_comp,ids_fish)
            fisher1 = self._compute_fisher_matrix(params_names)
            fisher2 = self._compute_compressed_fisher_matrix(params_names)
            results[i] = geometricMean(fisher1,fisher2)
        if verbose==True: 
            print(np.diag(np.mean(results,axis=0))**.5,np.diag(np.std(results,axis=0))**.5) 
        return np.linalg.inv(np.mean(results,axis=0))



    def run_combined_wShuffles_fisher_deriv_stablity_test(self,params_names,compress_fraction,nShuffles=10,sample_fractions=None,verbose=True,max_repeats=None):
        """
        This function provides a second means to test the convergence of the compbined Fisher forecasts using the reshuffling.
        There is an iterative step, where for each of nShuffles repeats the sims are shuffled and redivded between the compression and fisher estimation. 
        The combined fisher forecasts are performed using only a subset of the total available set of derivative simulations (as set by sample_fractions).
        The combined forecast should be roughly constant to changes of the number of simulations, when it is converged.

        For small subsets the Fisher forecast is repeated using as different divisions, by default all the possibe divissions. Eg. 2 for a sample_fraction of 1/2.

        Args:
            params_names ([list of names]): A list of the parameters to consider in the Fisher forecast
            compress_fraction (float) : The fraction of the total sims used in the compression.
            nShuffles (int): The number of redivisions of the simulations between compression and Fisher estimation to perform.
            sample_fractions (array of floats): An array of fractions (between 0 and 1) specifing the subsets of the total number of simulations to consider.
                                                If no arguement is given, use a default set of (1/10,1/5,1/3,2/5,1/2,1)  (default: `None`)
            verbose (bool): Print the Fisher forecast values for each sample fraction as they are evaluated (default: `False`)
            max_repeats (int): Limit the number of subdivisions to consider. E.g. for a sample fraction of 0.1 there are 10 divisions of the data. 
                               Leaving this as None will use all 10, however setting it to say 3 would mean only 3 of these are considered. (default: `None`)
        
        Returns:
            3 arrays : The number of simulations at each sample_fraction, the mean fisher information at each sample_fraction and the error on the mean fisher information (computed using the subsets)
        """

        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])

        n_params = len(params_names)
        nSims = np.zeros(sample_fractions.shape[0])
        mns  = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        stds = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        for i,s_frac in enumerate(sample_fractions):
            repeats_sfracs = []
            nRepeats  = int(1/s_frac)
            n_split = s_frac*self.n_sims_derivs
            nSims[i] = n_split
            ids_all = np.arange(0,self.n_sims_derivs)
            np.random.shuffle(ids_all)
            if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)
            for I in range(nRepeats):
                ids_rpt = ids_all[int(I*n_split):int((I+1)*n_split)]#np.arange(I*n_split,(I+1)*n_split).astype('int')
                #indices = np.arange(self.n_sims_covmat)
                mean_accummilate_Folds=0
                for J in range(nShuffles):
                    np.random.shuffle(ids_rpt)
                    ids_comp = ids_rpt[:int(n_split*compress_fraction)]
                    ids_fish = ids_rpt[int(n_split*compress_fraction):]
                    self._apply_deriv_split(ids_comp,ids_fish)
                    fisher1 = self._compute_fisher_matrix(params_names)
                    fisher2 = self._compute_compressed_fisher_matrix(params_names)
                    mean_accummilate_Folds +=1/nShuffles* geometricMean(fisher1,fisher2)#self.compute_combined_fisher_forecast(params_names)
                repeats_sfracs.append(np.linalg.inv(mean_accummilate_Folds))
            #print(tmp)
            mns[i]  = np.median(repeats_sfracs,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(repeats_sfracs,axis=0)*np.sqrt(1/(len(repeats_sfracs)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds
