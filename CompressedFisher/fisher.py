import numpy as np
import warnings
import scipy.linalg

central_difference_weights={
    2:np.array([-1/2.,+1/2.]),
    4:np.array([1/12.,-2./3,+2./3.,-1./12.]),
    6:np.array([-1./60,3./20,-3./4,0,3./4,-3./20,1./60])
}


def geometricMean(A,B):
    AB = np.linalg.solve(A,B)
    sqrt_AB = scipy.linalg.sqrtm(AB)
    return A.dot(sqrt_AB)

class baseFisher(object):

    def __init__(self):
        pass


    def _get_deriv_split_indices(self,f_split):
        indices = np.arange(self.n_sims_derivs)
        np.random.shuffle(indices)
        ids_a = indices[:int(self.n_sims_derivs*f_split)]
        ids_b = indices[int(self.n_sims_derivs*f_split):]
        return ids_a,ids_b

    def _get_covmat_split_indices(self,f_split):
        indices = np.arange(self.n_sims_covmat)
        np.random.shuffle(indices)
        ids_a = indices[:int(self.n_sims_covmat*f_split)]
        ids_b = indices[int(self.n_sims_covmat*f_split):]
        return ids_a,ids_b

    @property
    def deriv_finite_dif_weights(self):
        return self._deriv_finite_dif_weights


    def initailize_spline_weights(self,dict_param_spline_weights=None):
        if dict_param_spline_weights is None:
            self._deriv_finite_dif_weights ={}
            for param_name in self.param_names:
                self._deriv_finite_dif_weights[param_name] = central_difference_weights[self._deriv_spline_order]
        else:
            self._deriv_finite_dif_weights=dict_param_spline_weights

    def generate_deriv_sim_splits(self,compress_fraction=None,compress_number=None,ids_comp=None,ids_fish=None):
        if compress_fraction is None and compress_number is None:
            raise AssertionError(' Need to specify either the fraction of sims to use in the compression or the total number')
        if compress_fraction is None:
            compress_fraction = compress_number*1./self.n_sims_derivs   
        ids_comp,ids_fish = self._get_deriv_split_indices(compress_fraction)
        self._apply_deriv_split(ids_comp, ids_fish)
        
    def generate_covmat_sim_splits(self,compress_fraction=None,compress_number=None):
        if compress_fraction is None and compress_number is None:
            raise AssertionError(' Need to specify either the fraction of sims to use in the compression or the total number')
        if compress_fraction is None:
            compress_fraction = compress_number*1./self.n_sims_derivs  
        if not (self._store_covmat_sims):
            raise AssertionError('Cannot split covmat sims as these are not stored. Reinitalize covmat with store_covmat_sims set to True')

        ids_comp,ids_fish = self._get_covmat_split_indices(compress_fraction)
        self. _apply_covmat_split(ids_fish,ids_comp)


    def _compressed_fisher_matrix(self,compressed_derivs,compressed_covmat):
        n_params = compressed_derivs.shape[0]
        fisher = np.zeros([n_params,n_params])
        for i in range(n_params):
            for j in range(i+1):
                fisher[i,j] = fisher[j,i] = np.dot(compressed_derivs[i], np.linalg.solve(compressed_covmat,compressed_derivs[j]))
        return fisher

    def compute_fisher_forecast(self,params_names):
        return np.linalg.inv(self._compute_fisher_matrix(params_names))

    def est_fisher_bias(self,params_names):
        fish = self._compute_fisher_matrix(params_names)
        fish_err = self._compute_fisher_matrix_error(params_names)
        return np.linalg.solve(fish,np.linalg.solve(fish,fish_err).T)


    def compute_compressed_fisher_forecast(self,params_names):
        return np.linalg.inv(self._compute_compressed_fisher_matrix(params_names))

    def est_compressed_fisher_bias(self,params_names):
        fish = self._compute_compressed_fisher_matrix(params_names)
        fish_err = self._compute_compressed_fisher_matrix_error(params_names)
        return np.linalg.solve(fish,np.linalg.solve(fish,fish_err).T)

    def _compute_combined_fisher_matrix(self,params_names):
        fisher1 = self._compute_fisher_matrix(params_names)
        fisher2 = self._compute_compressed_fisher_matrix(params_names)
        return geometricMean(fisher1,fisher2)

    def compute_combined_fisher_forecast(self,params_names):
        return np.linalg.inv(self._compute_combined_fisher_matrix(params_names))


    def run_fisher_deriv_stablity_test(self,params_names,sample_fractions=None,verbose=True,max_repeats=None):
        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])
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
            for I in range(nRepeats):
                ids_fish = np.arange(I*n_split,(I+1)*n_split).astype('int')
                self._apply_deriv_split(None,ids_fish)
                tmp.append(self.compute_fisher_forecast(params_names))
            #print(tmp)
            mns[i]  = np.mean(tmp,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(tmp,axis=0)*np.sqrt(1/(len(tmp)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds

    def run_compressed_fisher_deriv_stablity_test(self,params_names,compress_fraction,sample_fractions=None,verbose=True,max_repeats=None):
        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])
        n_params = len(params_names)
        nSims = np.zeros(sample_fractions.shape[0])
        mns  = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        stds = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        for i,s_frac in enumerate(sample_fractions):
            tmp = []
            nRepeats  = int(1/s_frac)
            if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)
            n_split = s_frac*self.n_sims_derivs
            nSims[i] = n_split
            for I in range(nRepeats):
                ids_all = np.arange(I*n_split,(I+1)*n_split).astype('int')
                #indices = np.arange(self.n_sims_covmat)
                np.random.shuffle(ids_all)
                ids_comp = ids_all[:int(n_split*compress_fraction)]
                ids_fish = ids_all[int(n_split*compress_fraction):]

                self._apply_deriv_split(ids_comp,ids_fish)
                tmp.append(self.compute_compressed_fisher_forecast(params_names))
            #print(tmp)
            mns[i]  = np.mean(tmp,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(tmp,axis=0)*np.sqrt(1/(len(tmp)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds


    def run_combined_fisher_deriv_stablity_test(self,params_names,compress_fraction,sample_fractions=None,verbose=True,max_repeats=None):
        if sample_fractions is None:
            sample_fractions = np.array([.1,.2,1./3.,.4,.5,1.])
        n_params = len(params_names)
        nSims = np.zeros(sample_fractions.shape[0])
        mns  = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        stds = np.zeros([sample_fractions.shape[0],n_params,n_params ])
        for i,s_frac in enumerate(sample_fractions):
            tmp = []
            nRepeats  = int(1/s_frac)
            n_split = s_frac*self.n_sims_derivs
            nSims[i] = n_split
            if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)
            for I in range(nRepeats):
                ids_all = np.arange(I*n_split,(I+1)*n_split).astype('int')
                #indices = np.arange(self.n_sims_covmat)
                np.random.shuffle(ids_all)
                ids_comp = ids_all[:int(n_split*compress_fraction)]
                ids_fish = ids_all[int(n_split*compress_fraction):]
                self._apply_deriv_split(ids_comp,ids_fish)
                tmp.append(self.compute_combined_fisher_forecast(params_names))
            #print(tmp)
            mns[i]  = np.mean(tmp,axis=0)
            if nRepeats!=1:
                stds[i] = np.std(tmp,axis=0)*np.sqrt(1/(len(tmp)-1))
            if verbose: print(f"{nSims[i]} \n {np.diag(mns[i])**.5} \n {np.diag(stds[i])**.5}")
        return nSims,mns,stds




