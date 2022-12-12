import numpy as np
import warnings
import scipy.linalg
from ..fisher import baseFisher,central_difference_weights



class poissonFisher(baseFisher):
    def __init__(self,param_names,n_sims_derivs,n_sims_covmat=None,deriv_spline_order=None):
        
        
        self.param_names = param_names

        self.n_sims_derivs = n_sims_derivs
        self.n_sims_covmat = n_sims_covmat

        self._deriv_spline_order = deriv_spline_order
        self._n_params = len(self.param_names)
        self._deriv_rate_fisher = None
        self._covmat_fisher = None

        self._rate_comp = None

        self._deriv_rate_comp = None

        self._covmat_sims = None
        self._store_covmat_sims = False


        self._dict_deriv_sims = None
        self._dict_param_steps = None
        self._deriv_sim_ids = None
        self._has_deriv_sims = True

        self._deriv_rate_function = None
       
        self._deriv_finite_dif_weights=None

    @property
    def covmat_fisher(self):
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
        return self._covmat_fisher

    @property
    def rate_comp(self):
        if (self._deriv_rate_fisher is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.rate_comp = None
        return self._rate_comp

    @rate_comp.setter
    def rate_comp(self,ids):
        if ids is None:
            self._rate_comp = np.mean(self._covmat_sims,axis=0)
        else:
            self._rate_comp = np.mean(self._covmat_sims[ids],axis=0)
        return self._rate_comp

    @property
    def deriv_rate_fisher(self):
        if (self._deriv_rate_fisher is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_rate_fisher = None
        return self._deriv_rate_fisher

    @deriv_rate_fisher.setter
    def deriv_rate_fisher(self,ids):
        self._deriv_rate_fisher = {}
        for n in self.param_names:
            self._deriv_rate_fisher[n]=np.mean(self.get_deriv_rate_sims(n,ids),axis=0)
        self._deriv_sim_ids = ids
        return self._deriv_rate_fisher


    @property
    def deriv_rate_comp(self):
        if (self._deriv_rate_comp is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_rate_comp = None
        return self._deriv_rate_comp

    @deriv_rate_comp.setter
    def deriv_rate_comp(self,ids):
        self._deriv_rate_comp = {}
        for n in self.param_names:
            self._deriv_rate_comp[n]=np.mean(self.get_deriv_rate_sims(n,ids),axis=0)

        return self._deriv_rate_comp



    def initailize_covmat(self,covmat_sims,store_covmat_sims=True):
        self._store_covmat_sims = store_covmat_sims
        self._covmat_sims = covmat_sims
        self.covmat_fisher = None
        self.rate_comp = None
        if not store_covmat_sims:
            self._covmat_sims = None
        self.n_sims_covmat = covmat_sims.shape[0]

    def initailize_deriv_sims(self,dic_deriv_sims=None,dict_param_steps=None,deriv_rate_function=None):
        if dic_deriv_sims is None:
            assert(deriv_rate_function is not None)
            self._has_deriv_sims = False
            self._deriv_rate_function = deriv_rate_function
        else:
            self._has_deriv_sims = True
            self._dict_deriv_sims = dic_deriv_sims
            if self._deriv_spline_order is not None:
                assert(dict_param_steps is not None)
                self._dict_param_steps = dict_param_steps

    def get_deriv_rate_sims(self,param_name,ids):
        if not self._has_deriv_sims:
            if ids is None:
                ids = np.arange(self.n_sims_derivs)
            nSims = len(ids)
            for i,id0 in enumerate(ids):
                sim = self._deriv_rate_function(param_name,id0)
                if i==0:
                    sims = np.zeros((nSims,)+sim.shape)
                sims[i] = sim
        else:
            sims = self._dict_deriv_sims[param_name]
            if ids is not None:
                if self._deriv_spline_order is None:
                    sims = sims[ids]
                else:
                    sims = sims[:,ids]
            if self._deriv_spline_order is not None:
                if self._deriv_finite_dif_weights is None:
                    self.initailize_spline_weights()
                sims = np.einsum('i,ijk->jk', self.deriv_finite_dif_weights[param_name],sims)/self._dict_param_steps[param_name]
        return sims



    def _apply_deriv_split(self,ids_comp,ids_fish):
        self.deriv_rate_comp = ids_comp
        self.deriv_rate_fisher = ids_fish


    def _apply_covmat_split(self,ids_fish,ids_comp):
        self.covmat_fisher = ids_fish
        self.rate_comp = ids_comp



    def compress_vector(self,data,with_rate=False):

        data = np.atleast_2d(data)
        results = np.zeros([data.shape[0],self._n_params])
        if with_rate:
            data = data -self._rate_comp
        for i,n in enumerate(self.param_names):
            deriv = self.deriv_rate_comp[n]/self.rate_comp
            # np.sum((data-mu)*mu_deriv[i],axis=-1)
            results[:,i]= np.sum(data*deriv,axis=-1)
        return results

    def _compute_fisher_matrix(self,params_names=None):
        if params_names is None: params_names = self.params_names
        n_params = len(params_names)
        fisher = np.zeros([n_params,n_params])
        for i,n1 in enumerate(params_names):
            for j,n2 in enumerate(params_names[:i+1]):
                fisher[i,j] = fisher[j,i] = np.sum(self.covmat_fisher*self.deriv_rate_fisher[n1]/self.rate_comp*self.deriv_rate_fisher[n2]/self.rate_comp)
        return fisher
   

    def compute_deriv_rate_covar(self,params_names,input_ids=False):
        n_params = len(params_names)
        if input_ids is False:
            input_ids = self._deriv_sim_ids
        for i,n1 in enumerate(params_names):
            derivs1_ens = self.get_deriv_rate_sims(n1,input_ids)
            derivs1_mn = np.mean(derivs1_ens,axis=0)
            if i==0:
                nSims = derivs1_ens.shape[0]
                
                deriv_covmat = np.zeros([n_params,n_params,self.dim,])
            for j,n2 in enumerate(params_names[:i+1]):
                derivs2_ens = self.get_deriv_rate_sims(n2,input_ids)
                derivs2_mn = np.mean(derivs2_ens,axis=0)
                deriv_covmat[i,j] = deriv_covmat[j,i]= 1/(nSims-1)*np.einsum('ik,ik->k',(derivs1_ens-derivs1_mn),(derivs2_ens-derivs2_mn) )
        return deriv_covmat/nSims

    def _compute_fisher_matrix_error(self,params_names=None):
        if params_names is None: params_names = self.params_names

        n_params = len(params_names)


        derivs_covMat = self.compute_deriv_rate_covar(params_names)

        fisher_err = np.zeros([n_params,n_params])
        #hartlap_fac = (nSimsCovMat-nKs-2)/(nSimsCovMat-1)
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.sum(derivs_covMat[i,j]*self.covmat_fisher/self.rate_comp/self.rate_comp)
        return fisher_err
# def numericalDerivFisher(deltaParams,covMatSims,pairedSims,nSims,nSimsCovMat):
#     #covMat_full_est = generateSims(params,nSimsCovMat)
#     covMat_full_est = (np.var(covMatSims,axis=0))
#     mu = np.mean(covMat_full_est,axis=0)
#     #covMat_full_est=mu = pk_theory(ks,params)
#     (derivs,derivs_covMat) = est_numerical_derivs_fromPaired(deltaParams,pairedSims,nSims,doCov=True)
# #     derivs = theoretical_derivs(params)
# #     derivs/=mu
#     fisher = np.zeros([nParams,nParams])
#     fisher_err = np.zeros([nParams,nParams])
#     for i in range(nParams):
#         for j in range(i+1):
#             fisher[i,j] = fisher[j,i] = np.sum(covMat_full_est*derivs[i]*derivs[j])
# #            tmp_b = np.linalg.solve(covMat_full_est,derivs_covMat[j])*hartlapfac
#             fisher_err[i,j] = fisher_err[j,i] = np.sum(covMat_full_est*derivs_covMat[i,j])

# #            fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(covMat_full_est,derivs_mu_covMat[i,j]))
#     return fisher,fisher_err





    def _compute_compressed_fisher_matrix(self,params_names=None):
        if params_names is None: params_names = self.params_names
        n_params = len(params_names)
        fisher = self._compute_fisher_matrix(params_names)
        fisher_comp = np.zeros([n_params,n_params])
        for i,n1 in enumerate(params_names):
            deriv_a = self.compress_vector(self.deriv_rate_fisher[n1],with_rate=False)
            for j,n2 in enumerate(params_names[:i+1]):
                deriv_b = self.compress_vector(self.deriv_rate_fisher[n2],with_rate=False)
                fisher_comp[i,j] = fisher_comp[j,i] = deriv_a.dot(np.linalg.solve(fisher,deriv_b.T))
        return fisher_comp
   
    def _compute_compressed_fisher_matrix_error(self,params_names=None):
        if params_names is None: params_names = self.params_names
        n_params = len(params_names)       
        if not self._has_deriv_sims or self._deriv_spline_order is None or self._deriv_spline_order in [0,1]:
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
            for finite_dif_index,finite_dif_weight in enumerate(self._deriv_finite_dif_weights[param_name]):
                tmp = self.compress_vector(sims[finite_dif_index],with_rate=False)
                compressed_deriv_ens+=finite_dif_weight/self._dict_param_steps[param_name]*tmp

            compressed_rate_mean = self.compress_vector(self.deriv_rate_fisher[param_name],with_rate=False)
            #comp_up = #compression(nSimsCovMat,comp_deriv_sims_deriv_est[:,0,i],mu,covMat_full_est,deriv_mu_compressor,deriv_covmat_compressor)
            #comp_down = #compression(nSimsCovMat,comp_deriv_sims_deriv_est[:,1,i],mu,covMat_full_est,deriv_mu_compressor,deriv_covmat_compressor)
            derivs_comp_covMat[i] =(compressed_deriv_ens-compressed_rate_mean).T

    
        # One factor of n_fisher_sims
        mix_matrix = np.einsum('ijk,mnk->imjn',derivs_comp_covMat,derivs_comp_covMat)/(n_fisher_sims-1)/n_fisher_sims#(derivs_comp_covMat.shape[-1]-1)/comp_deriv_sims_deriv_est.shape[0]

        covMat_comp = self._compute_fisher_matrix(self.param_names)
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(covMat_comp,mix_matrix[i,j]))
        return fisher_err