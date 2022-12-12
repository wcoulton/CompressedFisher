import numpy as np
import warnings
import scipy.linalg
from ..fisher import baseFisher,central_difference_weights



class gaussianFisher(baseFisher):
    def __init__(self,param_names,n_sims_derivs,include_deriv_covmat=False,n_sims_covmat=None,deriv_spline_order=None):
        
        
        self.param_names = param_names
        self.include_deriv_covmat = include_deriv_covmat


        self.n_sims_derivs = n_sims_derivs
        self.n_sims_covmat = n_sims_covmat

        self._deriv_spline_order = deriv_spline_order
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
        if self.include_deriv_covmat:
            self.initailize_mean(self._covmat_sims[ids])

    @property
    def deriv_mean_fisher(self):
        if (self._deriv_mean_fisher is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_mean_fisher = None
        return self._deriv_mean_fisher

    @deriv_mean_fisher.setter
    def deriv_mean_fisher(self,ids):
        self._deriv_mean_fisher = {}
        for n in self.param_names:
            self._deriv_mean_fisher[n]=np.mean(self.get_deriv_mean_sims(n,ids),axis=0)
        self._deriv_sim_ids = ids
        return self._deriv_mean_fisher


    @property
    def deriv_mean_comp(self):
        if (self._deriv_mean_comp is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_mean_comp = None
        return self._deriv_mean_comp

    @deriv_mean_comp.setter
    def deriv_mean_comp(self,ids):
        self._deriv_mean_comp = {}
        for n in self.param_names:
            self._deriv_mean_comp[n]=np.mean(self.get_deriv_mean_sims(n,ids),axis=0)

        return self._deriv_mean_comp



    @property
    def deriv_covmat_fisher(self):
        if (self._deriv_covmat_fisher is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_covmat_fisher = None
        return self._deriv_covmat_fisher

    @deriv_covmat_fisher.setter
    def deriv_covmat_fisher(self,ids):
        self._deriv_covmat_fisher = {}
        for param_name in self.param_names:
            if self._has_deriv_sims is False or self._deriv_spline_order is None:
                self._deriv_covmat_fisher[param_name]=np.mean(self.get_deriv_covmat_sims(param_name,ids),axis=0)
            else:
                sims = self._dict_deriv_sims[param_name]
                if ids is not None:
                    sims = sims[:,ids]
                if self._deriv_finite_dif_weights is None:
                    self.initailize_spline_weights()
                sim_covs = 0
                for finite_dif_index,finite_dif_weight in enumerate(self._deriv_finite_dif_weights[param_name]):
                    sim_covs+=finite_dif_weight/self._dict_param_steps[param_name]*np.cov(sims[finite_dif_index].T)
                self._deriv_covmat_fisher[param_name] =sim_covs
        self._deriv_sim_ids = ids
        return self._deriv_covmat_fisher


    @property
    def deriv_covmat_comp(self):
        if (self._deriv_covmat_comp is None):
            warnings.warn('No division of sims given. All sims will be used in the compression')
            self.deriv_covmat_comp = None
        return self._deriv_covmat_comp

    @deriv_covmat_comp.setter
    def deriv_covmat_comp(self,ids):
        self._deriv_covmat_comp = {}
        for param_name in self.param_names:
            if self._has_deriv_sims is False or self._deriv_spline_order is None:
                self._deriv_covmat_comp[param_name]=np.mean(self.get_deriv_covmat_sims(param_name,ids),axis=0)
            else:
                sims = self._dict_deriv_sims[param_name]
                if ids is not None:
                    sims = sims[:,ids]
                if self._deriv_finite_dif_weights is None:
                    self.initailize_spline_weights()
                sim_covs = 0
                for finite_dif_index,finite_dif_weight in enumerate(self._deriv_finite_dif_weights[param_name]):
                    sim_covs+=finite_dif_weight/self._dict_param_steps[param_name]*np.cov(sims[finite_dif_index].T)
                self._deriv_covmat_comp[param_name] =sim_covs
        return self._deriv_covmat_comp



    def initailize_covmat(self,covmat_sims,store_covmat_sims=False):
        self._store_covmat_sims = store_covmat_sims
        self._covmat_sims = covmat_sims
        self.covmat_fisher = None
        self.covmat_comp = None
        if not store_covmat_sims:
            self._covmat_sims = None
        self.n_sims_covmat = covmat_sims.shape[0]
        if self.include_deriv_covmat and self._mean_comp is None:
            self.initailize_mean(covmat_sims)

    def initailize_mean(self,mean_sims):
        self._mean_comp = np.mean(mean_sims,axis=0)

    def initailize_deriv_sims(self,dic_deriv_sims=None,dict_param_steps=None,deriv_mean_function=None,deriv_covmat_function=None):
        if dic_deriv_sims is None:
            assert(deriv_mean_function is not None)
            self._has_deriv_sims = False
            self._deriv_mean_function = deriv_mean_function
            if self.include_deriv_covmat:
                assert(deriv_covmat_function is not None)
                self._deriv_covmat_function = deriv_covmat_function
        else:
            self._has_deriv_sims = True
            self._dict_deriv_sims = dic_deriv_sims
            if self._deriv_spline_order is not None:
                assert(dict_param_steps is not None)
                self._dict_param_steps = dict_param_steps


    def get_deriv_mean_sims(self,param_name,ids):
        if not self._has_deriv_sims:
            if ids is None:
                ids = np.arange(self.n_sims_derivs)
            nSims = len(ids)
            for i,id0 in enumerate(ids):
                sim = self._deriv_mean_function(param_name,id0)
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

    def get_deriv_covmat_sims(self,param_name,ids):
        if not self._has_deriv_sims:
            if ids is None:
                ids = np.arange(self.n_sims_derivs)
            nSims = len(ids)
            for i,id0 in enumerate(ids):
                sim = self._deriv_covmat_function(param_name,id0)
                if i==0:
                    sims = np.zeros((nSims,)+sim.shape)
                sims[i] = sim
        else:
            sims = self._dict_deriv_sims[param_name]
            if self._deriv_spline_order is not None:
               raise AssertionError("If finite difference sims are passed, it is easier to compute the covariance matrix of p+delta and at p-delta and compute the derivative")

            if ids is not None:
                if self._deriv_spline_order is None:
                    sims = sims[ids]
                else:
                    sims = sims[:,ids]
        return sims




    def _apply_deriv_split(self,ids_comp,ids_fish):
        self.deriv_mean_comp = ids_comp
        self.deriv_mean_fisher = ids_fish
        if self.include_deriv_covmat:
            self.deriv_covmat_comp = ids_comp
            self.deriv_covmat_fisher = ids_fish

    def _apply_covmat_split(self,ids_fish,ids_comp):
        self.covmat_comp = ids_comp
        self.covmat_fisher = ids_fish




    def compress_vector(self,vector,with_mean=True):
        if with_mean:
            assert(self._mean_comp is not None)

        if self.include_deriv_covmat:
            if self._mean_comp is None:
                raise AssertionError('For compression with parameter dependent cov mat. we require the mean')
            return self._compress_mean_and_covmat(vector)
        else:
            return self._compress_mean_only(vector,with_mean=with_mean)

    def _compress_mean_only(self,data,with_mean=False):
        data = np.atleast_2d(data)
        results = np.zeros([data.shape[0],self._n_params])
        if with_mean:
            data = data -self._mean_comp
        for i,n in enumerate(self.param_names):
            deriv = self.deriv_mean_comp[n]
            results[:,i]= self._hartlap_comp*deriv.dot(np.linalg.solve(self.covmat_comp,(data).T))
        return results

    def _compress_mean_and_covmat(self,data):
        data = np.atleast_2d(data)
        results = np.zeros([data.shape[0],self._n_params])
        data = data -self._mean_comp
        tmp = np.linalg.solve(self.covmat_comp,(data).T)*self._hartlap_comp
        for i,n in enumerate(self.param_names):
            mu_deriv = self.deriv_mean_comp[n]
            results[:,i]= mu_deriv.dot(tmp)

            results[:,i] += .5*np.einsum('ij,ij->i',tmp.T,(self.deriv_covmat_comp[n].dot(tmp)).T)

        return results

    def compute_deriv_mu_covmat(self,params_names,input_ids=False):
        n_params = len(params_names)
        if input_ids is False:
            input_ids = self._deriv_sim_ids
        for i,n1 in enumerate(params_names):
            derivs1_ens = self.get_deriv_mean_sims(n1,input_ids)
            derivs1_mn = np.mean(derivs1_ens,axis=0)
            if i==0:
                nSims = derivs1_ens.shape[0]
                
                deriv_covmat = np.zeros([n_params,n_params,self.dim,self.dim])
            for j,n2 in enumerate(params_names[:i+1]):
                derivs2_ens = self.get_deriv_mean_sims(n2,input_ids)
                derivs2_mn = np.mean(derivs2_ens,axis=0)
                deriv_covmat[i,j] = deriv_covmat[j,i]= 1/(nSims-1)*np.einsum('ij,ik->jk',(derivs1_ens-derivs1_mn),(derivs2_ens-derivs2_mn) )
        return deriv_covmat/nSims

    def _compute_fisher_matrix(self,params_names=None):
        if params_names is None: params_names = self.params_names
        if self.include_deriv_covmat:
            return self._compute_fisher_matrix_mean_and_covmat(params_names)
        else:
            return self._compute_fisher_matrix_mean_only(params_names)
            
    def _compute_fisher_matrix_mean_only(self,params_names):
        if params_names is None:
            params_names = self.params_names
        n_params = len(params_names)
        fisher = np.zeros([n_params,n_params])
        for i,n1 in enumerate(params_names):
            for j,n2 in enumerate(params_names[:i+1]):
                fisher[i,j] = fisher[j,i] = self._hartlap_fisher*np.dot(self.deriv_mean_fisher[n1], np.linalg.solve(self.covmat_fisher,self.deriv_mean_fisher[n2]))
        return fisher

    def _compute_fisher_matrix_mean_and_covmat(self,params_names):
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
        if params_names is None: params_names = self.params_names
        if self.include_deriv_covmat:
            return self._compute_fisher_matrix_mean_and_covmat_error(params_names)
        else:
            return self._compute_fisher_matrix_mean_only_error(params_names)
            
    def _compute_fisher_matrix_mean_only_error(self,params_names):
        n_params = len(params_names)


        derivs_covMat = self.compute_deriv_mu_covmat(params_names)

        fisher_err = np.zeros([n_params,n_params])
        #hartlap_fac = (nSimsCovMat-nKs-2)/(nSimsCovMat-1)
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(self.covmat_fisher,derivs_covMat[i,j]))*self._hartlap_fisher
        return fisher_err

    def _compute_fisher_matrix_mean_and_covmat_error(self,params_names):
        n_params = len(params_names)


        def estBiasCovMatVar(params_names):#(nSimsCovMat,covMat_est,derivs_covMat,derivs_covMat_a,derivs_covMat_b):
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
            #print(derivs_covMat_a,derivs_covMat_b)
            # Reset derivatives
            self.deriv_covmat_fisher = ids_0
            fisher_err = np.zeros([n_params,n_params])
            for i,n1 in enumerate(params_names):
                tmp_mat_a = np.linalg.solve(self.covmat_fisher,derivs_covMat[n1])*self._hartlap_fisher
                tmp_mat_splita_a = np.linalg.solve(self.covmat_fisher,derivs_covMat_a[n1])*self._hartlap_fisher
                tmp_mat_splitb_a = np.linalg.solve(self.covmat_fisher,derivs_covMat_b[n1])*self._hartlap_fisher
                #tmp_mat_true_a = np.linalg.solve(covMat,deriv_cm_true[i])
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


        derivs_covMat = self.compute_deriv_mu_covmat(params_names)

        fisher_err = np.zeros([n_params,n_params])
        #hartlap_fac = (nSimsCovMat-nKs-2)/(nSimsCovMat-1)
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(self.covmat_fisher,derivs_covMat[i,j]))*self._hartlap_fisher

        fisher_err+=estBiasCovMatVar(params_names)
        return fisher_err



    def _compute_compressed_fisher_matrix(self,params_names=None):
        if params_names is None: params_names = self.params_names
        if self.include_deriv_covmat:
            return self._compute_compressed_fisher_matrix_mean_and_covmat(params_names)
        else:
            return self._compute_compressed_fisher_matrix_mean_only(params_names)

    def _compute_compressed_fisher_matrix_mean_only(self,params_names):
        dim_fisher = len(params_names)
        compressed_derivs = np.zeros([dim_fisher,self._n_params])
        for i,n in enumerate(params_names):
            compressed_derivs[i]=self.compress_vector(self.deriv_mean_fisher[n],with_mean=False)

        compressed_covmat = self.compress_vector(self.covmat_fisher,with_mean=False)
        compressed_covmat = self.compress_vector(compressed_covmat.T,with_mean=False)/self._hartlap_comp # One factor cancels here as we have C^{-1}C C^{-1} 
        return self._compressed_fisher_matrix(compressed_derivs,compressed_covmat)

    def _compute_compressed_fisher_matrix_mean_and_covmat(self,params_names):
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
        if params_names is None: params_names = self.params_names
        if self.include_deriv_covmat:
            return self._compute_compressed_fisher_matrix_mean_and_covmat_error(params_names)
        else:
            return self._compute_compressed_fisher_matrix_mean_only_error(params_names)
            
    def _compute_compressed_fisher_matrix_mean_only_error(self,params_names):
        n_params = len(params_names)


        derivs_covMat = self.compute_deriv_mu_covmat(params_names)

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
                tmp = self.compress_vector(sims[finite_dif_index])
                compressed_deriv_ens+=finite_dif_weight/self._dict_param_steps[param_name]*tmp

            compressed_deriv_mean = self.compress_vector(self.deriv_mean_fisher[param_name])
            #comp_up = #compression(nSimsCovMat,comp_deriv_sims_deriv_est[:,0,i],mu,covMat_full_est,deriv_mu_compressor,deriv_covmat_compressor)
            #comp_down = #compression(nSimsCovMat,comp_deriv_sims_deriv_est[:,1,i],mu,covMat_full_est,deriv_mu_compressor,deriv_covmat_compressor)
            derivs_comp_covMat[i] =(compressed_deriv_ens-compressed_deriv_mean).T

    
        # One factor of n_fisher_sims
        mix_matrix = np.einsum('ijk,mnk->imjn',derivs_comp_covMat,derivs_comp_covMat)/(n_fisher_sims-1)/n_fisher_sims#(derivs_comp_covMat.shape[-1]-1)/comp_deriv_sims_deriv_est.shape[0]

        covMat_comp = self._compute_fisher_matrix(self.param_names)
        for i in range(n_params):
            for j in range(i+1):
                fisher_err[i,j] = fisher_err[j,i] = np.trace(np.linalg.solve(covMat_comp,mix_matrix[i,j]))
        return fisher_err