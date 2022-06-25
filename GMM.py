# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:13:41 2022

@author: ryanb
"""

import numpy as np
from Kmeans import k_means
from sklearn.cluster import KMeans as k_means_sklearn
from scipy import linalg

class gaussian_mixture_models(object):
    def __init__(self, n_components, covariance_type = 'full', tol = 0.001, reg_covar = 1e-3, max_iter = 100, n_init = 1, init_params = 'kmeans', use_sklearn = False, random_state = None, warm_start = False, verbose = False, verbose_interval = 10, vectorise = True, precision_flag = False):
        
        self.n_components = n_components
        self.covariance_type = covariance_type #full, tied, diag, spherical
        self.tol = tol #convergence threshold, stops when lower bound avg gain is below threshold
        self.reg_covar = reg_covar #Non-negative regularisation added to the diagonal of covariance. Ensures cov is positive
        self.max_iter = max_iter #Number of EM iterations
        self.n_init = n_init #Number of initialisations to perform
        self.init_params = init_params #'k-means' or 'random' - specifies how to initialise means and covariances
        self.use_sklearn = use_sklearn
        self.random_state = random_state #controls the random seed to initialise the parameters of the model
        self.warm_start = warm_start #if True, solution of last fitting is used to initialise the next run
        self.verbose = verbose #Controls what you print
        self.verbose_interval = verbose_interval #Intervals between prints
        self.vectorise = vectorise #Controls if you want to use loops or matrices to calculate x^T A x operations
        self.precision_flag = precision_flag
        
        #Attributes
        #self.weights_ = None #Mixture weights #(n_components,)
        #self.means_ = None #Mixture means (n_components, n_features)
        #self.covariances = None #Mixture covariances
        # spherical: (n_components, )
        # diagonal: (n_components, n_features)
        # tied: (n_features, n_features)
        # full: (n_components, n_features, n_features)
        #self.converged_ = False #Flag to tell if the model converged
        #self.n_iter_ = None #Number of steps to finalise run
        #self.lower_bound_ = None #lower bound value of the log-likelihood (training data wrt model for best fit model)
        #self.n_features_in_ = None #number of features seen during fit
    
    def set_params(self, X):
        
        Ntotal, self.n_features_in_ = X.shape
        
        if not hasattr(self, "mix_init_") or not hasattr(self, "mu_init_") or not hasattr(self, "cov_init_"):
        
            #Initialise weight (mixing coeffient) vector
            mix_init = np.zeros((self.n_components,))

            #Initialise mean matrix
            mu_init = np.zeros((self.n_components, self.n_features_in_))

            #Initialise covariance matrix
            if self.covariance_type == 'spherical':
                cov_init = np.zeros((self.n_components,))

            elif self.covariance_type == 'diagonal':
                cov_init = np.zeros((self.n_components, self.n_features_in_))

            elif self.covariance_type == 'tied':
                cov_init = np.zeros((self.n_features_in_, self.n_features_in_))

            elif self.covariance_type == 'full':
                cov_init = np.zeros((self.n_components, self.n_features_in_, self.n_features_in_))
            
            else:
                print("Incompatible covariance_type entered.")
                raise SystemExit
                
            if self.init_params == 'kmeans':

                if self.verbose:
                    print("\nRunning k-means to initialise model parameters...")

                if self.use_sklearn:
                    self.k_means_local = k_means_sklearn(self.n_components, 
                                                         random_state = self.random_state, 
                                                         verbose = self.verbose)

                    label = self.k_means_local.fit_predict(X)

                else:
                    self.k_means_local = k_means(self.n_components, 
                                                 'random', 
                                                 random_state = self.random_state, 
                                                 verbose = self.verbose,
                                                 vectorise = True)

                    label = self.k_means_local.fit_predict(X, calculate_scores = False)

                if self.verbose:
                    print("\nFinished k-means")

                for i in range(self.n_components):
                    label_indices = np.nonzero(label == i)[0]

                    x_cluster = X[label_indices, :]
                    N = len(label_indices)

                    #Update mixing matrix
                    mix_init[i] = len(label_indices) / Ntotal

                    #Save mean
                    mu_init[i, :] = np.mean(x_cluster, axis = 0)

                    #Calculate standard covariance
                    z = x_cluster - mu_init[i, :]
                    covariance_cluster = 1/(N) * np.dot(z.T, z)

                    #Add in regularisation term
                    covariance_cluster += np.eye(self.n_features_in_) * self.reg_covar

                    if self.covariance_type == 'spherical':
                        cov_init[i] = np.mean(np.diag(covariance_cluster))

                    elif self.covariance_type == 'diagonal':
                        cov_init[i, :] = np.diag(covariance_cluster)

                    elif self.covariance_type == 'tied':
                        cov_init += covariance_cluster

                    elif self.covariance_type == 'full':
                        cov_init[i, :, :] = covariance_cluster

                if self.covariance_type == 'tied':
                    cov_init /= self.n_components #Average covariance matrix


            elif self.init_params == 'random':
                print("'random' init_params initialisation not implemented.")
                raise SystemExit

            else:
                print("Incompatible init_params entered.")
                raise SystemExit

            #original initialisations
            self.mix_init_ = mix_init
            self.mu_init_ = mu_init
            self.cov_init_ = cov_init
        
        #Save matrices
        self.weights_ = self.mix_init_
        self.means_ = self.mu_init_
        self.covariances_ = self.cov_init_
        
        #Update precision
        if self.precision_flag:
            self.precisions_ = self.invert_covariance()
        
    def get_params(self):
        
        #returns covariances as a k x D x D matrix, regardless of type
        
        covariance = np.zeros((self.n_components, self.n_features_in_, self.n_features_in_))
        
        if self.precision_flag:
            precision = np.zeros((self.n_components, self.n_features_in_, self.n_features_in_))
        
        else:
            precision = None
        
        for k in range(self.n_components):
            if self.covariance_type == 'spherical':
                    covariance[k, :, :] = np.diag(self.covariances_[k] * np.ones(self.n_features_in_))
                    
                    if self.precision_flag:
                        precision[k, :, :] = np.diag(self.precisions_[k] * np.ones(self.n_features_in_))
                    
            elif self.covariance_type == 'diagonal':
                covariance[k, :, :] = np.diag(self.covariances_[k, :])
                
                if self.precision_flag:
                   precision[k, :, :] = np.diag(self.precisions_[k, :])

            elif self.covariance_type == 'tied':
                covariance[k, :, :] = self.covariances_
                
                if self.precision_flag:
                    precision[k, :, :] = self.precisions_

            elif self.covariance_type == 'full':
                covariance[k, :, :] = self.covariances_[k, :, :]
                
                if self.precision_flag:
                    precision[k, :, :] = self.precisions_[k, :, :]
                    
        return self.weights_, self.means_, covariance, precision
    
    def invert_covariance(self):
        
        precision = np.zeros_like(self.covariances_)
        
        for k in range(self.n_components):
            if self.covariance_type == 'spherical':
                precision[k] = 1/self.covariances_[k]
            
            elif self.covariance_type == 'diagonal':
                precision[k, :] = 1/self.covariances_[k, :]
            
            elif self.covariance_type == 'tied':
                precision = linalg.pinvh(self.covariances_)
            
            elif self.covariance_type == 'full':
                precision[k, :, :] = linalg.pinvh(self.covariances_[k, :, :])
        
        return precision
                
    def evaluate_gauss(self, x, mu, covariance, precision = None):
        #All vector/matrix inputs (so that shape is fine)
        #You could also store the precisions, may make this evaluation faster
        
        if self.vectorise:
            
            D = x.shape[1]
            z = x - mu.reshape(1, -1) #reshape mu
            
            if precision is None:
                precision = np.linalg.inv(covariance) #Sorry prof. Kok
            
            second_part = np.exp((-1 / 2) * np.sum(np.dot(z, precision) * z, axis = 1))
        
        else: #vector inputs for mu and x_vec
            D = x.shape[0]
            z = x - mu
        
            second_part = np.exp((-1 / 2) * np.dot(z.T, np.linalg.solve(covariance, z)))
        
        
        
        first_part = 1 / ((2 * np.pi)**(D / 2))  * 1 / (np.linalg.det(covariance)**(1/2))
        
        return first_part * second_part
    
    def evaluate_log_gauss(self, x, mu, covariance, precision = None):
        #All vector/matrix inputs (so that shape is fine)
        #You could also store the precisions, may make this evaluation faster
        
        if self.vectorise:
            D = x.shape[1]
            z = x - mu.reshape(1, -1)
            
            if precision is None:
                precision = np.linalg.inv(covariance) #Sorry prof. Kok
            
            third_part = -1/2 * np.sum(np.dot(z, precision) * z, axis = 1)
        
        else:     
            D = x.shape[0]
            z = x - mu
            third_part = -1/2 * np.dot(z.T, np.linalg.solve(covariance, z))

        first_part = -D/2 * np.log(2 * np.pi)
        second_part = -1/2 * np.log(np.linalg.det(covariance) + 1e-12)
        
        return first_part + second_part + third_part
    
    def data_likelihood(self, x): #p(x)
        
        #use log-sum-exp to ensure that you do not hit numerical underflow!
        mix, mu, cov, prec = self.get_params()
        
        if x.shape[1] == 1:
            N = 1
        else:
            N = x.shape[0]
        
        log_likelihoods = np.zeros((N, self.n_components))
        
        for k in range(self.n_components):
            
            mean_vec = mu[[k], :].T #N x 1 vector
            cov_mat = cov[k, :, :] #N x N matrix
            
            if self.precision_flag:
                prec_mat = prec[k, :, :]
            
            else:
                prec_mat = None
            
            if N == 1:
                log_likelihoods[0, k] = self.evaluate_log_gauss(x, mean_vec, cov_mat, prec_mat) + np.log(mix[k])
                
            elif N != 1 and self.vectorise:
                log_likelihoods[:, k] = self.evaluate_log_gauss(x, mean_vec, cov_mat, prec_mat) + np.log(mix[k])
            
            if N != 1 and not self.vectorise:
                
                for i in range(N):
                    log_likelihoods[i, k] = self.evaluate_log_gauss(x[[i], :].T, mean_vec, cov_mat, prec_mat) + np.log(mix[k])

        #Apply log-sum-exp trick
        A = np.max(log_likelihoods, axis = 1, keepdims=True)
    
        log_likelihood_sum = A + np.log(np.sum(np.exp(log_likelihoods - A), axis = 1, keepdims = True))
        
        p_x = np.exp(log_likelihood_sum)
        
        return p_x, log_likelihoods, log_likelihood_sum
    
    def posterior_likelihood(self, x): #Calculates the responsibilities 
        
        _, log_likelihoods, log_likelihood_sum = self.data_likelihood(x)
        
        responsibility = np.exp(log_likelihoods - log_likelihood_sum)
        
        return responsibility
    
    def log_likelihood(self, X, sum_flag = True): #return the log-likelihood sum
        
        N = X.shape[0]
        LL = np.zeros((N, 1))
        
        if self.vectorise:
            LL = self.data_likelihood(X)[2]
        
        else:
            for i in range(N):

                x_vec = X[[i], :].T #N x 1 vector

                #Add log-likelihood
                LL[i, 0] = self.data_likelihood(x_vec)[2]
        
        if sum_flag:
            return np.mean(LL)
        
        else:
            return LL
    
    def Nk(self, X):
        N = X.shape[0]
        
        resp_mat = np.zeros((N, self.n_components))
        
        if self.vectorise:
            resp_mat[:, :] = self.posterior_likelihood(X)
        
        else:
            for i in range(N):
                resp_mat[i, :] = self.posterior_likelihood(X[[i], :].T)
        
        N_vec = np.sum(resp_mat, axis = 0)
        
        return N_vec, resp_mat
    
    def update_mu(self, X, resp_mat, N_vec):
        
        mu_new = np.zeros_like(self.means_)
        
        for i in range(self.n_components):
            mu_new[i, :] = np.sum(resp_mat[:, [i]] * X, axis = 0) / N_vec[i]
        
        self.means_ = mu_new
    
    def update_covar(self, X, resp_mat, N_vec):
        
        N, D = X.shape
        cov_mat = np.zeros_like(self.covariances_)
        
        if self.covariance_type == 'spherical':
            
            for k in range(self.n_components):
                
                placeholder_cov = np.zeros((1, 1))
                
                if self.vectorise:
                    z = X - self.means_[[k], :]
                    
                    placeholder_cov[0, 0] = np.sum(resp_mat[:, [k]] * z**2)
                
                else:
                    for n in range(N):
                        z = np.transpose(X[[n], :] - self.means_[[k], :])

                        placeholder_cov += resp_mat[n, k] * np.dot(z.T, z)
                
                #Normalise by D * Nk
                placeholder_cov /= (D * N_vec[k])
                
                #Adjust diagonal covariance terms
                placeholder_cov = np.clip(placeholder_cov, self.reg_covar, None)
                
                #save
                cov_mat[k] = placeholder_cov[0, 0]

        elif self.covariance_type == 'diagonal':
            for k in range(self.n_components):
                
                placeholder_cov = np.zeros((self.n_features_in_, 1))
                
                if self.vectorise:
                    z = X - self.means_[[k], :]

                    placeholder_cov[:, 0] = np.sum(resp_mat[:, [k]] * (z**2), axis = 0)
                        
                else:
                    for n in range(N):
                        z = np.transpose(X[[n], :] - self.means_[[k], :])

                        placeholder_cov += resp_mat[n, k] * (z**2)
                
                #Normalise by Nk
                placeholder_cov /= N_vec[k]
                
                #Adjust diagonal covariance terms
                placeholder_cov = np.clip(placeholder_cov, self.reg_covar, None)
                
                #save
                cov_mat[k, :] = placeholder_cov[:, 0]

        elif self.covariance_type == 'tied':
            
            placeholder_cov = np.zeros_like(cov_mat)
            
            for k in range(self.n_components):

                if self.vectorise:
                    z = X - self.means_[[k], :]

                    einsum = resp_mat[:, k].reshape(-1, 1, 1) * np.einsum('ij, ik ->ijk', z, z)
                    placeholder_cov += np.sum(einsum, axis = 0)

                else:

                    for n in range(N):
                        z = np.transpose(X[[n], :] - self.means_[[k], :])

                        placeholder_cov += np.dot(z, z.T)
                
            #Normalise by N
            placeholder_cov /= (N)
            
            #Adjust diagonal covariance terms
            diag_indices = np.diag_indices(self.n_features_in_)
            placeholder_cov[diag_indices] = np.clip(placeholder_cov[diag_indices], self.reg_covar, None)
            
            #save
            cov_mat = placeholder_cov

        elif self.covariance_type == 'full':
            
            for k in range(self.n_components):
                
                placeholder_cov = np.zeros_like(cov_mat[k, :, :])
                
                if self.vectorise:
                    z = X - self.means_[[k], :]

                    einsum = resp_mat[:, k].reshape(-1, 1, 1) * np.einsum('ij, ik ->ijk', z, z)
                    placeholder_cov += np.sum(einsum, axis = 0)

                else:

                    for n in range(N):
                        z = np.transpose(X[[n], :] - self.means_[[k], :])

                        placeholder_cov += resp_mat[n, k] * np.dot(z, z.T)
                
                #Normalise by Nk
                placeholder_cov /= N_vec[k]
                
                #Adjust diagonal covariance terms
                diag_indices = np.diag_indices(self.n_features_in_)
                placeholder_cov[diag_indices] = np.clip(placeholder_cov[diag_indices], self.reg_covar, None)
            
                #save
                cov_mat[k, :, :] = placeholder_cov
        
        #Update covariance variable
        self.covariances_ = cov_mat
        
        if self.precision_flag:
            #update precisions
            self.precisions_ = self.invert_covariance()
    
    def update_mixture(self, X, N_vec):
        N = X.shape[0]
        
        mix_new = N_vec / N
        
        self.weights_ = mix_new
    
    def fit(self, X, calculate_scores = True):
        
        if self.verbose:
            print("\nBeginning GMM training...")
            
        iter_dict = {}
        iter_lower_bound = np.zeros(self.n_init)
        
        for outer_iter in range(self.n_init):
            
            self.set_params(X)

            if self.verbose:
                print("Running GMM iteration {}...".format(outer_iter))

            tol = np.inf
            cnt = 0
            loss_train = [self.log_likelihood(X) ]

            while cnt < self.max_iter and tol > self.tol:
                #Determine responsibilities for each class
                N_vec, resp_mat = self.Nk(X)

                #Update means
                self.update_mu(X, resp_mat, N_vec)

                #Update covariance
                self.update_covar(X, resp_mat, N_vec)

                #Update mixture coefficients
                self.update_mixture(X, N_vec)

                #Calculate log-likelihood
                loss = self.log_likelihood(X, sum_flag = True)

                tol = np.abs(loss_train[-1] - loss)

                if self.verbose:
                    print("\nLoss at epoch {}: {}\nTolerance: {}".format(cnt, loss, tol))

                #Update other variables
                cnt += 1
                loss_train.append(loss)
            
            #Check convergence
            if cnt <= self.max_iter or tol <= self.tol:
                converged = True

            else:
                converged = False
            
            #Store attributes
            n_iter_ = cnt
            lower_bound_ = loss_train[-1]
            loss_train_ = np.array(loss_train)
            labels_ = self.predict(X)
            
            #Save in iteration dictionary
            iter_dict[str(outer_iter)] = {"weights":self.weights_,
                                          "means":self.means_,
                                          "covariances":self.covariances_,
                                          "labels":labels_,
                                          "loss_list":loss_train_,
                                          "lower_bound":lower_bound_,
                                          "n_iter":n_iter_,
                                          "converged":converged}
            
            #Save lower bound
            iter_lower_bound[outer_iter] = lower_bound_
        
        #Select optimal model
        opt_index = str(np.argmax(iter_lower_bound))
        
        self.weights_ = iter_dict[opt_index]["weights"]
        self.means_ = iter_dict[opt_index]["means"]
        self.covariances_ = iter_dict[opt_index]["covariances"]
        self.labels_ = iter_dict[opt_index]["labels"]
        self.loss_train_ = iter_dict[opt_index]["loss_list"]
        self.lower_bound_ = iter_dict[opt_index]["lower_bound"]
        self.n_iter_ = iter_dict[opt_index]["n_iter"]
        self.converged = iter_dict[opt_index]["converged"]
        
        if self.verbose:
            print("\nFinished GMM training!\n")
            print("Converged: {}".format(self.converged))
        
        if calculate_scores:
            if self.verbose:
                print("Calculating Silhouette score...")

            self.silhouette_score = self.Silhouette(X, self.labels_)


            if self.verbose:
                print("Calculating AIC...")

            self.aic_score = self.aic(X)

            if self.verbose:
                print("Calculating BIC...")

            self.bic_score = self.bic(X)
        
        return self
    
    def predict(self, X):
        
        resp_mat = self.predict_proba(X)
        
        labels = np.argmax(resp_mat, axis = 1)
        
        return labels
    
    def fit_predict(self, X):
        
        self.fit(X)
        labels = self.predict(X)
        
        return labels
    
    def estimate_centroids(self):
        pass #No clue what this does xD
    
    @staticmethod
    def sample_gaussian(n_points, mean, covariance):
        #Find any real matrix A such that A.A^T = Cov. If Cov is positive definite, Cholesky decomposition can be used.
        #Alternatively, use spectral decomposition Cov = U.S.U^(-1) (A = U.sqrt(S)) (U is the eigenvector matrix)
        #Let z = (z_1, ..., z_n)^T be a vector whose components are N independent standard normal variables
        #let x = mu + Az.
        
        #Positive definite - x^T A x > 0 for all x (all eigenvalues are positive)
        #Positive semi-definite - x^T A x >= 0 for all x (all eigenvalues are non-negative)
        
        n_features = covariance.shape[0]
        
        eig_vals, eig_vect = np.linalg.eig(covariance)
        
        if np.sum(eig_vals > 0) == n_features:
            #print("\nCovariance is positive definite.")
            
            A = np.linalg.cholesky(covariance)
            
        elif np.sum(eig_vals >= 0) == n_features:
            #print("\nCovariance is positive semi-definite.")
            
            A = np.dot(eig_vect, np.diag(np.sqrt(eig_vals)))
        
        z = np.random.randn(n_points, n_features)
        x_samples = mean + np.dot(A, z.T).T
        
        return x_samples
    
    def sample(self, n_samples = 1):
        
        weights, means, covariances = self.get_params()
        
        cumsum = np.cumsum(weights)
        
        n_classes = np.zeros(self.n_components, dtype = int)
        
        #Select centers
        rand_sample = np.random.rand(n_samples)
        
        #Collect per class
        for i in rand_sample:
            pos_select = np.argmin(np.abs(cumsum - i)) - 1
            n_classes[pos_select] += 1
        
        print(n_classes)
        
        #Sample per class
        Xlist = []
        
        for k, n_points in enumerate(n_classes):
            
            Xlist.append(self.sample_gaussian(n_points, means[k, :], covariances[k, :, :]))
        
        Xsample = np.concatenate(Xlist, axis=0)
        
        #Generate labels
        labels = []
        
        for cnt, i in enumerate(n_classes):
            labels += [cnt] * i
        
        return Xsample, labels
        
    
    def predict_proba(self, X):
        
        _, responsibilities = self.Nk(X)
        
        return responsibilities
    
    def score(self, X):
        return self.log_likelihood(X, sum_flag = True)
    
    def score_samples(self, X):
        return self.log_likelihood(X, sum_flag = False)
    
    def n_parameters(self):
        return np.sum(self.means_.shape) + np.sum(self.covariances_.shape) + np.sum(self.weights_.shape)
    
    def aic(self, X, unbiased = True, adjust = True):
        
        N = X.shape[0]
        n_parameters = self.n_parameters()
        
        term1 = self.score(X)
        
        if unbiased:
            term2 = n_parameters + (n_parameters * (n_parameters + 1))/(N - n_parameters - 1)
        
        else:
            term2 = n_parameters
        
        aic_value = term1 - term2
        
        if adjust:
            aic_value *= -2
            
        return aic_value
    
    def bic(self, X, adjust = True):
        
        N = X.shape[0]
        
        bic_value = self.score(X) - 1/2 * self.n_parameters() * np.log(N)
        
        if adjust:
            bic_value *= -2
        
        return bic_value
    
    def KL_divergence(self):
        pass
        #Only useful if you decide to split the dataset in half and test divergence between them
    
    def JS_divergence(self):
        pass
        #Only useful if you decide to split the dataset in half and test divergence between them
    
    def Silhouette(self, X, labels):
        
        N, f = X.shape
        
        #Create label nested lists
        classes = []

        for i in range(np.max(labels) + 1): #labels are zero padded
            indices = np.argwhere(labels == i)
            
            if len(indices) != 0:
                classes.append(indices[:, 0])       
            
            else:
                classes.append([])
        
        #Create distance matrix
        upper_indices_row, upper_indices_col = np.triu_indices(N) #Utilise symmetry
        
        d_vals = []
        d_mat = np.zeros((N, N))
        
        for i,j in zip(upper_indices_row, upper_indices_col):
            
            if i == j:
                d_vals.append(0) #Distance is zero
            
            else:
                d_vals.append(np.sum((X[i, :] - X[j, :])**2) ** 0.5)
                
        d_mat[upper_indices_row, upper_indices_col] = d_vals
        
        d_mat = d_mat + d_mat.T - np.diag(np.diag(d_mat))
        
        S = np.zeros(N)
        
        for i in range(N):
            
            label_i = labels[i]
            C_I = len(classes[label_i])
            
            d_sum = []
            
            if C_I == 1 or C_I == 0:
                S[i] = 0
            
            else:
                for j in range(np.max(labels) + 1):

                    if j == label_i:
                        Ai = np.sum(d_mat[i, classes[j]]) / (C_I - 1)

                    else:
                        d_sum.append( np.mean(d_mat[i, classes[j]]) )
                
                if len(d_sum) == 0:
                    Bi = 0
                    
                else:
                    #Calculate B
                    Bi = np.min(d_sum)

                S[i] = (Bi - Ai) / max(Ai, Bi)

        return np.mean(S)