import numpy as np
import multiprocessing
from multiprocessing import Pool
import scipy.signal

class SENMF(object):
    def __init__(self, n_bases, window_width, X):
        self.n_bases = n_bases
        self.window_width = window_width
        self.n_timesteps, self.n_features = X.shape
        self.X = X
        self.A = None
        self.D = None
        self.R = None
        self.variances = None
        self.recon_error = None

    def rand_A(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.A = np.random.random((self.n_bases, self.n_timesteps))+2
        return self.A

    def rand_D(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.D = np.random.random((self.n_bases, self.window_width, self.n_features))+2
        return self.D

    def normalize_D(self):
        for i in range(self.n_bases):
            self.D[i] /= np.linalg.norm(self.D[i])

    def reconstruct(self):
        "Reconstruct an estimation of the training data"
        X_bar = np.zeros((self.n_timesteps, self.n_features))
        for basis, activation in zip(self.D, self.A):
            X_bar += scipy.signal.fftconvolve(basis.T, np.atleast_2d(activation)).T[:self.n_timesteps]
        return X_bar

    def reconstruct_component(self, component):
        return scipy.signal.fftconvolve(component[0].T, np.atleast_2d(component[1])).T[:self.n_timesteps]

    def reconstruct_parallel(self, p):
        "Reconstruct an estimation of the training data"
        return sum(p.map(self.reconstruct_component, zip(self.D, self.A)))

    def reconstruct_basis(self, basis):
        return scipy.signal.fftconvolve(self.D[basis].T, np.atleast_2d(self.A[basis]))

    def residual(self, p):
        "calculate the multiplicative residual error"
        return self.X / np.abs(self.reconstruct_parallel(p))
        #return self.X / np.abs(self.reconstruct())

    def update_residual(self, p):
        "calc and store residual for future use"
        self.R = self.residual(p)

    def update_A(self):
        "Using stored residual, calculate and apply an update to activations"
        for t_prime in range(self.window_width):
            U_A = np.einsum(
                    "jk,tk->jt",
                    self.D[:,t_prime,:]/np.atleast_2d(self.D[:,t_prime,:].sum(axis=1)).T,
                    self.R[t_prime:])
            self.A[:,:-t_prime or None] *= U_A

    def update_A_fast(self, p):
        "Using stored residual, calculate and apply an update to activations"
        for t_prime in range(self.window_width):
            self.update_residual(p)
            U_A = np.dot(
                self.D[:,t_prime,:]/np.atleast_2d(self.D[:,t_prime,:].sum(axis=1)).T,
                self.R[t_prime:].T
            )
            self.A[:,:-t_prime or None] *= U_A

    def D_delta(self):
        D_updates = np.zeros((self.n_bases, self.window_width, self.n_features))
        for t_prime in range(self.window_width):
            U_D = np.dot(
                self.A[:,:-t_prime or None]/np.atleast_2d(self.A[:,:-t_prime or None].sum(axis=1)).T,
                self.R[t_prime:]
            )
            D_updates[:,t_prime,:] = U_D
        return D_updates

    def update_D(self):
        "Using stored residual, calculate and apply an update to dictionary"
        self.D *= self.D_delta()

    def get_vars(self):
        p = Pool(self.n_bases)
        components = p.map(self.reconstruct_component, zip(self.D, self.A))
        reconstruction_variance = np.linalg.norm(sum(components))
        p.close()
        p.join()
        return p.map(lambda component: np.linalg.norm(reconstruct_component(component))/reconstruction_variance, components)

    def reconstruction_error(self, X = None):
        if X == None:
            X = self.X
        self.recon_error = np.linalg.norm(X*np.log(X/self.R)-X+self.R)
        return self.recon_error

    def fit(self, n_iter):
        p = Pool(self.n_bases)
        recon_errors = np.zeros((n_iter,))
        print("Norm of data matrix is: %d" %(np.linalg.norm(self.X)))
        for i in range(n_iter):
            self.update_A_fast(p)
            self.update_residual(p)
            self.update_D()
            print("Norm of dictionary matrix is: %d" %(np.linalg.norm(self.D)))
            print("Norm of activation matrix is: %d" %(np.linalg.norm(self.A)))
            print("Norm of reconstruction matrix is: %d" %(np.linalg.norm(self.R)))
            recon_errors[i] = self.reconstruction_error()
            print("Reconstruction error is: %d" %(self.recon_error))
        p.close()
        p.join()

    def transform(self, X_new, n_iter):
        p = Pool(self.n_bases)
        for _ in range(n_iter):
            self.update_A_fast(p)