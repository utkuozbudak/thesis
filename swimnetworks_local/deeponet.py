from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import numpy as np
import numpy.linalg as la


@dataclass
class DeepONet(BaseEstimator):
    branch_pipeline: Pipeline
    trunk_pipeline: Pipeline
    
    def __post_init__(self):
        self.pod_mean = None
        self.max_iter = 1
        self.tol = 1e-6 # tolerance for convergence
        self.n_modes = 32
        self.T = None
        self.t_0 = None
        self.prev_loss = float('inf')
    
    def fit(self, V, U, epsilon):
        """
        V: (256, 12000) -> m, N
        U: (256, 12000) -> m, N
        epsilon: (1, 256) -> 1, m
        p = 32
        """
        m, N = U.shape               # m = 256, N = 12000 
        e_N = np.ones(N)             # (12000,) = (N,)
        e_N = e_N.reshape(-1, 1)     # (12000, 1) = (N, 1)
        
        # Step 1: Initializations
        self.t_0 = np.mean(U, axis=1)     # (256,) = (m,)
        self.T = self._pod(U).T           # (32, 256) = (p, m)
        
        for iteration in range(self.max_iter):
            # Step 2: Find the weights for the branch network
            branch_goal_function = self.T @ (U - np.outer(self.t_0, e_N))   # (32, 12000) = (p, N)
            self.branch_pipeline.fit(V.T, branch_goal_function.T) # V.T = (12000, 256) | branch_goal_function.T = (12000, 32)
            
            # Step 3: Compute B_tilda, which is b(V) with the new weights
            B_tilda = self.branch_pipeline.transform(V.T).T       # (32, 12000) = (p, N)
            
            # Step 4: Set b^0 and B^0
            b_0 = np.mean(U, axis=1, keepdims=True)  # (256, 1) = (m, 1)
            # Orthogonalize B_tilda
            B_0 = self.orthogonalize(B_tilda)        # (32, 12000) = (p, N)
            
            # Step 5: Find the weights for the trunk network
            goal_functions_trunk = B_0 @ ((U - np.outer(b_0, e_N)).T) # (32, 256) = (p, m)
            self.trunk_pipeline.fit(epsilon.T, goal_functions_trunk.T) # epsilon.T = (256, 1) | goal_functions_trunk.T = (256, 32)
        
            # Step 6: Compute T_tilda, which is t(epsilon) with the new weights
            T_tilda = self.trunk_pipeline.transform(epsilon.T).T # T_tilda shape = (32, 256) = (p, m)
            
            # Step 7: Keep the same t_0 and set T^1 = orthogonalize(T_tilda)
            self.T = self.orthogonalize(T_tilda)
            
            # Check for convergence
            predictions = self.transform(V, epsilon)
            current_loss = np.mean((U - predictions)**2)
            
            print(f"Iteration {iteration} | Loss: {current_loss}")
            if self.is_converged(current_loss):
                print(f"Converged after {iteration} iterations | Loss: {current_loss}")
                break
            
            # Update the loss
            self.prev_loss = current_loss
        
        # After the loop, execute step 2 one final time
        branch_goal_function = self.T @ (U - np.outer(self.t_0, e_N))   # (32, 12000) = (p, N)
        self.branch_pipeline.fit(V.T, branch_goal_function.T) # V.T = (12000, 256) | branch_goal_function.T = (12000, 32)

    def is_converged(self, current_loss):
        if abs(self.prev_loss - current_loss) < self.tol:
            return True
        return False
    
    def _pod(self, U):
        self.pod_mean = np.mean(U, axis=1, keepdims=True)
        u_svd, _, _ = np.linalg.svd(U - self.pod_mean)
        self.pod_modes = u_svd[:, :self.n_modes]
        return self.pod_modes
    
    def _restore_output(self, pod_U):
        return pod_U @ self.pod_modes.T + self.pod_mean

    def transform(self, V_star, epsilon):
        # Compute b(v^*) using the branch network
        b_star = self.branch_pipeline.transform(V_star.T).T  # (p, N_star)

        # Compute the prediction 
        predictions = np.zeros((epsilon.shape[1], V_star.shape[1]))  # Initialize the predictions matrix (m x N_star)

        for j in range(self.n_modes):
            for k in range(V_star.shape[1]):  # for each test sample
                predictions[:, k] += self.T[j, :] * b_star[j, k]

        predictions += np.outer(self.t_0, np.ones(V_star.shape[1]))
        return predictions

    def orthogonalize(self, U, eps=1e-15):
        """ Gram Schmidt orthogonalization of the columns of U."""
        n = len(U[0])
        # numpy can readily reference rows using indices, but referencing full rows is a little
        # dirty. So, work with transpose(U)
        V = U.T
        for i in range(n):
            prev_basis = V[0:i]     # orthonormal basis before V[i]
            coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
            # subtract projections of V[i] onto already determined basis V[0:i]
            V[i] -= np.dot(coeff_vec, prev_basis).T
            if la.norm(V[i]) < eps:
                V[i][V[i] < eps] = 0.   # set the small entries to 0
            else:
                V[i] /= la.norm(V[i])
        return V.T