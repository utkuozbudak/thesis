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
        self.max_iter = 5
        self.tol = 1e-6 # tolerance for convergence
        self.n_modes = 32
        self.T = None
        self.t_0 = None
        self.prev_loss = float('inf')
    
    def fit(self, V, U, epsilon):
        """
        V: (12000, 256) -> m, N
        U: (12000, 256) -> m, N
        epsilon: (1, 256) -> 1, m
        p = 32
        """
        N, m = U.shape               # m = 256, N = 12000 
        e_N = np.ones(N)             # (12000,) = (N,)
        e_N = e_N.reshape(-1, 1)     # (12000, 1) = (N, 1)
        
        # Step 1: Initializations
        self.set_pod(U)
        self.t_0 = np.mean(U, axis=0)     # (256,) = (m,)
        self.T = self.apply_pod(U)        # (12000, 32) = (N, p)
        
        for iteration in range(self.max_iter):
            # Step 2: Find the weights for the branch network
            self.branch_pipeline.fit(V, self.T)
            
            # Step 3: Compute B_tilda, which is b(V) with the new weights
            B_tilda = self.branch_pipeline.transform(V)  # (12000, 32) = (N, p)
            
            # Step 4: Set b^0 and B^0
            b_0 = np.mean(U, axis=0, keepdims=True)        # (1, 256) = (1, m)
            # Orthogonalize B_tilda
            B_0, _ = np.linalg.qr(B_tilda, mode='reduced') # (12000, 32) = (N, p)
            
            # Step 5: Find the weights for the trunk network
            reduced = (U - b_0).T
            goal_functions_trunk = reduced @ B_0 # (256, 12000) x (12000, 32) = (256, 32)
            self.trunk_pipeline.fit(epsilon, goal_functions_trunk) # epsilon = (256, 1) | goal_functions_trunk = (256, 32)
        
            # Step 6: Compute T_tilda, which is t(epsilon) with the new weights
            T_tilda = self.trunk_pipeline.transform(epsilon) # T_tilda shape = (256, 32) = (m, p)
            
            # Step 7: Keep the same t_0 and set T^1 = orthogonalize(T_tilda)
            self.T, _ = np.linalg.qr(T_tilda, mode='reduced')  # (256, 32) = (m, p)
            
            # Check for convergence using the relative L2 loss
            predictions = self.transform(V, epsilon)
            current_loss = np.sum(la.norm(U - predictions, axis=1) / la.norm(U, axis=1)) / N
            print(f"Iteration {iteration} | Relative L2 Loss: {current_loss}")
            
            if self.is_converged(current_loss):
                print(f"Converged after {iteration} iterations | Loss: {current_loss}")
                break
            
            # Update the loss
            self.prev_loss = current_loss
        
        # After the loop, execute step 2 one final time
        self.branch_pipeline.fit(V, self.T)
        
    def set_pod(self, U):
        mean = U.mean(axis=0)
        shifted = U - mean
        _, _, vh = np.linalg.svd(shifted)
        self.pod_mean = mean
        self.pod_modes = vh.T[:, :self.n_modes]
        
    def apply_pod(self, U):
        return (U - self.pod_mean) @ self.pod_modes
    
    def restore_pod(self, pod_U):
        return pod_U @ self.pod_modes.T + self.pod_mean
    
    def transform(self, X, epsilon = None):
        b_star = self.branch_pipeline.transform(X)  # (N, 32)
        predictions = b_star @ self.T.T  
        predictions += self.t_0
        return predictions

    def is_converged(self, current_loss):
        if abs(self.prev_loss - current_loss) < self.tol:
            return True
        return False
    
    def transform_branch(self, x):
        """ Used for testing POD implementation. This is not the original transform method"""
        prediction = self.branch_pipeline.transform(x)
        print(f"b prediction shape: {prediction.shape}")
        restored = self.restore_pod(prediction)
        print(f"b restored shape: {restored.shape}")
        return restored