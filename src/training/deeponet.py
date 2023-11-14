import numpy as np
import numpy.linalg as la

from dataclasses import dataclass
from sklearn.base import BaseEstimator


@dataclass
class DeepONet(BaseEstimator):
    def __init__(
        self,
        branch_pipeline=None,
        trunk_pipeline=None,
        n_modes: int = 32,
        max_iter: int = 5,
        tolerance: float = 1e-6,
    ):
        self.n_modes = n_modes
        self.branch_pipeline = branch_pipeline
        self.trunk_pipeline = trunk_pipeline
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.__post_init__()

    def __post_init__(self):
        self.pod_mean = None
        self.T = None
        self.t_0 = None
        self.prev_loss = float("inf")
        self.pod_modes = None
        self.loss_dict = {}
        self.no_improvement_count = 0

    def fit(self, V, U, epsilon):
        """
        Fit the DeepONet model.

        Parameters:
        V (numpy.ndarray): Input features with shape (N, m).
        U (numpy.ndarray): Output features with shape (N, m).
        epsilon (numpy.ndarray): Additional parameters with shape (m, 1).

        Returns:
        dict: A dictionary containing training results.
        """
        N, m = U.shape  # m = 256, N = 12000 for Burgers 

        # Step 1: Initializations
        self._set_pod(U)
        self.t_0 = self.pod_mean  # (256,) = (m,)
        self.T = self.pod_modes  #  = (m, p)
        
        self.results_dict = {}  # Initialize results dictionary
        self.is_converged = False  # Initialize convergence status
        
        for iteration in range(1, self.max_iter + 1):
            # Step 2: Find the weights for the branch network
            goal_function_branch = (U - self.t_0) @ self.T
            self.branch_pipeline.fit(V, goal_function_branch)

            # u1 = transform_pod(v, epsilon)
            # Step 3: Compute B_tilda
            B_tilda = self.branch_pipeline.transform(V)

            # Step 4: Set b^0 and B^0
            b_0 = np.mean(U, axis=0, keepdims=True)
            B_0, _ = np.linalg.qr(B_tilda, mode="reduced") # instead do pod of u1 (maybe transpose): BB^T = I^p should hold

            # Step 5: Find the weights for the trunk network
            goal_function_trunk = (U - b_0).T @ B_0
            self.trunk_pipeline.fit(epsilon, goal_function_trunk)

            # u2 = 
            # Step 6: Compute T_tilda
            T_tilda = self.trunk_pipeline.transform(epsilon)

            # Step 7: Keep the same t_0 and set T^1
            self.T, _ = np.linalg.qr(T_tilda, mode="reduced") # 
            

            # Check for convergence
            predictions = self.transform(V, epsilon)
            current_loss = np.sum(la.norm(U - predictions, axis=1) / la.norm(U, axis=1)) / N
            
            print(f"Iteration {iteration} | Relative L2 Loss: {current_loss}")
            self.results_dict[f'iteration_{iteration}_loss'] = current_loss

            if self._is_converged(current_loss):
                print(f"Converged after {iteration} iterations | Loss: {current_loss}")
                self.is_converged = True
                break

            self.prev_loss = current_loss

        # After the loop, execute step 2 one final time
        goal_function_branch = (U - self.t_0) @ self.T
        self.branch_pipeline.fit(V, goal_function_branch)

        converged_iteration = iteration if self.is_converged else -1
        self.results_dict['is_converged'] = self.is_converged
        self.results_dict['iterations_to_convergence'] = converged_iteration
        
        return self.results_dict

    def _set_pod(self, U):
        mean = U.mean(axis=0)
        shifted = U - mean
        _, _, vh = np.linalg.svd(shifted)
        self.pod_mean = mean
        self.pod_modes = vh.T[:, : self.n_modes]

    def transform(self, X, epsilon=None):
        branch_output = self.branch_pipeline.transform(X)  # (N, 32)
        predictions = branch_output @ self.T.T  # (N, 32) x (32, 256) = (N, 256)
        predictions += self.t_0
        return predictions

    def _is_converged(self, current_loss):
        improvement = abs(self.prev_loss - current_loss)
        if improvement < self.tolerance:
            self.no_improvement_count += 1
            print(f"No improvement for the last {self.no_improvement_count} iterations")
        else:
            self.no_improvement_count = 0

        if self.no_improvement_count >= 3:
            return True
        return False

    """
    def apply_pod(self, U):
        return (U - self.pod_mean) @ self.pod_modes
    
    def restore_pod(self, pod_U):
        return pod_U @ self.pod_modes.T + self.pod_mean
        
    def transform_branch(self, x):
        Used for testing POD implementation. This is not the original transform method
        prediction = self.branch_pipeline.transform(x)
        print(f"b prediction shape: {prediction.shape}")
        restored = self.restore_pod(prediction)
        print(f"b restored shape: {restored.shape}")
        return restored
    """
