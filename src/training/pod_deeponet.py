from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import numpy as np


@dataclass
class DeepONetPOD(BaseEstimator):
    pipeline: Pipeline
    n_modes: int = 32
    results_dict = {}
    
    def __post_init__(self):
        self.pod_modes = None 
        self.pod_mean = None

    def fit(self, x, y = None):
        self._set_pod(y)
        pod_y = self._apply_pod(y)
        self.pipeline.fit(x, pod_y)
        # training loss
        prediction = self.pipeline.predict(x)
        restored = self._restore_output(prediction)
        # l2 mean relative loss
        l2_mean_relative_loss = np.mean(np.linalg.norm(y - restored, axis=1) / np.linalg.norm(y, axis=1))
        # mse loss
        mse_loss = np.mean((y-restored)**2)
        results_dict = {
            'l2_mean_relative_loss': l2_mean_relative_loss,
            'mse_loss': mse_loss
        }
        return results_dict

    def transform(self, x, y = None):
        prediction = self.pipeline.transform(x)
        print(f"prediction shape: {prediction.shape}")
        restored = self._restore_output(prediction)
        print(f"restored shape: {restored.shape}")
        return restored
    
    def _set_pod(self, y):
        mean = y.mean(axis=0)
        shifted = y - mean
        _, _, vh = np.linalg.svd(shifted)
        self.pod_mean = mean
        self.pod_modes = vh.T[:, :self.n_modes]
        print("pod-deeponet pod modes shape", self.pod_modes.shape)

    def _apply_pod(self, y):
        print(f"y shape: {y.shape}"
              f"pod_mean shape: {self.pod_mean.shape}"
              f"pod_modes shape: {self.pod_modes.shape}")
        print(f"goal shape: {((y - self.pod_mean) @ self.pod_modes).shape}")
        return (y - self.pod_mean) @ self.pod_modes

    def _restore_output(self, pod_y):
        #print(f"pod_y shape: {pod_y.shape}"
              #f"pod_modes.T shape: {self.pod_modes.T.shape}"
              #f"pod_mean shape: {self.pod_mean.shape}")
        return pod_y @ self.pod_modes.T + self.pod_mean