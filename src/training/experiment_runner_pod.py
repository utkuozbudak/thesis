import numpy as np
import yaml
import itertools
import json
import time

from datetime import datetime
from sklearn.pipeline import Pipeline
from src.training.dense import Dense
from src.training.linear import Linear
from src.training.pod_deeponet import DeepONetPOD

class ExperimentRunnerPOD:
    
    def __init__(self):
        self.params = self._load_params()

    def _load_params(self):
        with open('../params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params
    
    def _create_network(self, network_params, seed):
        steps = [
            ("dense", Dense(layer_width=network_params['dense_layer']['layer_width'], 
                            activation=network_params['dense_layer']['activation'],
                            parameter_sampler=network_params['dense_layer']['parameter_sampler'],
                            random_seed=seed)),
            ("linear", Linear(regularization_scale=network_params['linear_layer']['regularization_scale']))
        ]
        net = Pipeline(steps)
        return net
    
    def _run_experiments(self, X_train, X_test, y_train, y_test, grid):
        results = {}
        experiment_count = 1
        num_experiments = self.params['number_of_experiments']
        search_space = self.params['search_space_pod_deeponet']
        
        for n_modes, dense_layer_width, activation, reg_scale in itertools.product(
                    search_space['n_modes'], 
                    search_space['pod_deeponet_network']['dense_layer']['layer_width'],
                    search_space['pod_deeponet_network']['dense_layer']['activation'],
                    search_space['pod_deeponet_network']['linear_layer']['regularization_scale']):
            
                
            network_params = {
                'dense_layer': {
                    'layer_width': dense_layer_width,
                    'activation': activation,
                    'parameter_sampler': activation
                },
                'linear_layer': {'regularization_scale': float(reg_scale)}
            }
            
            # Current experiment information
            print("Current experiment:")
            print("n_modes: ", n_modes)
            print("PODDeepONet Network config: ", network_params)
            
            for i in range(num_experiments):
                seed = int(datetime.now().timestamp())

                # initialize branh and trunk networks
                network= self._create_network(network_params, seed)
                
                # create model
                model = DeepONetPOD(n_modes=n_modes, pipeline=network)
                # fit model and save the experiment time
                start_time = time.time()
                results_dict = model.fit(X_train, y_train)
                end_time = time.time()
                experiment_time = end_time - start_time

                # predictions
                predictions = model.transform(X_test)
                # compute losses
                l2_mean_relative_loss = np.mean(np.linalg.norm(predictions - y_test, axis=1) / np.linalg.norm(y_test, axis=1))
                mse_loss = np.mean((predictions-y_test)**2)
                
                # save results for JSON file
                results[f"experiment_{experiment_count}_run_{i+1}"] = {
                    'n_modes': n_modes,
                    'network_config': network_params,
                    'training_info': {
                        **results_dict,
                        'experiment_time': experiment_time,
                        'random_seed': seed},
                    'test_info': {
                        'l2_mean_relative_loss': l2_mean_relative_loss,
                        'mse_loss': mse_loss
                        }
                }
            experiment_count += 1
        # save results to JSON file
        print("Dataset: ", self.params['dataset'])
        if self.params['dataset'] == "burgers":
            with open('../outputs/pod_deeponet/results/pod_burgers_results.json', 'w') as file:
                json.dump(results, file, indent=4)
        elif self.params['dataset'] == "wave":
            with open('../outputs/pod_deeponet/results/pod_wave_results.json', 'w') as file:
                json.dump(results, file, indent=4)
        else:
            raise ValueError("Invalid dataset name in params.yaml file.") 

        return results
    
    @staticmethod
    def run(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, grid: np.ndarray):
        runner = ExperimentRunnerPOD()
        runner._run_experiments(X_train, X_test, y_train, y_test, grid)