import numpy as np
import yaml
import itertools
import json
import time

from datetime import datetime
from sklearn.pipeline import Pipeline
from src.training.deeponet import DeepONet
from src.training.dense import Dense
from src.training.linear import Linear

class ExperimentRunner:
    
    def __init__(self):
        self.params = self._load_params()

    def _load_params(self):
        with open('../params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params
            
    def _create_trunk_and_branch_network(self, branch_params, trunk_params, seed):
        
        # Branch net
        branch_steps = [
            ("dense", Dense(layer_width=branch_params['dense_layer']['layer_width'], 
                            activation=branch_params['dense_layer']['activation'],
                            parameter_sampler=branch_params['dense_layer']['parameter_sampler'],
                            random_seed=seed)),
            ("linear", Linear(regularization_scale=branch_params['linear_layer']['regularization_scale']))
        ]
        branch_net = Pipeline(branch_steps)
        
        # Trunk net
        trunk_steps = [
            ("dense", Dense(layer_width=trunk_params['dense_layer']['layer_width'], 
                            activation=trunk_params['dense_layer']['activation'],
                            parameter_sampler=trunk_params['dense_layer']['parameter_sampler'],
                            random_seed=seed)),
            ("linear", Linear(regularization_scale=trunk_params['linear_layer']['regularization_scale']))
        ]
        trunk_net = Pipeline(trunk_steps)
        return branch_net, trunk_net

    def _run_experiments(self, X_train, X_test, y_train, y_test, grid):
        results = {}
        experiment_count = 1
        num_experiments = self.params['number_of_experiments']
        search_space = self.params['search_space_deeponet']
        
        for n_modes, layer_width, branch_activation, reg_scale in itertools.product(
                search_space['n_modes'], 
                search_space['branch_network']['dense_layer']['layer_width'],
                search_space['branch_network']['dense_layer']['activation'],
                search_space['branch_network']['linear_layer']['regularization_scale']):
            
            branch_params = {
                'dense_layer': {
                    'layer_width': layer_width,
                    'activation': branch_activation,
                    'parameter_sampler': branch_activation
                },
                'linear_layer': {'regularization_scale': float(reg_scale)}
            }
            trunk_params = {
                'dense_layer': {
                    'layer_width': layer_width,
                    'activation': branch_activation, # same activation function as branch net
                    'parameter_sampler': branch_activation # same activation function as branch net
                },
                'linear_layer': {'regularization_scale': float(reg_scale)}
            }
            
            # Current experiment information
            print("Current experiment:")
            print("n_modes: ", n_modes)
            print("Branch net config: ", branch_params)
            print("Trunk net config: ", trunk_params)
            
            for i in range(num_experiments):
                seed = int(datetime.now().timestamp())

                # initialize branh and trunk networks
                branch_net, trunk_net = self._create_trunk_and_branch_network(branch_params, trunk_params, seed)
                
                # create model
                model = DeepONet(n_modes=n_modes, branch_pipeline=branch_net, trunk_pipeline=trunk_net)
                # fit model and save the experiment time
                start_time = time.time()
                results_dict = model.fit(X_train, y_train, grid, X_test, y_test)
                end_time = time.time()
                experiment_time = end_time - start_time

                # Extract test_info from the training results
                test_info_during_training = results_dict.pop('test_info', {})

                # Calculate the final test losses
                predictions = model.transform(X_test)
                num_test_samples = X_test.shape[0]
                test_loss_after_training = np.sum(np.linalg.norm(predictions - y_test, axis=1) / np.linalg.norm(y_test, axis=1)) / num_test_samples
                test_loss_after_training_mse = np.mean((predictions - y_test) ** 2)

                # Merge the test_info from training with the final test loss information
                test_info = {
                    **test_info_during_training,
                    'test_loss_after_training': test_loss_after_training,
                    'test_loss_after_training_mse': test_loss_after_training_mse
                }

                # Save results for the JSON file
                results[f"experiment_{experiment_count}_run_{i+1}"] = {
                    'n_modes': n_modes,
                    'branch_config': branch_params,
                    'trunk_config': trunk_params,
                    'training_info': {
                        **results_dict,
                        'experiment_time': experiment_time,
                        'random_seed': seed},
                    'test_info': test_info
                }

            experiment_count += 1
            
        n_modes = self.params['search_space_deeponet']['n_modes']  # Assuming 'n_modes' is stored in self.params
        if self.params['dataset'] == "burgers":
            burgers_filename = f'../outputs/burgers/results/burgers_results_n_{n_modes}.json'
            with open(burgers_filename, 'w') as file:
                json.dump(results, file, indent=4)
        elif self.params['dataset'] == "wave":
            wave_filename = f'../outputs/wave_son/results/wave_results_n_{n_modes}.json'
            with open(wave_filename, 'w') as file:
                json.dump(results, file, indent=4)
        else:
            raise ValueError("Invalid dataset name in params.yaml file.")


        return results

    @staticmethod
    def run(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, grid):
        runner = ExperimentRunner()
        return runner._run_experiments(X_train, X_test, y_train, y_test, grid)