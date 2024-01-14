import json
import os

def average_results(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)

    averaged_results = {}
    experiment_count = 1

    while True:
        experiment_key = f'experiment_{experiment_count}'
        run_keys = [f'{experiment_key}_run_{i}' for i in range(1, 4)]

        if not all(key in data for key in run_keys):
            break 

        averaged_experiment = {key: data[run_keys[0]][key] for key in data[run_keys[0]] if key not in ['training_info', 'test_info', 'random_seed']}
        averaged_experiment['training_info'] = {}
        averaged_experiment['training_info']['random_seeds'] = [data[key]['training_info']['random_seed'] for key in run_keys]

        
        averaged_experiment['test_info'] = {}

        for key in run_keys:
            for section in ['training_info', 'test_info']:
                for sub_key, sub_value in data[key][section].items():
                    if sub_key == 'is_converged':
                        averaged_experiment[section].setdefault('is_converged_values', [])
                        averaged_experiment[section]['is_converged_values'].append(sub_value)
                    elif isinstance(sub_value, (int, float)):
                        averaged_experiment[section].setdefault(sub_key, 0)
                        averaged_experiment[section][sub_key] += sub_value / len(run_keys)

        averaged_results[experiment_key] = averaged_experiment
        experiment_count += 1

    with open(output_file, 'w') as file:
        json.dump(averaged_results, file, indent=4)
        print(f"Created file: {output_file}")

base_directory = 'outputs/wave_son/results'
x_values = [2, 4, 8, 16, 32, 64, 128, 256]

for x in x_values:
    input_file = os.path.join(base_directory, f'wave_results_n_[{x}].json')
    output_file = os.path.join(base_directory, f'wave_results_n_[{x}]_avg.json')
    print(f"Processing file: {input_file}")
    average_results(input_file, output_file)
