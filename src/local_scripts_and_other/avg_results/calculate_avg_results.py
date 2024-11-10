import os
from collections import defaultdict

base_dir = '../../../exp/cross_validation_anditi/'
sampling_method = 'gaussian'
runs_dir = os.path.join(base_dir, sampling_method)

results = defaultdict(list)

for curr_run_name in os.listdir(runs_dir):
    run_dir = os.path.join(runs_dir, curr_run_name)
    
    if os.path.isdir(run_dir):
        results_path = os.path.join(run_dir, 'results.txt')
        
        if os.path.isfile(results_path):
            print(f'Found results.txt in dir: {run_dir}')

            with open(results_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    if line:
                        split_line = line.split(':')
                        results[split_line[0]].append(float(split_line[1].strip()))

with open(f'./{sampling_method}_results_avg.txt', 'w') as avg_file:
    for line, vals in results.items():
        avg_val = sum(vals) / len(vals)
        avg_file.write(f"{line}: {avg_val}\n")