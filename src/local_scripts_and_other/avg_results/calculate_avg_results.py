import os
from collections import defaultdict
import numpy as np

base_dir = '../../../exp/cross_validation_anditi/'
sampling_method = 'dynamic/gaussian'
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

with open(f'./{sampling_method.replace("/", "_")}_results_avg.txt', 'w') as avg_file:
    for i, (line, vals) in enumerate(results.items()):
        vals_np = np.array(vals)
        avg = np.average(vals_np)
        stddev = np.sqrt(np.var(vals_np))

        avg_file.write(f"{line}: {avg.item():.3f} +- {stddev.item():.3f}\n")

        if i > 0 and (i + 1) % 3 == 0:
            avg_file.write("\n")