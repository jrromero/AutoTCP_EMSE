# J.R. Romero, A. Ramírez, C. García.
# "Automated machine learning for test case prioritisation".
# 2025

# Script used to train on the ahmadreza dataset

import os
import subprocess

base_path = os.path.dirname(os.path.abspath(__file__))
versions_path = os.path.join(base_path, '../datasets/data/')

if __name__ == '__main__':
	for index, seed in enumerate([000, 123, 456, 789, 231]):  # , 564, 897, 321, 654, 987]):  # , 000]): # 123
		# Settings
		train_program = 'evoflow_train.py' # To train in all versions
		datasets_folder = versions_path.replace(' ', '\ ')
		results_folder = f'out_ahmadreza_1seed_auc_reduce_75_75_{index}' # Results folder should end in _1, _2, _3, etc. to indicate the iteration
		config_file = os.path.join(base_path, 'evoflow/configs/evoflow_1h.py')
		cv = 5 # Number of folds
		processes = 3 # Number of parallel processes

		# Command line to train in all versions
		cmd = f'python3 {train_program} {datasets_folder} {results_folder} {config_file} -cv {cv} -s {seed} -p {processes}'
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		stdout, stderr = p.communicate()
		print(stdout.decode('utf-8'))
		print(stderr.decode('utf-8'))
		
