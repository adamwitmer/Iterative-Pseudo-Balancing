import numpy as np  
import os
import pdb

base_folder = ''  # save folder

config_folders = []  # configuration subfolders

models = ['student']

for config in config_folders:

	for model in models:

		load_folder = os.path.join(base_folder, config)
	
		metrics = []
		roc_auc = []
		for fold in range(5):
	
			fold_metrics = os.path.join(load_folder, f'fold_{fold}')
	
			metrics.append(np.loadtxt(os.path.join(fold_metrics, f'{model}_metrics.csv'), delimiter=","))
	
		metrics_mean = np.array(metrics).mean(0)
		metrics_std = np.array(metrics).std(0)
	
		np.savetxt(os.path.join(load_folder, f'{model}_tpr_fold.csv'), np.array([metrics_mean[0], metrics_std[0]]), fmt="%0.4f", delimiter=",")
		np.savetxt(os.path.join(load_folder, f'{model}_tnr_fold.csv'), np.array([metrics_mean[1], metrics_std[1]]), fmt="%0.4f", delimiter=",")
		np.savetxt(os.path.join(load_folder, f'{model}_f1_fold.csv'), np.array([metrics_mean[2], metrics_std[2]]), fmt="%0.4f", delimiter=",")
		np.savetxt(os.path.join(load_folder, f'{model}_acc_fold.csv'), np.array([metrics_mean[3], metrics_std[3]]), fmt="%0.4f", delimiter=",")

