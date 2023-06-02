import numpy as np  
import os
import pdb

base_folder = '/media/adam/dc156fa0-1275-46c2-962c-bc8c9fcf1cb0/ucr_data/data1/contrastive_learning/MPL_save'

config_folders = ['balanced_MPL_ML_MS_new'] # 'balanced_MPL', 'balanced_MPL_MS'

for config in config_folders:

	load_folder = os.path.join(base_folder, config)

	metrics = []
	roc_auc = []
	for fold in range(5):

		fold_metrics = os.path.join(load_folder, f'fold_{fold}')

		metrics.append(np.loadtxt(os.path.join(fold_metrics, 'teacher_metrics.csv'), delimiter=","))
		# metrics.append(np.loadtxt(os.path.join(fold_metrics, 'finetune_metrics.csv'), delimiter=","))

		# roc_auc.append(np.loadtxt(os.path.join(fold_metrics, 'finetune_roc_auc.csv'), delimiter=","))

	metrics_mean = np.array(metrics).mean(0)
	metrics_std = np.array(metrics).std(0)
	# roc_auc = np.array(roc_auc)

	np.savetxt(os.path.join(load_folder, 'teacher_tpr_fold.csv'), np.array([metrics_mean[0], metrics_std[0]]), fmt="%0.4f", delimiter=",")
	np.savetxt(os.path.join(load_folder, 'teacher_tnr_fold.csv'), np.array([metrics_mean[1], metrics_std[1]]), fmt="%0.4f", delimiter=",")
	np.savetxt(os.path.join(load_folder, 'teacher_f1_fold.csv'), np.array([metrics_mean[2], metrics_std[2]]), fmt="%0.4f", delimiter=",")
	np.savetxt(os.path.join(load_folder, 'teacher_acc_fold.csv'), np.array([metrics_mean[3], metrics_std[3]]), fmt="%0.4f", delimiter=",")
	# np.savetxt(os.path.join(load_folder, 'roc_auc_fold.csv'), np.array([roc_auc.mean(0), roc_auc.std(0)]), fmt="%0.4f", delimiter=",")

	# np.savetxt(os.path.join(load_folder, 'tpr_fold.csv'), np.array([metrics_mean[0], metrics_std[0]]), fmt="%0.4f", delimiter=",")
	# np.savetxt(os.path.join(load_folder, 'tnr_fold.csv'), np.array([metrics_mean[1], metrics_std[1]]), fmt="%0.4f", delimiter=",")
	# np.savetxt(os.path.join(load_folder, 'f1_fold.csv'), np.array([metrics_mean[2], metrics_std[2]]), fmt="%0.4f", delimiter=",")
	# np.savetxt(os.path.join(load_folder, 'acc_fold.csv'), np.array([metrics_mean[3], metrics_std[3]]), fmt="%0.4f", delimiter=",")

