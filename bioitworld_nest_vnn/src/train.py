import argparse
import copy

from vnn_trainer import *
from optuna_nn_trainer import *

def main():

	torch.set_printoptions(precision = 5)

	parser = argparse.ArgumentParser(description = 'Train VNN')
	parser.add_argument('-onto', help = 'Ontology file used to guide the neural network', type = str)
	parser.add_argument('-train', help = 'Training dataset', type = str)
	parser.add_argument('-epoch', help = 'Training epochs for training', type = int, default = 300)
	parser.add_argument('-lr', help = 'Learning rate', type = float, default = 0.001)
	parser.add_argument('-wd', help = 'Weight decay', type = float, default = 0.001)
	parser.add_argument('-alpha', help = 'Loss parameter alpha', type = float, default = 0.3)
	parser.add_argument('-batchsize', help = 'Batchsize', type = int, default = 64)
	parser.add_argument('-modeldir', help = 'Folder for trained models', type = str, default = 'MODEL/')
	parser.add_argument('-cuda', help = "GPU index, or 'cpu' for CPU-only", type = str, default = '0')
	parser.add_argument('-gene2id', help = 'Gene to ID mapping file', type = str)
	parser.add_argument('-cell2id', help = 'Cell to ID mapping file', type = str)
	parser.add_argument('-genotype_hiddens', help = 'Mapping for the number of neurons in each term in genotype parts', type = int, default = 4)
	parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
	parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
	parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)
	parser.add_argument('-fusions', help = 'Fusion information for cell lines', type = str, default = None)
	parser.add_argument('-task', help = 'Task type: continuous (regression) or binary (classification)', type = str, default = 'continuous', choices = ['continuous', 'binary'])
	parser.add_argument('-label', help = 'Label column to use from training data (e.g. os_months, binary_os, dfs_months, binary_dfs, pfs_months, binary_pfs)', type = str, default = None)
	parser.add_argument('-optimize', help = 'Hyper-parameter optimization', type = int, default = 1)
	parser.add_argument('-zscore_method', help='zscore method (zscore/robustz)', type=str, default = 'auc')
	parser.add_argument('-std', help = 'Standardization File', type = str, default = 'MODEL/std.txt')
	parser.add_argument('-patience', help = 'Early stopping epoch limit', type = int, default = 30)
	parser.add_argument('-delta', help = 'Minimum change in loss to be considered an improvement', type = float, default = 0.001)
	parser.add_argument('-min_dropout_layer', help = 'Start dropout from this Layer number', type = int, default = 2)
	parser.add_argument('-dropout_fraction', help = 'Dropout Fraction', type = float, default = 0.3)
	parser.add_argument('-mlflow',    help = 'Enable MLflow tracking (default: on; kept for backward compat)', action = 'store_true', default = True)
	parser.add_argument('-no_mlflow', help = 'Disable MLflow experiment tracking', action = 'store_true', default = False)
	parser.add_argument('-seed', help = 'Random seed for reproducible train/val split', type = int, default = None)

	opt = parser.parse_args()
	data_wrapper = TrainingDataWrapper(opt)

	if opt.optimize == 1:
		VNNTrainer(data_wrapper).train_model()

	elif opt.optimize == 2:
		trial_params = OptunaNNTrainer(data_wrapper).exec_study()
		for key, value in trial_params.items():
			if hasattr(data_wrapper, key):
				setattr(data_wrapper, key, value)
		VNNTrainer(data_wrapper).train_model()

	else:
		print("Wrong value for optimize.")
		exit(1)

if __name__ == "__main__":
	main()