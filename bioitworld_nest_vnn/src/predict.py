import argparse
import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import util


def predict(predict_data, gene_dim, model_file, hidden_folder, batch_size, result_file, cell_features, task, mlflow_enabled=False, label=None, clinical_features=None):

	feature_dim = gene_dim

	model = torch.load(model_file, map_location=DEVICE, weights_only=False)

	predict_feature, predict_label = predict_data

	predict_label_gpu = predict_label.to(DEVICE)

	model.to(DEVICE)
	model.eval()

	test_loader = du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

	#Test
	test_predict = torch.zeros(0,0).to(DEVICE)
	hidden_embeddings_map = {}

	saved_grads = {}
	def save_grad(element):
		def savegrad_hook(grad):
			saved_grads[element] = grad
		return savegrad_hook

	for batch_idx, (inputdata, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		features = util.build_input_vector(inputdata, cell_features)

		cuda_features = Variable(features.to(DEVICE), requires_grad=True)
		clinical_batch = util.build_clinical_vector(inputdata, clinical_features).to(DEVICE) if clinical_features is not None and clinical_features.shape[1] > 0 else None

		# make prediction for test data
		aux_out_map, hidden_embeddings_map = model(cuda_features, clinical=clinical_batch)

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		# First batch overwrites; subsequent batches append within the same run.
		file_mode = 'wb' if batch_idx == 0 else 'ab'

		for element, hidden_map in hidden_embeddings_map.items():
			hidden_file = hidden_folder + '/' + element + '.hidden'
			with open(hidden_file, file_mode) as f:
				np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

		for element, _ in hidden_embeddings_map.items():
			hidden_embeddings_map[element].register_hook(save_grad(element))

		## Do backprop
		aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))

		# Save Feature Grads
		for feat_i in range(len(cuda_features[0, 0, :])):
			feature_grad = cuda_features.grad.data[:, :, feat_i]
			with open(result_file + '_feature_grad_' + str(feat_i) + '.txt', file_mode) as f:
				np.savetxt(f, feature_grad.cpu().numpy(), '%.4e', delimiter='\t')

		# Save Hidden Grads
		for element, hidden_grad in saved_grads.items():
			hidden_file = hidden_folder + '/' + element + '.hidden_grad'
			with open(hidden_file, file_mode) as f:
				np.savetxt(f, hidden_grad.data.cpu().numpy(), '%.4e', delimiter='\t')

	if task == 'binary':
		test_probs = torch.sigmoid(test_predict)
		test_preds_binary = (test_probs >= 0.5).float()
		correct = (test_preds_binary.view(-1) == predict_label_gpu.view(-1)).float()
		acc = correct.sum() / len(correct)
		print("Test accuracy\t%s\t%.4f" % (model.root, acc))
		# Save probabilities and binary predictions
		np.savetxt(result_file + '_probabilities.txt', test_probs.cpu().numpy(), '%.4e')
		np.savetxt(result_file + '_predictions.txt', test_preds_binary.cpu().numpy(), '%d')
		metric_value, metric_name = acc.item(), "test_accuracy"
	else:
		test_corr = util.pearson_corr(test_predict, predict_label_gpu)
		print("Test correlation\t%s\t%.4f" % (model.root, test_corr))
		metric_value, metric_name = float(test_corr), "test_pearson_r"

	if mlflow_enabled:
		try:
			import mlflow
			from pathlib import Path
			mlflow.set_experiment("nest_vnn")
			model_path = Path(model_file)
			study_id = (model_path.parts[model_path.parts.index("output") + 1]
			            if "output" in model_path.parts else "unknown")
			# Read training run_id saved by vnn_trainer
			run_id_path = model_path.parent / "mlflow_run_id.txt"
			training_run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
			with mlflow.start_run(parent_run_id=training_run_id or None) as active_run:
				params = {"study_id": study_id}
				if label:
					params["label"] = label
				mlflow.log_params(params)
				mlflow.log_metric(metric_name, metric_value)

				# Persist run_id so annotate step can resume this run to log artifacts
				predict_run_id_path = Path(result_file).parent / "mlflow_run_id.txt"
				predict_run_id_path.write_text(active_run.info.run_id)
		except Exception as e:
			print(f"Warning: MLflow logging failed: {e}")

	np.savetxt(result_file + '.txt', test_predict.cpu().numpy(),'%.4e')


parser = argparse.ArgumentParser(description='Predict VNN')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)
parser.add_argument('-load', help='Model file', type=str)
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='hidden/')
parser.add_argument('-result', help='Result file prefix', type=str, default='result/predict')
parser.add_argument('-cuda', help="GPU index, or 'cpu' for CPU-only", type=str, default='0')
parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)
parser.add_argument('-fusions', help = 'Fusion information for cell lines', type = str, default = None)
parser.add_argument('-task', help = 'Task type: continuous or binary', type = str, default = 'continuous', choices = ['continuous', 'binary'])
parser.add_argument('-label', help = 'Label column to use from test data', type = str, default = None)
parser.add_argument('-std', help = 'Standardization File', type = str)
parser.add_argument('-mlflow',    help = 'Enable MLflow tracking (default: on; kept for backward compat)', action = 'store_true', default = True)
parser.add_argument('-no_mlflow', help = 'Disable MLflow experiment tracking', action = 'store_true', default = False)

opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_data, cell2id_mapping = util.prepare_predict_data(opt.predict, opt.cell2id, opt.std, opt.label, opt.task)
gene2id_mapping = util.load_mapping(opt.gene2id, "genes")

# load cell/drug features
mutations = np.genfromtxt(opt.mutations, delimiter = ',')
cn_deletions = np.genfromtxt(opt.cn_deletions, delimiter = ',')
cn_amplifications = np.genfromtxt(opt.cn_amplifications, delimiter = ',')

feature_layers = [mutations, cn_deletions, cn_amplifications]
if opt.fusions is not None:
	fusions = np.genfromtxt(opt.fusions, delimiter = ',')
	feature_layers.append(fusions)
cell_features = np.dstack(feature_layers)

num_cells = len(cell2id_mapping)
num_genes = len(gene2id_mapping)

# Load covariate columns (cov_*) from predict file if present
import pandas as _pd
_clinical_features = None
with open(opt.predict) as _f:
    _header_line = _f.readline().strip()
if 'cell_line' in _header_line:
    _cov_cols = [c for c in _header_line.split('\t') if c.startswith('cov_')]
    if _cov_cols:
        _df = _pd.read_csv(opt.predict, sep='\t')
        _clin_arr = np.zeros((num_cells, len(_cov_cols)))
        for _, _row in _df.iterrows():
            _sid = _row['cell_line']
            if _sid not in cell2id_mapping:
                continue
            _idx = cell2id_mapping[_sid]
            for _j, _col in enumerate(_cov_cols):
                _val = _row[_col]
                try:
                    _clin_arr[_idx, _j] = float(_val) if _pd.notna(_val) else 0.0
                except (ValueError, TypeError):
                    pass
        _clinical_features = _clin_arr

from training_data_wrapper import resolve_device
DEVICE = resolve_device(opt.cuda)

predict(predict_data, num_genes, opt.load, opt.hidden, opt.batchsize, opt.result, cell_features, opt.task, mlflow_enabled=not opt.no_mlflow, label=opt.label, clinical_features=_clinical_features)