import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState
from optuna.samplers import GridSampler

import util
from vnn_trainer import *
from training_data_wrapper import *
from drugcell_nn import *


class OptunaNNTrainer(VNNTrainer):

	def __init__(self, data_wrapper):
		super().__init__(data_wrapper)


	def exec_study(self):
		# Use this routine for GridSearch
		#search_space = {
		#	"genotype_hiddens": [4],
		#	"lr": [1.2e-4, 1.5e-4, 1.8e-4, 2e-4, 3e-4, 4e-4, 5e-4, 1e-3]
		#}
		#study = optuna.create_study(sampler=GridSampler(search_space), direction="maximize")
		#study.optimize(self.train_model, n_trials=8)

		study = optuna.create_study(direction="maximize")
		study.optimize(self.train_model, n_trials=20)
		return self.print_result(study)


	def setup_trials(self, trial):

		# Default routine; runs the genetic algorithm
		self.data_wrapper.genotype_hiddens = trial.suggest_categorical("genotype_hiddens", [4])
		self.data_wrapper.lr = trial.suggest_categorical("lr", [1.2e-4, 1.5e-4, 1.8e-4, 2e-4, 3e-4, 4e-4, 5e-4, 1e-3])

		batch_size = self.data_wrapper.batchsize
		if batch_size > len(self.train_feature)/4:
			batch_size = 2 ** int(math.log(len(self.train_feature)/4, 2))
			self.data_wrapper.batchsize = trial.suggest_categorical("batchsize", [batch_size])

		for key, value in trial.params.items():
			print("{}: {}".format(key, value))


	def train_model(self, trial):

		epoch_start_time = time.time()
		max_metric = 0.0
		min_loss = None
		early_stopping_counter = 0
		train_metric_at_min_loss = 0.0

		self.setup_trials(trial)

		self.model = DrugCellNN(self.data_wrapper)
		self.model.to(self.data_wrapper.device)

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.device)
		for name, param in self.model.named_parameters():
			if '_direct_gene_layer.weight' in name:
				term_name = name.split('_direct_gene_layer')[0]
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=True)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.wd)
		optimizer.zero_grad()

		if self.task == 'binary':
			print("epoch\ttrain_acc\ttrain_loss\tval_acc\tval_loss\telapsed_time")
		else:
			print("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\telapsed_time")
		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).to(self.data_wrapper.device)

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.to(self.data_wrapper.device))
				cuda_labels = Variable(labels.to(self.data_wrapper.device))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				loss_fn = self._get_loss_fn()
				aux_loss_fn = self._get_aux_loss_fn()
				for name, output in aux_out_map.items():
					if name == 'final':
						total_loss += loss_fn(output, cuda_labels)
					else:
						aux_loss = aux_loss_fn(output, cuda_labels)
						if not torch.isnan(aux_loss):
							total_loss += self.data_wrapper.alpha * aux_loss

				if torch.is_tensor(total_loss) and not torch.isnan(total_loss):
					total_loss.backward()
				else:
					continue

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_direct_gene_layer')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
				optimizer.step()

			train_metric, _ = self._compute_metrics(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).to(self.data_wrapper.device)

			val_loss = 0
			with torch.no_grad():
				for i, (inputdata, labels) in enumerate(val_loader):
					# Convert torch tensor to Variable
					features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
					cuda_features = Variable(features.to(self.data_wrapper.device))
					cuda_labels = Variable(labels.to(self.data_wrapper.device))

					aux_out_map, _ = self.model(cuda_features)

					if val_predict.size()[0] == 0:
						val_predict = aux_out_map['final'].data
						val_label_gpu = cuda_labels
					else:
						val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
						val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

					for name, output in aux_out_map.items():
						loss_fn = self._get_loss_fn()
						if name == 'final':
							val_loss += loss_fn(output, cuda_labels)

			val_metric, _ = self._compute_metrics(val_predict, val_label_gpu)

			epoch_end_time = time.time()
			if self.task == 'binary':
				print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
					epoch, train_metric, total_loss, val_metric, val_loss,
					epoch_end_time - epoch_start_time))
			else:
				true_auc = torch.mean(train_label_gpu)
				pred_auc = torch.mean(train_predict)
				print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
					epoch, train_metric, total_loss, true_auc, pred_auc,
					val_metric, val_loss, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

			trial.report(val_metric, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()

			if min_loss == None:
				min_loss = val_loss
			elif min_loss - val_loss > self.data_wrapper.delta:
				min_loss = val_loss
				early_stopping_counter = 0
				max_metric = val_metric
				train_metric_at_min_loss = train_metric
			elif min_loss - val_loss < self.data_wrapper.delta:
				early_stopping_counter += 1
				if early_stopping_counter >= self.data_wrapper.patience:
					break

		#torch.save(self.model, self.data_wrapper.modeldir + '/model_trial_' + str(trial.number) + '.pt')
		return max_metric


	def print_result(self, study):

		pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
		complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

		print("Study statistics:")
		print("Number of finished trials:", len(study.trials))
		print("Number of pruned trials:", len(pruned_trials))
		print("Number of complete trials:", len(complete_trials))

		print("Best trial:")
		best_trial = study.best_trial

		print("Value: ", best_trial.value)

		best_params = {}
		print("Params:")
		for key, value in best_trial.params.items():
			print("{}: {}".format(key, value))
			best_params[key] = value
		for key, value in best_trial.user_attrs.items():
			print("{}: {}".format(key, value))
			best_params[key] = value

		return best_params