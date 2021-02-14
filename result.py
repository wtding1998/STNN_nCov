 #-*-coding:utf-8 -*-
import json
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas
import torch
import scipy.io as scio
from keras.models import load_model

from get_dataset import get_relations, get_time_data, get_true
from stnn import (SaptioTemporalNN_A,
                  SaptioTemporalNN_classical, SaptioTemporalNN_I,)
from utils import DotDict, next_dir
import train_stnn


class Exp():
    def __init__(self, xp, outputdir):
        self.path = outputdir
        self.exp_name = xp
        self.dir_path = os.path.join(outputdir, xp)
        self.config = self.get_config()
        self.model_name = xp.split('_')[0]
        if self.model_name == 'keras':
            self.model_name == self.config['rnn_model']
        self.nt = self.config['nt']
        self.nx = self.config['nx']
        self.nd = self.config['nd']
        self.nz = self.config.get('nz')
        self.nt_train = self.config['nt_train']
        self.increase = False
        self.model = self.get_model()
        self.datas_order = self.config['datas_order']

    def get_relations(self):
        relations, _ = get_relations(self.config['datadir'], self.config['dataset'], self.config['khop'], 'all', self.config['relations_order'])
        return relations
        
    def get_log(self):
        with open(os.path.join(self.path, self.exp_name, 'logs.json')) as f:
            log = json.load(f)
        return log

    def get_config(self):
        with open(os.path.join(self.path, self.exp_name, 'config.json')) as f:
            config = json.load(f)
        return config

    def plot_log(self, data_names=['train_epoch.train_loss'], train=False, val=False, test=False, normalize=False, show_fig=False):
        """
        plot the figure of train loss, val loss and test loss
        data_names: the data to be poltted
        train: if true, then all the data in train will be ploted
        val: if true, then all the data in validation will be ploted
        test: if true, then all the data in test will be ploted
        """
        log_data = self.get_log()
        all_data_names = log_data.keys()
        plt.clf()
        if train:
            train_names = [a for a in all_data_names if "train" in a]
            data_names = list(set(train_names).union(set(data_names)))
        if val:
            valid_names = [a for a in all_data_names if "valid" in a]
            data_names = list(set(valid_names).union(set(data_names)))
        if test:
            test_names = [a for a in all_data_names if "test" in a]
            data_names = list(set(test_names).union(set(data_names)))
        for data_name in data_names:
            plotted_data = np.array(log_data[data_name])
            data_label = data_name
            if normalize:
                data_max = np.max(plotted_data)
                plotted_data = plotted_data / data_max
                data_label += "_" + str(np.ceil(data_max))
            x_axis = np.arange(plotted_data.shape[0])
            plt.plot(x_axis, plotted_data, label=data_label)
        plt.legend()
        plt.savefig(os.path.join(self.path, self.exp_name, 'logs.pdf'))
        if show_fig:
            plt.show()

    def print_bounded_epoch(self, want_val_bound=1e4, start_epoch=0, test_bound=1e8):
        """
        print the loss of epoch whose test loss and val loss are less than test and val bound
        """
        val_sum = self.get_log()['validation_epoch.sum']
        val_rmse = self.get_log()['validation_epoch.rmse']
        test_sum = self.get_log()['test_epoch.sum']
        bounded_val = []
        bounded_ind = []
        bounded_test = []
        bounded_rmse = []
        print('min sum val sum', val_sum[self.config['min_test_sum_epoch']])
        for i in range(len(val_sum)):
            if i > start_epoch:
                if val_sum[i] <= want_val_bound and test_sum[i] <= test_bound:
                    bounded_val.append(val_sum[i])
                    bounded_ind.append(i)
                    bounded_test.append(test_sum[i])
                    bounded_rmse.append(val_rmse[i])
                print(i, val_sum[i], val_rmse[i], test_sum[i])
        return bounded_val, bounded_ind, bounded_test, bounded_rmse

    def get_model(self):
        """
        Load model
        """
        print('Load', self.model_name, 'model.')
        if self.model_name == 'keras-rnn' or self.model_name =='GRU':
            model = load_model(os.path.join(self.path, self.exp_name, 'keras_model.h5'))
            self.formal_name = 'GRU'
        if self.model_name == 'stnn-classical' or self.model_name=='STNN-Classical':
            model = SaptioTemporalNN_classical(self.get_relations(), self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                                               self.config['nhid_de'], self.config['nlayers_de'], self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt'), map_location=torch.device('cpu')))
            self.formal_name = 'STNN'
        if self.model_name == 'lstnn-classical' or self.model_name=='STNN-Classical':
            model = SaptioTemporalNN_large_classical(self.get_relations(), self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                                               self.config['nhid_de'], self.config['nlayers_de'], self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt'), map_location=torch.device('cpu')))
            self.formal_name = 'lSTNN'
        if self.model_name == 'stnn-A' or self.model_name=='STNN-A':
            model = SaptioTemporalNN_A(self.get_relations(), self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                                               self.config['nhid_de'], self.config['nlayers_de'], self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt'), map_location=torch.device('cpu')))
            self.formal_name = 'STNN-A'

        if self.model_name == 'lstnn-A' or self.model_name=='STNN-A':
            model = SaptioTemporalNN_large_A(self.get_relations(), self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                                               self.config['nhid_de'], self.config['nlayers_de'], self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt'), map_location=torch.device('cpu')))
            self.formal_name = 'lSTNN-A'

        if self.model_name == 'stnn-I' or self.model_name=='STNN-I':
            model = SaptioTemporalNN_I(self.get_relations(), self.get_ground_truth(normalize=True, tensor_form=True)[:self.config['nt_train']], self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                                               self.config['nhid_de'], self.config['nlayers_de'], self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'], self.config['nhid_in'], self.config['nlayers_in'], self.config['dropout_in'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt'), map_location=torch.device('cpu')))
            self.formal_name = 'STNN-I'

        if self.model_name == 'lstnn-I' or self.model_name=='STNN-I':
            model = SaptioTemporalNN_large_I(self.get_relations(), self.get_ground_truth(normalize=True, tensor_form=True)[:self.config['nt_train']], self.config['nx'], self.config['nt_train'], self.config['nd'], self.config['nz'], self.config['mode'], self.config['nhid'], self.config['nlayers'],
                                               self.config['nhid_de'], self.config['nlayers_de'], self.config['dropout_f'], self.config['dropout_d'], self.config['activation'], self.config['periode'], self.config['nhid_in'], self.config['nlayers_in'], self.config['dropout_in'])
            model.load_state_dict(torch.load(os.path.join(self.path, self.exp_name, 'model.pt'), map_location=torch.device('cpu')))
            self.formal_name = 'lSTNN-I'
        return model

    def generate(self, t, rescaled=True):
        """
        Generate the prediction of the experiment
        rescaled: whether to get the true value the output of the model
        """
        x, z = self.model.generate(t)
        x = x.detach().numpy()
        if rescaled:
            x = self.get_true(x)
        return x

    def get_true(self, data):
        '''
        convert the scaled data into original scale
        '''
        normalize_config = {}
        normalize_config['normalize'] = self.config['normalize']
        normalize_config['nx'] = self.config['nx']
        normalize_config['std'] = self.config.get('std', None)
        normalize_config['mean'] = self.config.get('mean', None)
        normalize_config['min'] = self.config.get('min', None)
        normalize_config['max'] = self.config.get('max', None)
        if 'data_normalize' not in self.config.keys():
            normalize_config['data_normalize'] = 'd'
        else:
            normalize_config['data_normalize'] = self.config['data_normalize']
        if normalize_config['data_normalize'] == 'x':
            data = np.reshape(data, (-1, self.nx, self.nd))
        return get_true(data, normalize_config, use_torch=False)

    def get_stnn_fitting_prediction(self, nt_pred=0, datas='all'):
        '''
        Return the fitting data for training set and generate pred for nt_pred
        '''
        factors = self.model.factors
        train_pred = []
        for i in range(self.nt_train):
            train_pred.append(self.model.decode_z(factors[i]))
        train_pred = torch.stack(train_pred, dim=0)
        fitting = self.get_true(train_pred.detach().numpy())
        if nt_pred > 0:
            pred = self.generate(nt_pred)
            all_data = np.concatenate((fitting, pred), axis=0)
        else:
            all_data = fitting
        return self.resort(all_data, order=datas)

    def get_rnn_fitting_prediction(self, nt_pred=0):
        nt_train = self.config['nt_train']
        data, _ = get_time_data(self.config['datadir'], self.config['dataset'], time_datas='all', use_torch=False)
        normalize = self.config['normalize']
        train_data = data[:nt_train]
        mean = np.mean(train_data)
        seq_len = self.config['seq_length']
        if normalize == 'max_min':
            min = np.min(train_data)
            max = np.max(train_data)
            data = (data - mean) / (max-min)
        elif normalize == 'variance':
            std = np.std(train_data) * np.sqrt(train_data.size) / np.sqrt(train_data.size-1)
            data = (data - mean) / std
        # split train / test
        data = np.reshape(data, (self.nt, self.config['nx']*self.config['nd']))
        train_data = data[:nt_train]
        data_input = [] # (batch, squence_length, self.config['nx']*self.config['nd'])
        data_output = [] # (batch, self.config['nx']*self.config['nd'])
        for i in range(self.nt - seq_len):
            data_input.append(data[i:i+seq_len][np.newaxis, ...])
            data_output.append(data[i+seq_len][np.newaxis, ...])
        data_input = np.concatenate(data_input, axis=0)
        data_output = np.concatenate(data_output, axis=0)
        train_size = nt_train - seq_len
        train_input = data_input[:train_size]
        test_input = data[nt_train - seq_len:nt_train]
        train_fitting = []
        for i in range(seq_len):
            train_fitting.append(np.squeeze(data[i]))
        for i in range(train_input.shape[0]):
            train_fitting.append(np.squeeze(self.model.predict(train_input[[i]])))
        train_fitting = np.stack(train_fitting, axis=0)
        pred = []
        last_sequence = test_input[np.newaxis, ...]
        for i in range(nt_pred):
            new_pred = self.model.predict(last_sequence)
            pred.append(new_pred)
            new_pred = new_pred[np.newaxis, ...]
            last_sequence = np.concatenate([last_sequence[:, 1:, :], new_pred], axis=1)
        pred = np.concatenate(pred, axis=0)
        pred = np.reshape(pred, (nt_pred))
        fitting = np.concatenate((train_fitting, pred))
        if self.config['normalize'] == 'max_min':
            fitting = fitting * (max - min) + mean
        if self.config['normalize'] == 'variance':
            fitting = fitting * std + mean
        return fitting

    def get_ground_truth(self, normalize=False, tensor_form=False):
        if self.increase:
            dataset = self.config['dataset'].replace('_increase', '')
        else:
            dataset = self.config['dataset']
        if 'time_datas' in self.config.keys():
            data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset, time_datas=self.config['time_datas'], use_torch=False)
        else:
            data, _ = get_time_data(data_dir=self.config['datadir'], disease_name=dataset, time_datas='all', use_torch=False)
        if normalize:
            if self.config['normalize'] == 'max_min' and self.config['data_normalize'] != 'x':
                data = (data - self.config['mean']) / (self.config['max']-self.config['min'])
            elif self.config['normalize'] == 'variance' and self.config['data_normalize'] != 'x':
                data = (data - self.config['mean']) / self.config['std']
            elif self.config['normalize'] == 'variance' and self.config['data_normalize'] == 'x':
                for i in range(self.config['nx']):
                    data[:, i,:] = (data[:, i,:] - self.config['mean'][i]) / self.config['std'][i]
            elif self.config['normalize'] == 'max_min' and self.config['data_normalize'] == 'x':
                for i in range(self.config['nx']):
                    data[:, i,:] = (data[:, i,:] - self.config['mean'][i]) / (self.config['max'][i] - self.config['min'][i])
        if tensor_form:
            data = torch.Tensor(data)
        return data

    def get_total_fitting(self, nt_pred=None, indicate_data=None, save=False):
        '''
        Return the total fitting as a form of sum
        (nt_train + nt_pred)
        '''
        if not indicate_data:
            indicate_data = self.config['indicate_data']
        if not nt_pred:
            nt_pred = self.config['nt'] - self.config['nt_train']
        if 'stnn' in self.model_name:
            fitting = self.get_stnn_fitting_prediction(nt_pred=nt_pred, datas=[indicate_data])
            fitting = fitting.sum(1)
            fitting = np.squeeze(fitting)
        else:
            fitting = self.get_rnn_fitting_prediction(nt_pred=nt_pred)
        # save fitting
        if save:
            save_path = os.path.join(self.path, self.exp_name, self.formal_name+'.txt')
            np.savetxt(save_path, fitting)
        return fitting


    def resort(self, all_data, order='all'):
        # return the data according to the data kind
        if type(order) == list:
            choosen_data = []
            for data in order:
                idx = self.datas_order.index(data)
                choosen_data.append(all_data[:, :, idx])
            converted_data = np.stack(choosen_data, axis=2)
            return converted_data
        else:
            return all_data

    def plot_fitting(self, line_time=0, title='Pred', indicate_data='confirmed', show_fig=False):
        nt_val = self.config['nt_val']
        nt_pred = self.config['nt'] - self.config['nt_train']
        nt_test = nt_pred - nt_val
        fitting = self.get_total_fitting(indicate_data=indicate_data)
        ground_truth = self.get_ground_truth()
        ground_truth = self.resort(ground_truth, order=[indicate_data])
        ground_truth = ground_truth.sum(1)
        if line_time < 0:
            line_time = ground_truth.shape[0] - nt_pred
        nt = ground_truth.shape[0]
        x_axis = np.arange(nt)
        line_time = nt - nt_pred - 1
        plt.clf()
        plt.grid()
        plt.plot(x_axis, fitting, label=self.model_name, marker='*', linestyle='--')
        plt.plot(x_axis, ground_truth, label='ground_truth', marker='o')
        plt.axvline(x=line_time+1, ls="--")
        test_start_time = self.config['nt_train'] + self.config['nt_val']
        plt.axvline(x=test_start_time, ls="-")
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(self.path, self.exp_name, 'fitting.pdf'))
        # get ground truth
        ground_truth = self.get_ground_truth()
        ground_truth = self.resort(ground_truth, order=[indicate_data])
        ground_truth = ground_truth.sum(1)
        ground_truth = np.squeeze(ground_truth)
        # compute loss
        rmse_train_loss = np.linalg.norm(fitting[:-nt_test] - ground_truth[:-nt_test]) / np.sqrt(nt - nt_test)
        rmse_test_loss = np.linalg.norm(fitting[-nt_test:] - ground_truth[-nt_test:]) / np.sqrt(nt_test)
        print('rmse train loss: ', rmse_train_loss)
        print('rmse test loss: ', rmse_test_loss)
        self.config['rmse_train_loss'] = rmse_train_loss
        self.config['rmse_test_loss'] = rmse_test_loss
        self.save_config()

        if show_fig:
            plt.show()
        return ground_truth

    def save_config(self):
        with open(os.path.join(self.path, self.exp_name,'config.json'), 'w') as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

    def show_result(self, show=False):
        self.plot_fitting(show_fig=show)
        # plot logs
        self.plot_log(test=True, show_fig=show, normalize=True, val=True)
        print('min test sum: ', self.config['min_test_sum'])
        print('min test epoch: ', self.config['min_test_sum_epoch'])
        # print('final test sum: ', self.config['test_sum_score'])

    def run_min_exp(self, indicator='sum'):
        min_opt = self.config
        # min sum
        min_opt['nepoch'] = self.config['min_test_sum_epoch'] + 1
        min_opt['xp'] = self.config['xp'] + '_minsum'
        min_opt['outputdir'] = self.config['outputdir'].split('/')[-1]
        min_opt['xp_time'] = False
        min_opt['xp_model'] = False
        min_opt['run_min'] = False
        # min_opt['device'] = -1
        min_opt = DotDict(min_opt)
        if 'lstnn' in min_opt['xp']:
            train_lstnn.train_by_opt(min_opt)
        else:
            train_stnn.train_by_opt(min_opt)

    def rerun_exp(self, suffix='rerun'):
        min_opt = self.config
        # rerun experiment
        min_opt['nepoch'] = self.config['nepoch']
        min_opt['xp'] = self.exp_name + '_' + suffix
        min_opt['outputdir'] = self.config['outputdir'].split('/')[-1]
        # min_opt['xp_override'] = True
        min_opt['run_min'] = False
        min_opt['device'] = -1
        min_opt['load_exp'] = True
        min_opt['show_fig'] = True
        # min_opt['test'] = False
        min_opt = DotDict(min_opt)
        if 'lstnn' in min_opt['xp']:
            train_lstnn.train_by_opt(min_opt)
        else:
            train_stnn.train_by_opt(min_opt)

    def plot_relations(self):
        relations = self.config['relations_order']
        logs = self.get_log()
        nepoch = self.config['nepoch']
        epochs = np.arange(nepoch)
        # relations_result_dir = {}
        for i, relation in enumerate(relations):
            max_list = logs["train_epoch." + relation + "_max"]
            min_list = logs["train_epoch." + relation + "_min"]
            mean_list = logs["train_epoch." + relation + "_mean"]
            plt.title(relation + ' change ')
            plt.plot(epochs, max_list, label=relation + '_max')
            plt.plot(epochs, min_list, label=relation+'_min')
            plt.plot(epochs, mean_list, label=relation + '_mean')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()

class FolderResult():
    def __init__(self, outputdir):
        self.outputdir = outputdir

    def save_fitting(self):
        for exp_name in self.exps_name():
            print(exp_name)
            exp = Exp(exp_name, self.outputdir)
            exp.get_total_fitting(save=True)
        return 0

    def get_result_df(self, col=['train_loss', 'test_loss', 'true_loss', 'nhid', 'nlayers'], required_list = 'all', increase=False, nt_train=0, print_tex=False):
        if isinstance(required_list, str):
            required_list = next_dir(self.outputdir)
        df_dir = {}
        for exp_name in required_list:
            try:
                exp = Exp(exp_name, self.outputdir)
                config = exp.get_config()
                df_dir[exp_name] = config
            except:
                print(exp_name, ' x')
        df = pandas.DataFrame(df_dir)
        df = pandas.DataFrame(df.values.T, index=df.columns, columns=df.index)
        if nt_train > 0:
            df = df.loc[df['nt_train'] == nt_train]
        if 'increase' in df.columns:
            if increase:
                df = df.loc[df['increase'] == True]
            else:
                df = df.loc[df['increase'] == False]
        df['used_model'] = df.index
        for i in range(len(df.index)):
            exp_name = df.iloc[i, -1]
            used_model = exp_name.split('-')[0]
            df.iloc[i, -1] = used_model
        df = df[col]
        if print_tex:
            print(df.to_tex())
        return df

    def get_exps(self):
        return os.listdir(self.outputdir)

    def get_exps_dir(self):
        di = {}
        for exp in self.get_exps():
            di[exp] = exp
        return di

    def plot_fitting(self, exp_dir=None, start_time=0, indicate_data='confirmed'):
        '''
        pred : {'model_name': (nt_pred, nx, nd)}
        data : (nt, nx, nd)
        '''
        fitting_dir = {}
        folder = self.outputdir
        if not exp_dir:
            exp_dir = self.get_exps_dir()
        for model_name, exp_name in exp_dir.items():
            exp = Exp(exp_name, folder)
            fitting = exp.get_total_fitting(indicate_data=indicate_data)
            fitting_dir[exp_name] = fitting
        data = exp.get_ground_truth()
        data = exp.resort(data, order=['confirmed'])
        config = exp.config
        nt = config['nt']
        nt_train = config['nt_train']
        nt_val = config['nt_val']
        nt_test = nt - nt_train - nt_val
        data_sum = data.sum(1)
        data_sum = np.squeeze(data_sum)
        print(exp.config['dataset'])
        x_axis = np.arange(nt)
        # plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
        # plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure()
        plt.grid()
        plt.scatter(x_axis[start_time:], data_sum[start_time:], label='Ground Truth')
        for model_name, pred_sum in fitting_dir.items():
            plt.plot(x_axis[start_time:], pred_sum[start_time:], label=model_name)
        plt.axvline(x=nt_train,ls="--")
        plt.axvline(x=nt_train+nt_val,ls="-")
        plt.legend()
        plt.show()


class ExportResult():
    def __init__(self, dir_path, dataset='Jan', data_kind='confirmed'):
        self.dir_path = dir_path
        self.dataset = dataset
        self.data_kind = data_kind
        self.ground_truth = self.get_ground_truth()

    def get_ground_truth(self):
        data, data_order = get_time_data('data', self.dataset, use_torch=False)
        data = data[:, :, data_order.index(self.data_kind)]
        data = data.sum(1)
        data = np.squeeze(data)
        self.nt = data.shape[0]
        return data

    def get_exp_name(self):
        exps = os.listdir(self.dir_path)
        exp_dir = {}
        for i in exps:
            exp_name = i.split('.')[0]
            exp_dir[exp_name] = exp_name
        return exp_dir

    def split_mat(self, file_name):
        data = scio.loadmat(os.path.join(self.dir_path, file_name))
        fitting_datas = data['predictions'][0]
        fitting_dir = {}
        for i in range(len(fitting_datas)):
            predictors = fitting_datas[i]
            pred_name = predictors[0][0]
            train_fitting = np.squeeze(predictors[1])
            test_fitting = np.squeeze(predictors[2])
            fitting = np.concatenate((train_fitting, test_fitting))
            fitting_dir[pred_name] = fitting
            np.savetxt(os.path.join(self.dir_path, pred_name+'.txt'), fitting)
        return fitting_dir

    def split_SEIR_mat(self, file_name):
        data = scio.loadmat(os.path.join(self.dir_path, file_name))
        for k, v in data.items():
            print(k)
            if 'Infected' in k:
                fitting = np.squeeze(v)
        np.savetxt(os.path.join(self.dir_path, 'SEIR'+'.txt'), fitting)
        return fitting

    def get_exp_data(self, exp_dir=None):
        if not exp_dir:
            exp_dir = self.get_exp_name()
        print(exp_dir)
        fitting_dir = {}
        for k, v in exp_dir.items():
            if os.path.exists(os.path.join(self.dir_path, v + '.txt')):
                data = np.genfromtxt(os.path.join(self.dir_path, v + '.txt'))
                fitting_dir[k] = data
        return fitting_dir

    def ax_plot(self, ax, nt_test=1, plot_train_day=30, exp_dir=None, complete_no_train=True, xmax=None):
        ground_truth = self.get_ground_truth()
        x_axis = np.arange(ground_truth.shape[0])
        truth_test = ground_truth[-nt_test:]
        fitting_dir = self.get_exp_data(exp_dir)
        train_loss_dir = {}
        test_loss_dir = {}
        # ground_truth_data
        ax.scatter(x_axis, ground_truth, label='Ground Truth')
        models=['STNN', 'STNN-A', 'STNN-I', 'BPNN', 'GRU', 'GAUSS', 'EXP', 'POLY', 'SEIR']
        for model_name in models:
            if model_name in fitting_dir.keys():
                fitting = fitting_dir[model_name]
                test_fitting = fitting[-nt_test:]
                train_fitting = fitting[:-nt_test]
                nt_train = train_fitting.shape[0]
                nt_fitting = fitting.shape[0]
                test_loss = np.linalg.norm(test_fitting - truth_test) / np.sqrt(nt_test)
                truth_train = ground_truth[-nt_test-nt_train:-nt_test]
                train_loss = np.linalg.norm(train_fitting - truth_train) / np.sqrt(nt_train)
                if complete_no_train:
                    complete_fitting = np.concatenate((ground_truth[:-nt_fitting], fitting))
                    ax.plot(x_axis, complete_fitting, label=model_name)
                else:
                    ax.plot(x_axis[-nt_fitting:], fitting, label=model_name)
                print('-'*25)
                print(model_name)
                print('train loss: ',train_loss)
                print('test loss: ',test_loss)
                train_loss_dir[model_name] = train_loss
                test_loss_dir[model_name] = test_loss
        start_time = np.max([0, ground_truth.shape[0] - plot_train_day])
        if not xmax:
            ax.set_xlim(xmin=start_time, xmax=xmax)
        else:
            ax.set_xlim(xmin=start_time)
        ax.axvline(x=self.nt-nt_test,ls="--")
        ax.set_xlabel('Days')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)
        ax.yaxis.get_offset_text().set_fontsize(30)
        return train_loss_dir, test_loss_dir

def generate_China_result():
    # Plot
    fig = plt.figure(figsize=(15, 15))
    axes = fig.subplots(nrows=2, ncols=2)
    dirs = ['Jan', 'Feb', 'Mar', 'Peak']
    df = pandas.DataFrame()
    # plot_train_days = [30, 30, 30, 30]
    plot_train_days = [10, 10, 10, 10]

    for i in range(4):
        ax = fig.axes[i]
        a = ExportResult('../output_result/' + dirs[i]+'_plot', dataset=dirs[i]+'_rnn')
        train_loss, test_loss = a.ax_plot(ax, nt_test=5, plot_train_day=plot_train_days[i])
        df[dirs[i]+'_train'] = pandas.Series(train_loss)
        df[dirs[i]+'_test'] = pandas.Series(test_loss)
        # ax.set_title(dirs[i], y=-0.2, fontsize=30)
        ax.set_title(dirs[i], fontsize=40)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper center', bbox_to_anchor=(0.5, 0.999), ncol=5, fontsize=26)
    # plt.subplots_adjust(left=0.00, right=0.96, top=0.9, bottom=0.07, wspace =0.1, hspace =0.23)
    # pandas.set_option('display.float_format', '{:.3E}'.format)
    df = df / 1000
    pandas.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df.to_latex())
    plt.subplots_adjust(left=0.07, right=0.96, top=0.8, bottom=0.1, hspace =0.6)
    # plt.savefig('China.pdf')
    plt.savefig('China.eps')
    plt.show()

def generate_Inter_result():
    # Plot
    fig = plt.figure(figsize=(15, 8))
    axes = fig.subplots(nrows=1, ncols=2)
    plot_train_days = [150, 90]
    dirs = ['USA', 'ITALY']
    test_days = [30,  15]
    x_maxs = [335, 320]
    df = pandas.DataFrame()
    major_ls = [30, 20]

    for i in range(2):
        ax = fig.axes[i]
        a = ExportResult('../output_result/' + dirs[i]+'_plot', dataset=dirs[i]+'_rnn')
        train_loss, test_loss = a.ax_plot(ax, nt_test=test_days[i], plot_train_day=plot_train_days[i], xmax=x_maxs[i])
        df[dirs[i]+'_train'] = pandas.Series(train_loss)
        df[dirs[i]+'_test'] = pandas.Series(test_loss)
        # ax.set_title(dirs[i], y=-0.2, fontsize=30)
        ax.set_title(dirs[i], fontsize=40)
        x_major_locator=MultipleLocator(major_ls[i])
        ax.xaxis.set_major_locator(x_major_locator)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper center', bbox_to_anchor=(0.5, 0.999), ncol=5, fontsize=26)
    # plt.subplots_adjust(left=0.00, right=0.96, top=0.9, bottom=0.07, wspace =0.1, hspace =0.23)
    # pandas.set_option('display.float_format', '{:.3E}'.format)
    df = df / 10000
    df = df.fillna(0)
    pandas.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df.to_latex())
    plt.subplots_adjust(left=0.07, right=0.96, top=0.70, bottom=0.13, hspace =0.6)
    plt.savefig('Inter.pdf')
    plt.savefig('Inter.eps')
    plt.show()

def generate_result(dataset='China'):
    if dataset == 'China':
        dirs = ['Jan', 'Feb', 'Mar', 'Peak']
        plot_train_days = [50, 50, 50, 50]
        test_days = [5, 5, 5, 5]
        scale = 3
        # x_major_locator = MultipleLocator(5)
        # y_major_locator = MultipleLocator(1e4)
    else:
        dataset = 'Inter'
        dirs = ['USA', 'UK', 'FRANCE', 'ITALY']
        test_days = [30, 20, 15, 15]
        scale = 4
        # x_major_locator = MultipleLocator(10)
        # y_major_locator = MultipleLocator(1e6)
        plot_train_days = [40, 30, 30, 30]
    fig = plt.figure(figsize=(15, 15))
    axes = fig.subplots(nrows=2, ncols=2)
    df = pandas.DataFrame()
    for i in range(4):
        ax = fig.axes[i]
        a = ExportResult('../output_result/' + dirs[i]+'_plot', dataset=dirs[i]+'_rnn')
        train_loss, test_loss = a.ax_plot(ax, nt_test=test_days[i], plot_train_day=plot_train_days[i])
        df[dirs[i]+'_train'] = pandas.Series(train_loss)
        df[dirs[i]+'_test'] = pandas.Series(test_loss)
        # ax.set_title(dirs[i], y=-0.2, fontsize=30)
        ax.set_title(dirs[i], fontsize=40)
        # ax.xaxis.set_major_locator(x_major_locator)
        # ax.yaxis.set_major_locator(y_major_locator)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper center', bbox_to_anchor=(0.5, 0.999), ncol=5, fontsize=26)
    # plt.subplots_adjust(left=0.00, right=0.96, top=0.9, bottom=0.07, wspace =0.1, hspace =0.23)
    df = df / 10 ** (scale)
    df = df.fillna(0)
    pandas.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df.to_latex())
    plt.subplots_adjust(left=0.07, right=0.96, top=0.8, bottom=0.1, hspace =0.6)
    plt.savefig(dataset + '.eps')
    # plt.savefig(dataset + '.pdf')
    plt.show()

if __name__ == "__main__":

    generate_Inter_result()
    # generate_result('China')
    #
    # Jan
    # output fitting file
    # b = Printer('../output_result/Jan')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/Jan_plot', dataset='Jan_rnn')
    # a.split_SEIR_mat('seir_case1.mat')
    # a.plot_by_dir(nt_test=5)

    # Feb
    # b = Printer('../output_result/Feb_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/Feb_plot', dataset='Feb_rnn')
    # a.split_SEIR_mat('seir_case2.mat')
    # a.plot_by_dir(nt_test=5)

    # Mar
    # b = Printer('../output_result/Mar_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/Mar_plot', dataset='Mar_rnn')
    # a.split_SEIR_mat('seir_case3.mat')
    # a.plot_by_dir(nt_test=5)

    # Peak
    # b = Printer('../output_result/Peak_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/Peak_plot', dataset='Peak_rnn')
    # a.split_SEIR_mat('seir_case4.mat')
    # a.plot_by_dir(nt_test=5)

    # Italy
    # b = Printer('../output_result/ITALY_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/ITALY_plot', dataset='ITALY_rnn')
    # a.split_SEIR_mat('Italy_seir.mat')
    # a.plot_by_dir(nt_test=15)

    # US
    # b = Printer('../output_result/USA_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/USA_plot', dataset='USA_rnn')
    # a.split_SEIR_mat('us_seir.mat')
    # a.plot_by_dir(nt_test=30)
    #
    # FRANCE
    # b = Printer('../output_result/FRANCE_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/FRANCE_plot', dataset='FRANCE_rnn')
    # a.split_SEIR_mat('france_seir.mat')
    # a.split_mat('France.mat')
    # a.plot_by_dir(nt_test=30)

    # UK
    # b = Printer('../output_result/UK_plot')
    # b.save_fitting()
    # output fitting figure
    # a = all_exp_result('../output_result/UK_plot', dataset='UK_rnn')
    # a.split_SEIR_mat('uk_seir.mat')
    # a.split_mat('UK.mat')
    # a.plot_by_dir(nt_test=30)
