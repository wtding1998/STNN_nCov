 #-*-coding:utf-8 -*-

import os
import random
import json
from collections import defaultdict
import datetime

import configargparse
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.optim as optim

from get_dataset import get_stnn_data, get_true_indicate
from utils import DotDict, Logger, rmse, boolean_string, get_dir, get_time, time_dir, rmse_sum_confirmed, sample
from stnn import SaptioTemporalNN_classical, SaptioTemporalNN_A, SaptioTemporalNN_I
import result

def get_opt():
    """
    get options from command line
    """
    p = configargparse.ArgParser()
    # -- data
    p.add('--datadir', type=str, help='path to dataset', default='data')
    p.add('--dataset', type=str, help='dataset name', default='Italy')
    p.add('--nt_train', type=int, help='time for training', default=250)
    p.add('--nt_val', type=int, help='time for validation', default=10)
    p.add('--data_normalize', type=str, help='data normalization object (d:for all data|x: for each region)', default='x')
    p.add('--relation_normalize', type=str, help='relation normalizatino object (all:for all data|row: for each row)', default='all')
    p.add('--relations', type=str, nargs='+', help='list of relations', default='all')
    p.add('--time_datas', type=str, nargs='+', help='list of data', default='all')
    p.add('--indicate_data', type=str, help='the data whose val_loss and test_loss are computed', default='confirmed')
    # -- output
    p.add('--outputdir', type=str, help='path to save xp', default='test')
    p.add('--xp', type=str, help='xp name', default='stnn')
    p.add('--xp_model', type=boolean_string, help='add model name after xp', default=True)
    p.add('--xp_time', type=boolean_string, help='add time after xp', default=True)
    # -- model
    p.add('--model', type=str, help='STNN model (classical|A|I)', default='classical')
    p.add('--mode', type=str, help='STNN mode (default|refine|discover)', default='default')
    p.add('--nz', type=int, help='laten factors size', default=1)
    p.add('--activation', type=str, help='dynamic module activation function (identity|sigmoid|tanh)', default='tanh')
    p.add('--khop', type=int, help='spatial depedencies order', default=1)
    p.add('--nhid', type=int, help='hidden size of state network', default=0)
    p.add('--nlayers', type=int, help='num of layers of state network', default=1)
    p.add('--nhid_de', type=int, help='hidden size of observation network', default=0)
    p.add('--nlayers_de', type=int, help='num of layers of observation network', default=1)
    p.add('--nhid_in', type=int, help='hidden size of input gate (for STNN-I) ', default=0)
    p.add('--nlayers_in', type=int, help='num of layers of input gate (for STNN-I)', default=1)
    p.add('--dropout_f', type=float, help='dropout for input factors', default=0)
    p.add('--dropout_d', type=float, help='dropout for state network', default=0)
    p.add('--dropout_de', type=float, help='dropout for observation network', default=0)
    p.add('--dropout_in', type=float, help='dropout for input gate', default=0)
    p.add('--wd', type=float, help='l2 regularization on parametrs of networks', default=1e-6)
    p.add('--wd_z', type=float, help='l2 regularzation on latent factors', default=1e-3)
    p.add('--l2_z', type=float, help='l2 between consecutives latent factors (to make them more continous)', default=0.)
    p.add('--l1_rel', type=float, help='l1 regularization on relation discovery mode', default=0.)
    p.add('--lambd', type=float, help='lambda between observation and state losses', default=1)
    # -- optim
    p.add('--lr', type=float, help='learning rate', default=1e-3)
    p.add('--optimizer', type=str, help='learning algorithm (Adam|Rmsprop|SGD|Adagrad)', default='Adam')
    p.add('--beta1', type=float, help='adam beta1', default=.9)
    p.add('--beta2', type=float, help='adam beta2', default=.999)
    p.add('--eps', type=float, help='adam eps', default=1e-8)
    # -- learning
    p.add('--nepoch', type=int, help='number of epochs to train for', default=10000)
    p.add('--batch_size', type=int, help='batch size', default=1000000,)
    p.add('--sample_method', type=str, help='sample method (swr: sampling with replacement|\
    iswr: independent sampling without replacement|nswr: tau-nice sampling without replacement|uniform)', default='uniform')
    # swr: sampling with replacement
    p.add('--validate', type=boolean_string, help='validate during training', default=True,)
    # reduce lr according to val loss
    p.add('--es_start', type=int, help='num of epoch to start considering early stop', default=0)
    p.add('--patience', type=int, help='number of epoch to wait before trigerring lr decay', default=100)
    p.add('--es_val_bound', type=float, help='upper bound of val_loss when the epoch is taken into consideration of early stop', default=1e7)
    p.add('--es_wd_factor', type=float, help='lr decay factor after patience', default=0.1)
    # reduce lr after certain epoch or not
    p.add('--reduce_start', type=int, help='num of epoch to start reduce, 0 will not activate this functino', default=0)
    p.add('--reduce_factor', type=float, help='lr reduce factor', default=0.1)
    p.add('--test', type=boolean_string, help='save test loss after each epoch', default=True)
    p.add('--train_meantime', type=boolean_string, help='train state and observation nerwork meantime', default=False)
    # -- gpu
    p.add('--device', type=int, help='-1: cpu; > -1: cuda device id', default=-1)
    # -- seed
    p.add('--manualSeed', type=int, help='manual seed')
    # -- logs
    p.add('--checkpoint_interval', type=int,help='check point interval', default=100)
    p.add('--log', type=boolean_string, help='log loss during training and save it', default=True)
    p.add('--log_relations', type=boolean_string, help='log relations', default=False)
    p.add('--run_min', type=boolean_string, help='training again and stop at the epoch with the minimal test loss', default=False)
    p.add('--show_fig', type=boolean_string, help='show fitting figure in the end', default=False)
    # parse
    opt = DotDict(vars(p.parse_args()))
    return opt

def train_by_opt(opt):
    """
    train stnn according to option
    """
    # path of output directory
    opt.outputdir = get_dir(opt.outputdir)
    if opt.xp_model:
        opt.xp = opt.xp + '-' + opt.model
    if opt.xp_time:
        opt.xp = opt.xp + "_" + get_time()
    # mode
    opt.mode = opt.mode if opt.mode in ('refine', 'discover') else None
    # log time
    opt.start = time_dir()
    start_st = datetime.datetime.now()
    opt.st = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    # cudnn
    if opt.device > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.device > -1:
        torch.cuda.manual_seed_all(opt.manualSeed)
    #######################################################################################################################
    # Data
    #######################################################################################################################
    # -- load data
    setup, (train_data, no_train_data, validation_data), relations = get_stnn_data(opt.datadir, opt.dataset, opt.nt_train, opt.khop, data_normalize=opt.data_normalize, relation_normalize=opt.relation_normalize, nt_val=opt.nt_val , relations_names=opt.relations, time_datas=opt.time_datas)
    test_data = no_train_data[opt.nt_val:]
    test_data = test_data.to(device)
    train_data = train_data.to(device)
    relations = relations.to(device)
    validation_data = validation_data.to(device)
    # -- load option
    for k, v in setup.items():
        opt[k] = v
    # --- get true validation
    true_validation = get_true_indicate(validation_data, opt, indicate_data=opt.indicate_data).to(device)
    # -- train inputs
    t_idx = torch.arange(opt.nt_train, out=torch.LongTensor()).unsqueeze(1).expand(opt.nt_train, opt.nx).contiguous()
    x_idx = torch.arange(opt.nx, out=torch.LongTensor()).expand_as(t_idx).contiguous()
    # decoder
    idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(device)
    nex_dec = idx_dec.size(1)
    #######################################################################################################################
    # Model
    #######################################################################################################################
    if opt.model == "classical":
        model = SaptioTemporalNN_classical(relations, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers,
                                           opt.nhid_de, opt.nlayers_de, opt.dropout_f, opt.dropout_d, opt.activation, opt.periode).to(device)
        idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
        nex_dyn = idx_dyn.size(1)
        params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
                {'params': model.dynamic.parameters()},
                {'params': model.decoder.parameters()}]
    elif opt.model == "A":
        model = SaptioTemporalNN_A(relations, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers, opt.nhid_de, opt.nlayers_de,
                                opt.dropout_f, opt.dropout_d, opt.activation, opt.periode).to(device)
        idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
        nex_dyn = idx_dyn.size(1)
        params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
                {'params': model.dynamic.parameters()},
                {'params': model.decoder.parameters()}]
    elif opt.model == "I":
        model = SaptioTemporalNN_I(relations, train_data, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers, opt.nhid_de, opt.nlayers_de,
                                   opt.dropout_f, opt.dropout_d, opt.activation, opt.periode, opt.nhid_in, opt.nlayers_in, opt.dropout_in).to(device)
        idx_dyn = torch.stack((t_idx[2:], x_idx[2:])).view(2, -1).to(device)
        nex_dyn = idx_dyn.size(1)
        params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
                {'params': model.dynamic.parameters()},
                {'params': model.decoder.parameters()},
                {'params': model.input_gate.parameters()}]
    #######################################################################################################################
    # Optimizer
    #######################################################################################################################
    # append parameters to be updated if refine or discover
    if opt.mode in ('refine', 'discover'):
        params.append({'params': model.rel_parameters(), 'weight_decay': 0.})
    # -- optimizer
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=opt.lr, weight_decay=opt.wd)
    elif opt.optimizer == 'Rmsprop':
        optimizer = optim.RMSprop(params, lr=opt.lr, weight_decay=opt.wd)
    elif opt.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(params, lr=opt.lr, weight_decay=opt.wd)
    # -- lr scheduler
    if opt.patience > 0:
        test_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience, factor=opt.es_wd_factor, verbose=True)
    #######################################################################################################################
    # Logs
    #######################################################################################################################
    logger = Logger(opt.outputdir, opt.xp, opt.checkpoint_interval)
    if opt.log_relations:
        relations_0 = model.get_relations()[:, 1:]
    #######################################################################################################################
    # Training
    #######################################################################################################################
    lr = opt.lr
    opt.min_val_sum = 1e8
    opt.min_val_sum_epoch = 0
    opt.min_val_rmse = 1e8
    opt.min_val_rmse_epoch = 0
    opt.min_test_sum = 1e8
    opt.min_test_sum_epoch = 0
    opt.min_test_rmse = 1e8
    opt.min_test_rmse_epoch = 0
    opt.epoch = 0
    pb = trange(opt.nepoch)
    for e in pb:
        opt.epoch += 1
        # ------------------------ Train ------------------------
        model.train()
        # --- decoder ---
        # idx_perm = torch.randperm(nex_dec).to(device)
        # batches_dec = idx_perm.split(opt.batch_size)
        batches_dec, v = sample(nex_dec, batch_size=opt.batch_size, sample_method=opt.sample_method)
        logs_train = defaultdict(float)
        if opt.train_meantime:
            for i, batch in enumerate(batches_dec):
                optimizer.zero_grad()
                # data
                input_t = idx_dec[0][batch]
                input_x = idx_dec[1][batch]
                x_target = train_data[input_t, input_x]
                # closure
                x_rec = model.dec_closure(input_t, input_x)
                # times weight
                x_target = x_target * v[i]
                x_rec = x_rec * v[i]
                mse_dec = F.mse_loss(x_rec, x_target)
                # log
                # logger.log('train_iter.mse_dec', mse_dec.item())
                logs_train['mse_dec'] += mse_dec.item() * len(batch)
                # dyn
                batch_dyn = []
                v_dyn = []
                for j in range(len(batch)):
                    if batch[j] < nex_dyn:
                        batch_dyn.append(batch[j])
                        v_dyn.append(v[i][j])
                # batch_dyn = [b for b in batch if b < nex_dyn]
                if batch_dyn == []:
                    mse_dec.backward()
                    optimizer.step()
                    continue
                else:
                    batch_dyn = torch.tensor(batch_dyn).to(device)
                    v_dyn = torch.tensor(v_dyn).view(-1, 1).to(device)
                # data
                input_t = idx_dyn[0][batch_dyn]
                input_x = idx_dyn[1][batch_dyn]
                # closure
                z_inf = model.factors[input_t, input_x]
                z_pred = model.dyn_closure(input_t - 1, input_x)
                # times weight
                z_inf = z_inf * v_dyn
                z_pred = z_pred * v_dyn
                # loss
                mse_dyn = z_pred.sub(z_inf).pow(2).mean()
                loss_dyn = mse_dyn * opt.lambd
                if opt.l2_z > 0:
                    loss_dyn += opt.l2_z * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()
                if opt.mode in('refine', 'discover') and opt.l1_rel > 0:
                    loss_dyn += opt.l1_rel * model.get_relations().abs().mean()
                # backward
                train_loss = loss_dyn + mse_dec
                train_loss.backward()
                # step
                optimizer.step()
                # log
                # logger.log('train_iter.mse_dyn', mse_dyn.item())
                logs_train['mse_dyn'] += mse_dyn.item() * len(batch_dyn)
                logs_train['loss_dyn'] += loss_dyn.item() * len(batch_dyn)
                # --- relation diffenerce
                if opt.log_relations:
                    relation_diff = model.get_relations()[:, 1:] - relations_0
                    for i, rel_name in enumerate(opt.relations_order):
                        logs_train[rel_name + '_max'] += relation_diff[:, i].max().item()
                        logs_train[rel_name + '_min'] += relation_diff[:, i].min().item()
                        logs_train[rel_name + '_mean'] += relation_diff[:, i].mean().item()
        else:
            # train respectively
            for i, batch in enumerate(batches_dec):
                optimizer.zero_grad()
                # data
                input_t = idx_dec[0][batch]
                input_x = idx_dec[1][batch]
                x_target = train_data[input_t, input_x]
                # closure
                x_rec = model.dec_closure(input_t, input_x)
                x_target = x_target * v[i]
                x_rec = x_rec * v[i]
                mse_dec = F.mse_loss(x_rec, x_target)
                # backward
                mse_dec.backward()
                # step
                optimizer.step()
                # log
                # logger.log('train_iter.mse_dec', mse_dec.item())
                logs_train['mse_dec'] += mse_dec.item() * len(batch)
                # === relation difference ===
                if opt.log_relations:
                    relation_diff = model.get_relations()[:, 1:] - relations_0
                    for i, rel_name in enumerate(opt.relations_order):
                        logs_train[rel_name + '_max'] += relation_diff[:, i].max().item()
                        logs_train[rel_name + '_min'] += relation_diff[:, i].min().item()
                        logs_train[rel_name + '_mean'] += relation_diff[:, i].mean().item()
            # --- dynamic ---
            # idx_perm = torch.randperm(nex_dyn).to(device)
            # batches_dyn = idx_perm.split(opt.batch_size)
            batches_dyn, v = sample(nex_dyn, batch_size=opt.batch_size)
            for i, batch in enumerate(batches_dyn):
                optimizer.zero_grad()
                # data
                input_t = idx_dyn[0][batch]
                input_x = idx_dyn[1][batch]
                # closure
                z_inf = model.factors[input_t, input_x]
                z_pred = model.dyn_closure(input_t - 1, input_x)
                # times weight
                z_inf = z_inf * v[i]
                z_pred = z_pred * v[i]
                # loss
                mse_dyn = z_pred.sub(z_inf).pow(2).mean()
                loss_dyn = mse_dyn * opt.lambd
                if opt.l2_z > 0:
                    loss_dyn += opt.l2_z * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()
                if opt.mode in('refine', 'discover') and opt.l1_rel > 0:
                    loss_dyn += opt.l1_rel * model.get_relations().abs().mean()
                # backward
                loss_dyn.backward()
                optimizer.step()
                # log
                # logger.log('train_iter.mse_dyn', mse_dyn.item())
                logs_train['mse_dyn'] += mse_dyn.item() * len(batch)
                logs_train['loss_dyn'] += loss_dyn.item() * len(batch)
                # === relation diffenerce ===
                if opt.log_relations:
                    relation_diff = model.get_relations()[:, 1:] - relations_0
                    for i, rel_name in enumerate(opt.relations_order):
                        logs_train[rel_name + '_max'] += relation_diff[:, i].max().item()
                        logs_train[rel_name + '_min'] += relation_diff[:, i].min().item()
                        logs_train[rel_name + '_mean'] += relation_diff[:, i].mean().item()
        # --- logs ---
        logs_train['mse_dec'] /= nex_dec
        logs_train['mse_dyn'] /= nex_dyn
        logs_train['loss_dyn'] /= nex_dyn
        logs_train['train_loss'] = logs_train['mse_dec'] + logs_train['loss_dyn']
        if opt.log:
            logger.log('train_epoch', logs_train)
            # checkpoint
            # logger.log('train_epoch.lr', lr)
            logger.checkpoint(model)
    # ------------------------ Test ------------------------
        if opt.test:
            model.eval()
            with torch.no_grad():
                x_pred, _ = model.generate(opt.nt - opt.nt_train)
                x_pred_test = x_pred[opt.nt_val:]
                opt.test_rmse_score = rmse(get_true_indicate(x_pred_test, opt, opt.indicate_data), get_true_indicate(test_data, opt, opt.indicate_data))
                opt.test_sum_score = rmse_sum_confirmed(get_true_indicate(x_pred_test, opt, opt.indicate_data), get_true_indicate(test_data, opt, opt.indicate_data))
            if opt.min_test_sum > opt.test_sum_score:
                opt.min_test_sum = opt.test_sum_score
                opt.min_test_sum_epoch = e
            if opt.min_test_rmse > opt.test_rmse_score:
                opt.min_test_rmse = opt.test_rmse_score
                opt.min_test_rmse_epoch = e
            if opt.log:
                logger.log('test_epoch.rmse', opt.test_rmse_score)
                logger.log('test_epoch.sum', opt.test_sum_score)
                pb.set_postfix(loss=logs_train['train_loss'], val=opt.val_sum_score, test=opt.test_sum_score)
        # ------------------------ validation ------------------------
        if opt.validate:
            model.eval()
            with torch.no_grad():
                x_pred, _ = model.generate(opt.nt_val)
                true_pred = get_true_indicate(x_pred, opt, opt.indicate_data)
                opt.val_rmse_score = rmse(true_pred, true_validation)
                opt.val_sum_score = rmse_sum_confirmed(true_pred, true_validation) 
            # pb.set_postfix(loss=logs_train['train_loss'], sum=opt.val_sum_score, rmse=opt.val_rmse_score)
            if opt.log:
                logger.log('validation_epoch.rmse', opt.val_rmse_score)
                logger.log('validation_epoch.sum', opt.val_sum_score)
            if opt.min_val_sum > opt.val_sum_score:
                opt.min_val_sum = opt.val_sum_score
                opt.min_val_sum_epoch = e
            if opt.min_val_rmse > opt.val_rmse_score:
                opt.min_val_rmse = opt.val_rmse_score
                opt.min_val_rmse_epoch = e
                # schedule lr
            if opt.patience > 0 and opt.val_sum_score < opt.es_val_bound and e > opt.es_start and opt.es_start > 0:
                test_lr_scheduler.step(opt.val_sum_score)
            lr = optimizer.param_groups[0]['lr']
            if opt.reduce_start > 0:
                if opt.reduce_start == e:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr * opt.reduce_factor
            if lr <= 1e-6:
                break
    opt.dec_loss = logs_train['mse_dec']
    opt.mse_dyn = logs_train['mse_dyn']
    opt.dyn_loss = logs_train['loss_dyn']
    opt.train_loss = logs_train['train_loss']
    opt.end = time_dir()
    end_st = datetime.datetime.now()
    opt.time = str(end_st - start_st)
    opt.et = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    with open(os.path.join(get_dir(opt.outputdir), opt.xp, 'config.json'), 'w') as f:
        json.dump(opt, f, sort_keys=True, indent=4)
    # if opt.log:
    logger.save(model)
    print(opt.xp)
    exp = result.Exp(opt.xp, get_dir(opt.outputdir))
    exp.get_total_fitting(save=True)
    exp.show_result(show=opt.show_fig)
    if opt.run_min:
        exp.run_min_exp()

def train():
    opt = get_opt()
    train_by_opt(opt)

if __name__ == "__main__":
    train()
