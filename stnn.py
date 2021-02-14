 #-*-coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from module import MLP, MLP_tanh, MLP_sigmoid
from utils import identity, copy_nonzero_weights

class SaptioTemporalNN_classical(nn.Module):
    def __init__(self,
                 relations,
                 nx,
                 nt,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 nhid_de=0,
                 nlayers_de=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1,
                 auto_encoder=False):
        super(SaptioTemporalNN_classical, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        self.auto_encoder = auto_encoder
        # kernel
        # self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'default' or mode == 'refine':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), torch.ones(
                    nx, relations.size(1), nx).to(device)), 1)
        self.nr = self.relations.size(1)  # number of relations, here nr = 2
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.randn(nt, nx, nz))
        if activation == 'tanh':
            self.dynamic = MLP_tanh(nz, nhid, nz, nlayers, dropout_d)
            self.decoder = MLP_tanh(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = torch.tanh
            if self.auto_encoder:
                self.encoder = MLP_tanh(nd, nhid, nz, nlayers, dropout_d)
        elif activation == 'sigmoid':
            self.dynamic = MLP_sigmoid(nz, nhid, nz, nlayers, dropout_d)
            self.decoder = MLP_sigmoid(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = identity
            if self.auto_encoder:
                self.encoder = MLP_sigmoid(nd, nhid, nz, nlayers, dropout_d)
        else:
            self.dynamic = MLP(nz, nhid, nz, nlayers, dropout_d)
            self.decoder = MLP(nz, nhid_de, nd, nlayers_de, dropout_d)
            # self.activation = torch.relu
            self.activation = torch.identity
            if self.auto_encoder:
                self.encoder = MLP(nd, nhid, nz, nlayers, dropout_d)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).bool()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, self.nr - 1, nx))
        # init
        self._init_factors(periode)
        if self.mode == 'refine':
            self.rel_weights.data = copy_nonzero_weights(relations)
        elif self.mode == 'discover':
            self.rel_weights.data = relations.data

    def _init_factors(self, periode):
        # if use auto-encoder to gain factors and decoder:
        if self.auto_encoder:
            # train encoder and decoder
            self.train_auto_encoder()

        else:
            initrange = 1.0
            if periode >= self.nt:
                self.factors.data.uniform_(-initrange, initrange)
            else:
                timesteps = torch.arange(self.factors.size(0)).long()
                for t in range(periode):
                    idx = timesteps % periode == t
                    idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                    init = torch.Tensor(self.nx, self.nz).uniform_(
                        -initrange, initrange).repeat(idx.sum().item(), 1, 1)
                self.factors.data.masked_scatter_(idx_data, init.view(-1))
            # if self.mode == 'refine':
            #     self.rel_weights.data.fill_(0.5)
            # elif self.mode == 'discover':
            #     self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None or self.mode == 'default':
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(
                    self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1,
                                                   self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        '''
        z : (nx, nz)
        '''
        z_context = self.get_relations().matmul(z).sum(1)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        z_context = rels[x_idx].matmul(z_input).sum(1)
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps, start = -1):
        self.eval()
        z = self.factors[start] #(nx, nz)
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights

class SaptioTemporalNN_A(nn.Module):
    def __init__(self,
                 relations,
                 nx,
                 nt,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 nhid_de=0,
                 nlayers_de=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1):
        super(SaptioTemporalNN_A, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        # self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'default' or mode == 'refine':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), torch.ones(
                    nx, relations.size(1), nx).to(device)), 1)
        self.nr = self.relations.size(1) # number of relations, here nr = 2
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.randn(nt, nx, nz))
        if activation == 'tanh':
            self.dynamic = MLP_tanh(nz * self.nr, nhid, nz, nlayers, dropout_d)
            self.decoder = MLP_tanh(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.dynamic = MLP_sigmoid(nz * self.nr, nhid, nz, nlayers, dropout_d)
            self.decoder = MLP_sigmoid(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = identity
        else:
            self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
            self.decoder = MLP(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = torch.relu
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).bool()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, self.nr - 1, nx))
        # init
        self._init_factors(periode)
        if self.mode == 'refine':
            self.rel_weights.data = copy_nonzero_weights(relations)
        elif self.mode == 'discover':
            self.rel_weights.data = relations.data

    def _init_factors(self, periode):
        initrange = 1.0
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(
                    -initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        # if self.mode == 'refine':
        #     self.rel_weights.data.fill_(0.5)
        # elif self.mode == 'discover':
        #     self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None or self.mode == 'default':
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(
                    self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1,
                                                   self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        '''
        z : (nx, nz)
        '''
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        z_context = rels[x_idx].matmul(z_input).view(-1,
                                                     self.nr * self.nz)
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps, start = -1):
        self.eval()
        z = self.factors[start] #(nx, nz)
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights


class SaptioTemporalNN_I(nn.Module):
    def __init__(self,
                 relations,
                 x_input,
                 nx,
                 nt,
                 nd,
                 nz,
                 mode=None,
                 nhid=0,
                 nlayers=1,
                 nhid_de=0,
                 nlayers_de=1,
                 dropout_f=0.,
                 dropout_d=0.,
                 activation='tanh',
                 periode=1,
                 nhid_in=0,
                 nlayers_in=1,
                 dropout_in=0):
        super(SaptioTemporalNN_I, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        # self.activation = torch.tanh if activation == 'tanh' else identity if activation == 'identity' else None
        device = relations.device
        if mode is None or mode == 'refine' or mode == 'default':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat(
                (torch.eye(nx).to(device).unsqueeze(1), torch.ones(
                    nx, relations.size(1), nx).to(device)), 1)
        self.nr = self.relations.size(1) # number of relations, here nr = 2
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.randn(nt, nx, nz))
        self.input_data = x_input
        if x_input.size() != (nt, nx, nd):
            print('input size not match')
        if activation == 'tanh':
            self.dynamic = MLP_tanh(nz * (self.nr + 1), nhid, nz, nlayers, dropout_d)
            self.decoder = MLP_tanh(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.dynamic = MLP_sigmoid(nz * (self.nr + 1), nhid, nz, nlayers, dropout_d)
            self.decoder = MLP_sigmoid(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = identity
        else:
            self.dynamic = MLP(nz * (self.nr + 1), nhid, nz, nlayers, dropout_d)
            self.decoder = MLP(nz, nhid_de, nd, nlayers_de, dropout_d)
            self.activation = torch.relu
        self.input_gate = MLP(nd, nhid_in, nz, nlayers_in, dropout_in)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).bool()
            self.rel_weights = nn.Parameter(
                torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(nx, relations.size(1), nx))
        # init
        self._init_factors(periode)
        if self.mode == 'refine':
            self.rel_weights.data = copy_nonzero_weights(relations)
        elif self.mode == 'discover':
            self.rel_weights.data = relations.data

    def _init_factors(self, periode):
        initrange = 1.0
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(
                    -initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
        # if self.mode == 'refine':
        #     self.rel_weights.data.fill_(0.5)
        # elif self.mode == 'discover':
        #     self.rel_weights.data.fill_(1 / self.nx)

    def get_relations(self):
        if self.mode is None or self.mode == 'default':
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(
                    self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1,
                                                   self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)

    def update_z(self, z, x):
        '''
        z : (nx, nz)
        x : (nx, nd)
        '''
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        x_into_state = self.input_gate(x).view(-1, self.nz)
        z_next = self.dynamic(torch.cat([z_context, x_into_state], dim=1))
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.drop(self.factors[t_idx, x_idx])
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.drop(self.factors[t_idx])
        x_into_state = self.input_gate(self.input_data[t_idx-1, x_idx]).view(-1, self.nz)
        z_context = rels[x_idx].matmul(z_input).view(-1,
                                                     self.nr*self.nz)
        z_gen = self.dynamic(torch.cat([z_context, x_into_state], dim=1))
        return self.activation(z_gen)

    def generate(self, nsteps, start=-1):
        # z = self.factors[start] #(nx, nz)
        # x = self.input_data[start]
        # z_gen = []
        # x_gen = []
        # for t in range(nsteps):
        #     z = self.update_z(z, x)
        #     z_gen.append(z)
        #     x = self.decode_z(z)
        #     x_gen.append(x)
        # z_gen = torch.stack(z_gen)
        # x_gen = torch.stack(x_gen)
        self.eval()
        states = self.factors[start:]
        inputs = self.input_data[start-1:]
        for i in range(nsteps):
            new_state = self.update_z(states[-1].unsqueeze(0), inputs[-2])
            new_input = self.decode_z(new_state)
            states = torch.cat((states, new_state.unsqueeze(0)))
            inputs = torch.cat((inputs, new_input.unsqueeze(0)))
        return inputs[2:], states[1:]

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights
