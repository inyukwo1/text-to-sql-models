import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import dgl
from dgl import DGLGraph
import dgl.function as fn
from .net_utils import col_tab_name_encode


def table_to_dgl_graph(par_tab_nums, foreign_keys, col_enc, tab_enc):
    g = DGLGraph()
    col_id_offset = max(par_tab_nums) + 1
    g.add_nodes(len(par_tab_nums) + col_id_offset)
    # column id: max table num + 1 + original column num
    table_id_list = range(col_id_offset)
    col_id_list = range(len(par_tab_nums))
    g.add_edges(table_id_list, table_id_list)
    g.add_edges(col_id_list, col_id_list)
    edge_types = [0] * len(table_id_list) + [1] * len(col_id_list)
    table_children_src = []
    table_children_dst = []
    for idx, par_tab_num in enumerate(par_tab_nums):
        if par_tab_num != -1:
            table_children_src.append(par_tab_num)
            table_children_dst.append(idx + col_id_offset)
    g.add_edges(table_children_src, table_children_dst)
    g.add_edges(table_children_dst, table_children_src)
    edge_types += [2] * len(table_children_src) + [3] * len(table_children_dst)

    if foreign_keys:
        foreign_key_srcs, foreign_key_dsts = zip(*foreign_keys)
        foreign_key_srcs = list(map(lambda col_num: col_num + col_id_offset, foreign_key_srcs))
        foreign_key_dsts = list(map(lambda col_num: col_num + col_id_offset, foreign_key_dsts))
        g.add_edges(foreign_key_srcs, foreign_key_dsts)
        g.add_edges(foreign_key_dsts, foreign_key_srcs)
        edge_types += [4] * len(foreign_key_srcs) + [5] * len(foreign_key_dsts)

    edge_types = torch.from_numpy(np.array(edge_types))
    if torch.cuda.is_available():
        edge_types = edge_types.cuda()
    g.edata.update({'rel_type': edge_types})
    g.ndata['h'] = torch.cat((tab_enc[:col_id_offset], col_enc[:len(par_tab_nums)]))
    return g


def batch_table_to_dgl_batch_graph(batch_par_tab_nums, batch_foreign_keys, batch_col_enc, batch_tab_enc):
    graphs = []
    for b in range(len(batch_par_tab_nums)):
        g = table_to_dgl_graph(batch_par_tab_nums[b], batch_foreign_keys[b], batch_col_enc[b], batch_tab_enc[b])
        graphs.append(g)
    return dgl.batch(graphs)


class NGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=True, activation=F.relu):
        super(NGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                self.out_feat))
        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.bias)

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, bg):
        weight = self.weight

        def message_func(edges):
            w = weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias is not None:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        bg.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class SchemaAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super(SchemaAggregator, self).__init__()
        self.aggregator = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bg):
        graph_list = dgl.unbatch(bg)
        batch = []
        for graph in graph_list:
            ndata = graph.ndata['h']
            x = self.aggregator(ndata).sum(0)
            x = F.relu(x)
            batch.append(x)
        batch = torch.stack(batch)
        return batch


class SchemaEncoder(nn.Module):
    def __init__(self, hidden_dim, lower_dim):
        super(SchemaEncoder, self).__init__()
        self.col_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim//2,
                num_layers=1, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.lower = nn.Sequential(nn.Linear(hidden_dim, lower_dim), nn.ReLU())

        self.layer1 = NGCNLayer(lower_dim, lower_dim, 6)
        self.layer2 = NGCNLayer(lower_dim, lower_dim, 6)
        self.layer3 = NGCNLayer(lower_dim, lower_dim, 6)
        self.upper = nn.Sequential(nn.Linear(lower_dim, hidden_dim), nn.ReLU())
        self.skipper = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, batch_par_tab_nums, batch_foreign_keys,
                col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len):
        def add_padding(tensor, goal_size):
            tensor_len, hidden_dim = list(tensor.size())
            padding = Variable(torch.zeros(goal_size - tensor_len, hidden_dim))
            if torch.cuda.is_available():
                padding = padding.cuda()
            return torch.cat((tensor, padding), 0)
        origin_batch_col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)
        origin_batch_tab_enc, _ = col_tab_name_encode(table_emb_var, table_name_len, table_len, self.col_lstm)
        batch_col_enc = self.lower(origin_batch_col_enc)
        batch_tab_enc = self.lower(origin_batch_tab_enc)
        # (batch size, max seq len, hidden)
        b, max_col_len, hidden_dim = list(origin_batch_col_enc.size())
        b, max_tab_len, hidden_dim = list(origin_batch_tab_enc.size())
        bg_origin  = batch_table_to_dgl_batch_graph(batch_par_tab_nums, batch_foreign_keys, origin_batch_col_enc, origin_batch_tab_enc)
        bg = batch_table_to_dgl_batch_graph(batch_par_tab_nums, batch_foreign_keys, batch_col_enc, batch_tab_enc)
        origin_ndata = bg_origin.ndata['h']
        self.layer1(bg)
        self.layer2(bg)
        self.layer3(bg)
        graph_list = dgl.unbatch(bg)
        table_tensors = []
        col_tensors = []
        for b, graph in enumerate(graph_list):
            table_col_tensors = graph.ndata['h']
            par_tab_nums = batch_par_tab_nums[b]
            col_id_offset = max(par_tab_nums) + 1
            table_tensors.append(add_padding(table_col_tensors[:col_id_offset], max_tab_len))
            col_tensors.append(add_padding(table_col_tensors[col_id_offset:], max_col_len))
        table_tensors = torch.stack(table_tensors)
        col_tensors = torch.stack(col_tensors)
        table_tensors = self.upper(table_tensors) + self.skipper(origin_batch_tab_enc)
        col_tensors = self.upper(col_tensors) + self.skipper(origin_batch_col_enc)
        bg.ndata['h'] = self.upper(bg.ndata['h']) + self.skipper(origin_ndata)
        return table_tensors, col_tensors, bg
    #
    # def forward(self, batch_par_tab_nums, batch_foreign_keys,
    #             col_emb_var, col_name_len, col_len, table_emb_var, table_name_len, table_len):
    #     origin_batch_col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)
    #     origin_batch_tab_enc, _ = col_tab_name_encode(table_emb_var, table_name_len, table_len, self.col_lstm)
    #     bg = batch_table_to_dgl_batch_graph(batch_par_tab_nums, batch_foreign_keys, origin_batch_col_enc, origin_batch_tab_enc)
    #
    #     return origin_batch_tab_enc, origin_batch_col_enc, bg




