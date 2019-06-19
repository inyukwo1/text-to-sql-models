import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from commons.utils import run_lstm
from commons.embeddings.graph_utils import *
from commons.embeddings.bert_container import BertContainer
from commons.embeddings.word_embedding import WordEmbedding

class FromPredictor(nn.Module):
    def __init__(self, H_PARAM):
        super(FromPredictor, self).__init__()
        self.acc_num = 1
        self.acc = 0

        self.N_h = H_PARAM['N_h']
        self.N_depth = H_PARAM['N_depth']
        self.gpu = H_PARAM['gpu']
        self.use_hs = H_PARAM['use_hs']
        self.B_word = H_PARAM['B_WORD']
        self.N_word = H_PARAM['N_WORD']
        self.threshold = H_PARAM['threshold'] #0.5

        self.save_dir = H_PARAM['save_dir']
        self.bert = BertContainer(H_PARAM['bert_lr'])
        self.onefrom = H_PARAM['onefrom']
        self.encoded_num = H_PARAM['encoded_num'] #1024
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        self.embed_layer = WordEmbedding(H_PARAM['glove_path'].format(self.B_word, self.N_word),
                                         self.N_word, gpu=self.gpu, SQL_TOK=self.SQL_TOK, use_bert=True)

        self.hs_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.outer1 = nn.Sequential(nn.Linear(self.N_h + self.encoded_num, self.N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(self.N_h, 1))
        if self.onefrom:
            self.onefrom_vec = nn.Parameter(torch.zeros(self.N_h))
        if self.gpu:
            self.cuda()

    def load_model(self):
        print('Loading from model...')
        dev_type = 'cuda' if self.gpu else 'cpu'
        device = torch.device(dev_type)

        self.load_state_dict(torch.load(os.path.join(self.save_dir, "from_models.dump"), map_location=device))
        self.bert.main_bert.load_state_dict(torch.load(os.path.join(self.save_dir, "bert_from_models.dump"), map_location=device))
        self.bert.bert_param.load_state_dict(torch.load(os.path.join(self.save_dir, "bert_from_params.dump"), map_location=device))


    def save_model(self, acc):
        print('tot_err:{}'.format(acc))

        if acc > self.acc:
            self.acc = acc
            torch.save(self.state_dict(), os.path.join(self.save_dir, 'from_models.dump'))
            torch.save(self.bert.main_bert.state_dict(), os.path.join(self.save_dir + "bert_from_models.dump"))
            torch.save(self.bert.bert_param.state_dict(), os.path.join(self.save_dir + "bert_from_params.dump"))

    def train(self, mode=True):
        super().train(mode)
        if self.bert:
            self.bert.train()

    def eval(self):
        super().train(False)
        if self.bert:
            self.bert.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()
        if self.bert:
            self.bert.zero_grad()

    def step(self):
        self.optimizer.step()
        if self.bert:
            self.bert.step()

    def _forward(self, q_emb, q_len, q_q_len, hs_emb_var, hs_len, sep_embeddings):
        B = len(q_len)
        q_enc = self.bert.bert(q_emb, q_len, q_q_len, sep_embeddings)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        hs_enc = hs_enc[:, 0, :]
        if self.onefrom:
            hs_enc = self.onefrom_vec.view(1, -1).expand(B,  -1)

        q_enc = q_enc[:, 0, :]
        q_enc = torch.cat((q_enc, hs_enc), dim=1)
        x = self.outer1(q_enc)
        x = self.outer2(x).squeeze(1)
        return x

    def forward(self, input_data, single_forward=False):
        # Parse Input
        q_embs, q_lens, q_q_lens, hs_emb_var, hs_len, sep_embedding_lists = input_data

        max_size = 4

        if self.training or single_forward:
            scores = self._forward(*input_data)
        else:
            scores = []
            for i in range(len(q_embs)):
                length = len(q_embs[i])
                one_batch_score = []
                for j in range(0, length, max_size):
                    score = self._forward(q_embs[i][j:min(length,j+max_size)], q_lens[i][j:min(length,j+max_size)],
                        q_q_lens[i][j:min(length,j+max_size)], hs_emb_var, hs_len, sep_embedding_lists[i][j:min(length,j+max_size)])
                    one_batch_score.append(score.data.cpu().numpy())
                one_batch_score = np.concatenate(one_batch_score)
                scores.append(one_batch_score)

        return scores

    def loss(self, score, labels):
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def check_acc(self, scores, gt_data, batch=None, log=False):
        # Parse Input
        graph, table_graph_list, full_graph_lists, foreign_keys, primary_keys = gt_data

        graph_correct_list = []
        selected_tbls = []
        full_graphs = []
        for i in range(len(scores)):
            selected_graph = table_graph_list[i][np.argmax(scores[i])]
            ans_graph = {}
            for t in graph[i]:
                ans_graph[int(t)] = graph[i][t]
            graph_correct = graph_checker(selected_graph, ans_graph, foreign_keys[i], primary_keys[i])
            if log and not graph_correct:
                print("==========================================")
                print("question: {}".format(batch[i]["question"]))
                print("sql: {}".format(batch[i]["query"]))
                for table_num, table_name in enumerate(batch[i]["tbl"]):
                    print("Table: {}".format(table_name))
                    for col_num, (par_num, col_name) in enumerate(batch[i]["column"]):
                        if par_num == table_num:
                            print("  {}: {}".format(col_num, col_name))
                print("ans: {}".format(ans_graph))
                print("selected: {}".format(selected_graph))
                print("cands: ")
                for cand_idx in range(len(scores[i])):
                    print(table_graph_list[i][cand_idx])
                    print("score: {}".format(scores[i][cand_idx]))
                    print("%%%")
                print(graph_correct)
            graph_correct_list.append(graph_correct)
            selected_tbls.append(selected_graph.keys())
            full_graphs.append(full_graph_lists[i][np.argmax(scores[i])])

        return graph_correct_list, selected_tbls, full_graphs

    def evaluate(self, score, gt_data, batch=None, log=False):
        return self.check_acc(score, gt_data, batch, log)[0].count(True)

    def preprocess(self, batch):
        q_seq = []
        history = []
        labels = []
        tabs = []
        cols = []
        f_keys = []
        p_keys = []

        # For acc
        q_embs = []
        q_lens = []
        q_q_lens = []
        table_graph_lists = []
        full_graph_lists = []
        sep_embedding_lists = []

        if self.training:
            for item in batch:
                history.append(item['history'] if self.use_hs else ['root', 'none'])
                labels.append(item['join_table_dict'])
                f_keys.append(item['foreign_keys'])
                p_keys.append(item['primary_keys'])
                q_seq.append(item['question_toks'])
                tabs.append(item['tbl'])    # Original being used
                cols.append(item['column']) # Original being used
            q_embs, q_lens, q_q_lens, labels, sep_embedding_lists = self.embed_layer.gen_bert_batch_with_table(q_seq, tabs, cols, f_keys, p_keys, labels)
        else:
            for item in batch:
                history.append(item['history'] if self.use_hs else ['root', 'none'])
                labels.append(item['join_table_dict'])
                f_keys.append(item['foreign_keys'])
                p_keys.append(item['primary_keys'])
                q_emb, q_len, q_q_len, table_graph_list, full_graph_list, sep_embeddings = self.embed_layer.gen_bert_for_eval(
                    item['question_toks'], item['tbl'], item['column'], item['foreign_keys'], item['primary_keys'])
                q_embs.append(q_emb)
                q_lens.append(q_len)
                q_q_lens.append(q_q_len)
                table_graph_lists.append(table_graph_list)
                full_graph_lists.append(full_graph_list)
                sep_embedding_lists.append(sep_embeddings)

        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

        input_data = q_embs, q_lens, q_q_lens, hs_emb_var, hs_len, sep_embedding_lists
        gt_data = labels if self.training else (labels, table_graph_lists, full_graph_lists, f_keys, p_keys)

        return input_data, gt_data