import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from commons.utils import run_lstm, col_tab_name_encode, plain_conditional_weighted_num
from models.syntaxsql.net_utils import to_batch_tables, to_batch_seq

class DesAscLimitPredictor(nn.Module):
    def __init__(self, H_PARAM, embed_layer, bert=None):
        super(DesAscLimitPredictor, self).__init__()
        self.N_word = H_PARAM['N_WORD']
        self.N_depth = H_PARAM['N_depth']
        self.N_h = H_PARAM['N_h']
        self.gpu = H_PARAM['gpu']
        self.use_hs = H_PARAM['use_hs']
        self.table_type = H_PARAM['table_type']

        self.acc_num = 1
        self.embed_layer = embed_layer

        self.use_bert = True if bert else False
        if bert:
            self.q_bert = bert
            encoded_num = 768
        else:
            self.q_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
            encoded_num = self.N_h

        self.hs_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(encoded_num, self.N_h)
        self.hs_att = nn.Linear(self.N_h, self.N_h)
        self.dat_out_q = nn.Linear(encoded_num, self.N_h)
        self.dat_out_hs = nn.Linear(self.N_h, self.N_h)
        self.dat_out_c = nn.Linear(self.N_h, self.N_h)
        self.dat_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 4)) #for 4 desc/asc limit/none combinations

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if self.gpu:
            self.cuda()

    def forward(self, input_data):
        q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col = input_data

        B = len(q_len)

        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        # get target/predicted column's embedding
        # col_emb: (B, hid_dim)
        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb) # [B, dim]
        q_weighted = plain_conditional_weighted_num(self.q_att, q_enc, q_len, col_emb)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted = plain_conditional_weighted_num(self.hs_att, hs_enc, hs_len, col_emb)
        # dat_score: (B, 4)
        dat_score = self.dat_out(self.dat_out_q(q_weighted) + int(self.use_hs)* self.dat_out_hs(hs_weighted) + self.dat_out_c(col_emb))

        return dat_score

    def loss(self, score, truth):
        loss = 0
        data = torch.from_numpy(np.array(truth))
        if self.gpu:
            data = data.cuda()
        truth_var = Variable(data)
        loss = self.CE(score, truth_var)

        return loss

    def evaluate(self, score, gt_data):
        return self.check_acc(score, gt_data)

    def check_acc(self, score, truth):
        err = 0
        B = len(score)
        pred = []
        for b in range(B):
            if self.gpu:
                argmax_score = np.argmax(score[b].data.cpu().numpy())
            else:
                argmax_score = np.argmax(score[b].data.numpy())
            pred.append(argmax_score)
        for b, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                err += 1

        return err

    def preprocess(self, batch):
        q_seq, history, label = to_batch_seq(batch)
        q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

        col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(batch, self.table_type)
        col_emb_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col_seq)
        gt_col = np.zeros(q_len.shape, dtype=np.int64)

        index = 0
        for item in batch:
            gt_col[index] = item["gt_col"]
            index += 1

        input_data = (q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col)
        gt_data = label

        return input_data, gt_data

    def save_model(self, save_dir):
        print('Saving model...')
        torch.save(self.state_dict(), os.path.join(save_dir, "desc_asc_models.dump"))