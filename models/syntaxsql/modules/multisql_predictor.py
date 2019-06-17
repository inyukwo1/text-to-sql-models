import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from commons.utils import run_lstm, seq_conditional_weighted_num, SIZE_CHECK
from models.syntaxsql.net_utils import to_batch_seq

class MultiSqlPredictor(nn.Module):
    '''Predict if the next token is (multi SQL key words):
        NONE, EXCEPT, INTERSECT, or UNION.'''
    def __init__(self, H_PARAM, embed_layer, bert=None):
        super(MultiSqlPredictor, self).__init__()
        self.N_word = H_PARAM['N_WORD']
        self.N_depth = H_PARAM['N_depth']
        self.N_h = H_PARAM['N_h']
        self.gpu = H_PARAM['gpu']
        self.use_hs = H_PARAM['use_hs']
        self.table_type = H_PARAM['table_type']

        self.acc = 0.0
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

        self.mkw_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_att = nn.Linear(encoded_num, self.N_h)
        self.hs_att = nn.Linear(self.N_h, self.N_h)
        self.multi_out_q = nn.Linear(encoded_num, self.N_h)
        self.multi_out_hs = nn.Linear(self.N_h, self.N_h)
        self.multi_out_c = nn.Linear(self.N_h, self.N_h)
        self.multi_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 1))

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        if self.gpu:
            self.cuda()

    def forward(self, input_data):
        q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len = input_data

        B = len(q_len)

        # q_enc: (B, max_q_len, hid_dim)
        # hs_enc: (B, max_hs_len, hid_dim)
        # mkw: (B, 4, hid_dim)
        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        mkw_enc, _ = run_lstm(self.mkw_lstm, mkw_emb_var, mkw_len)

        # Compute attention values between multi SQL key words and question tokens.
        q_weighted = seq_conditional_weighted_num(self.q_att, q_enc, q_len, mkw_enc)
        SIZE_CHECK(q_weighted, [B, 4, self.N_h])

        # Same as the above, compute SQL history embedding weighted by key words attentions
        hs_weighted = seq_conditional_weighted_num(self.hs_att, hs_enc, hs_len, mkw_enc)

        # Compute prediction scores=
        mulit_score = self.multi_out(self.multi_out_q(q_weighted) + int(self.use_hs)* self.multi_out_hs(hs_weighted) + self.multi_out_c(mkw_enc)).view(B, 4)

        return mulit_score

    def loss(self, score, truth):
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

        mkw_emb_var = self.embed_layer.gen_word_list_embedding(["none", "except", "intersect", "union"], len(batch))
        mkw_len = np.full(q_len.shape, 4, dtype=np.int64)

        input_data = (q_emb_var, q_len, hs_emb_var, hs_len, mkw_emb_var, mkw_len)
        gt_data = label

        return input_data, gt_data

    def save_model(self, save_dir):
        print('Saving model...')
        torch.save(self.state_dict(), os.path.join(save_dir, "multi_sql_models.dump"))