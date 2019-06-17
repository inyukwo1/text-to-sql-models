import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from commons.utils import run_lstm
from models.syntaxsql.net_utils import to_batch_seq


class AndOrPredictor(nn.Module):
    def __init__(self, H_PARAM, embed_layer, bert=None):
        super(AndOrPredictor, self).__init__()
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

        self.q_att = nn.Linear(encoded_num, self.N_h)
        self.hs_att = nn.Linear(self.N_h, self.N_h)
        self.ao_out_q = nn.Linear(encoded_num, self.N_h)
        self.ao_out_hs = nn.Linear(self.N_h, self.N_h)
        self.ao_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 2)) #for and/or

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if self.gpu:
            self.cuda()

    def forward(self, input_data):
        q_emb_var, q_len, hs_emb_var, hs_len = input_data

        max_q_len = max(q_len)
        max_hs_len = max(hs_len)
        B = len(q_len)

        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)

        att_np_q = np.ones((B, max_q_len))
        att_val_q = torch.from_numpy(att_np_q).float()
        if self.gpu:
            att_val_q = att_val_q.cuda()
        att_val_q = Variable(att_val_q)
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_q[idx, num:] = -100
        att_prob_q = self.softmax(att_val_q)
        q_weighted = (q_enc * att_prob_q.unsqueeze(2)).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        att_np_h = np.ones((B, max_hs_len))
        att_val_h = torch.from_numpy(att_np_h).float()
        if self.gpu:
            att_val_h = att_val_h.cuda()
        att_val_h = Variable(att_val_h)
        for idx, num in enumerate(hs_len):
            if num < max_hs_len:
                att_val_h[idx, num:] = -100
        att_prob_h = self.softmax(att_val_h)
        hs_weighted = (hs_enc * att_prob_h.unsqueeze(2)).sum(1)
        # ao_score: (B, 2)
        ao_score = self.ao_out(self.ao_out_q(q_weighted) + int(self.use_hs)* self.ao_out_hs(hs_weighted))

        return ao_score

    def loss(self, score, truth):
        loss = 0
        data = torch.from_numpy(np.array(truth))
        truth_var = Variable(data.cuda())
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

        input_data = (q_emb_var, q_len, hs_emb_var, hs_len)
        gt_data = label

        return input_data, gt_data

    def save_model(self, save_dir):
        print('Saving model...')
        torch.save(self.state_dict(), os.path.join(save_dir, "andor_models.dump"))