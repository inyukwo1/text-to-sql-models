import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from commons.utils import run_lstm, col_tab_name_encode, plain_conditional_weighted_num, SIZE_CHECK
from models.syntaxsql.net_utils import to_batch_seq, to_batch_tables


class AggPredictor(nn.Module):
    def __init__(self, H_PARAM, embed_layer, bert=None):
        super(AggPredictor, self).__init__()
        self.table_type = H_PARAM['table_type']
        self.N_word = H_PARAM['N_WORD']
        self.N_depth = H_PARAM['N_depth']
        self.N_h = H_PARAM['N_h']
        self.gpu = H_PARAM['gpu']
        self.use_hs = H_PARAM['use_hs']
        self.table_type = H_PARAM['table_type']

        self.acc_num = 3
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

        self.q_num_att = nn.Linear(encoded_num, self.N_h)
        self.hs_num_att = nn.Linear(self.N_h, self.N_h)
        self.agg_num_out_q = nn.Linear(encoded_num, self.N_h)
        self.agg_num_out_hs = nn.Linear(self.N_h, self.N_h)
        self.agg_num_out_c = nn.Linear(self.N_h, self.N_h)
        self.agg_num_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 4)) #for 0-3 agg num

        self.q_att = nn.Linear(encoded_num, self.N_h)
        self.hs_att = nn.Linear(self.N_h, self.N_h)
        self.agg_out_q = nn.Linear(encoded_num, self.N_h)
        self.agg_out_hs = nn.Linear(self.N_h, self.N_h)
        self.agg_out_c = nn.Linear(self.N_h, self.N_h)
        self.agg_out = nn.Sequential(nn.Tanh(), nn.Linear(self.N_h, 5)) #for 1-5 aggregators

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if self.gpu:
            self.cuda(0)


    def forward(self, input_data):
        q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, gt_col = input_data

        B = len(q_len)

        if self.use_bert:
            q_enc = self.q_bert(q_emb_var, q_len)
        else:
            q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        hs_enc, _ = run_lstm(self.hs_lstm, hs_emb_var, hs_len)
        col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)

        col_emb = []
        for b in range(B):
            col_emb.append(col_enc[b, gt_col[b]])
        col_emb = torch.stack(col_emb)

        # Predict agg number
        q_weighted_num = plain_conditional_weighted_num(self.q_num_att, q_enc, q_len, col_emb)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted_num = plain_conditional_weighted_num(self.hs_num_att, hs_enc, hs_len, col_emb)
        agg_num_score = self.agg_num_out(self.agg_num_out_q(q_weighted_num) + int(self.use_hs) * self.agg_num_out_hs(hs_weighted_num) + self.agg_num_out_c(col_emb))
        SIZE_CHECK(agg_num_score, [B, 4])

        # Predict aggregators
        q_weighted = plain_conditional_weighted_num(self.q_att, q_enc, q_len, col_emb)

        # Same as the above, compute SQL history embedding weighted by column attentions
        hs_weighted = plain_conditional_weighted_num(self.hs_att, hs_enc, hs_len, col_emb)
        # agg_score: (B, 5)
        agg_score = self.agg_out(self.agg_out_q(q_weighted) + int(self.use_hs)* self.agg_out_hs(hs_weighted) + self.agg_out_c(col_emb))

        score = (agg_num_score, agg_score)

        return score


    def loss(self, score, truth):
        loss = 0
        B = len(truth)
        agg_num_score, agg_score = score
        #loss for the column number
        truth_num = [len(t) for t in truth] # double check truth format and for test cases
        data = torch.from_numpy(np.array(truth_num))
        if self.gpu:
            data = data.cuda(0)
        truth_num_var = Variable(data)
        loss += self.CE(agg_num_score, truth_num_var)
        #loss for the key words
        T = len(agg_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth[b]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            data = data.cuda(0)
        truth_var = Variable(data)
        #loss += self.mlsml(agg_score, truth_var)
        #loss += self.bce_logit(agg_score, truth_var) # double check no sigmoid
        pred_prob = self.sigm(agg_score)
        bce_loss = -torch.mean( 3*(truth_var * \
                torch.log(pred_prob+1e-10)) + \
                (1-truth_var) * torch.log(1-pred_prob+1e-10) )
        loss += bce_loss

        return loss

    def evaluate(self, score, gt_data):
        return self.check_acc(score, gt_data)

    def check_acc(self, score, truth):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth)
        pred = []
        if self.gpu:
            agg_num_score, agg_score = [x.data.cpu().numpy() for x in score]
        else:
            agg_num_score, agg_score = [x.data.numpy() for x in score]

        for b in range(B):
            cur_pred = {}
            agg_num = np.argmax(agg_num_score[b]) #double check
            cur_pred['agg_num'] = agg_num
            cur_pred['agg'] = np.argsort(-agg_score[b])[:agg_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth)):
            agg_num, agg = p['agg_num'], p['agg']
            flag = True
            if agg_num != len(t): # double check truth format and for test cases
                num_err += 1
                flag = False
            if flag and set(agg) != set(t):
                err += 1
                flag = False
            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))

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
        torch.save(self.state_dict(), os.path.join(save_dir, "agg_models.dump"))
