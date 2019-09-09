import os
import torch
import numpy as np
import traceback
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable
from models.typesql.modules.sel_predict import SelPredictor
from models.typesql.modules.cond_predict import CondPredictor
from models.typesql.modules.group_predict import GroupPredictor
from models.typesql.modules.order_predict import OrderPredictor
from models.frompredictor.from_predictor import FromPredictor
from commons.embeddings.word_embedding import WordEmbedding
from commons.embeddings.bert_container import BertContainer
from copy import deepcopy

AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
WHERE_OPS = ['not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists']
VALUE = '''"VALUE"'''
DESC_ASC_LIMIT = ["asc limit 1", "desc limit 1", "asc", "desc"]

class TypeSQL(nn.Module):
    def __init__(self, H_PARAM):
        super(TypeSQL, self).__init__()
        self.acc_num = 6
        self.acc = [0.0] * 5

        self.train_emb = H_PARAM['trainable_emb']
        self.save_dir = H_PARAM['save_dir']
        self.gpu = H_PARAM['gpu']
        self.N_word = H_PARAM['N_WORD']
        self.B_word = H_PARAM['B_WORD']
        self.N_h = H_PARAM['N_h']
        self.N_depth = H_PARAM['N_depth']
        self.use_from = H_PARAM['use_from']

        self.bert = BertContainer(H_PARAM['bert_lr']) if H_PARAM['bert'] else None

        self.max_col_num = H_PARAM['max_col_num'] #45
        self.max_tok_num = H_PARAM['max_tok_num'] #200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        self.embed_layer = WordEmbedding(H_PARAM['glove_path'].format(self.B_word, self.N_word), self.N_word, self.gpu,
                self.SQL_TOK, use_bert=(self.bert is not None), trainable=self.train_emb, use_small=H_PARAM['use_small'])

        #Predict select clause
        self.sel_pred = SelPredictor(self.N_word, self.N_h, self.N_depth, self.gpu)
        #Predict from clause
        self.from_pred = FromPredictor(H_PARAM) if H_PARAM['use_from'] else None
        #Predict where condition
        self.cond_pred = CondPredictor(self.N_word, self.N_h, self.N_depth, self.gpu)
        #Predict group by
        self.group_pred = GroupPredictor(self.N_word, self.N_h, self.N_depth, self.gpu)
        #Predict order by
        self.order_pred = OrderPredictor(self.N_word, self.N_h, self.N_depth, self.gpu)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if self.gpu:
            self.cuda()

    def _find_shortest_path(self, start, end, graph):
        stack = [[start, []]]
        visited = set()
        while len(stack) > 0:
            ele, history = stack.pop()
            if ele == end:
                return history
            for node in graph[ele]:
                if node[0] not in visited:
                    stack.append((node[0], history + [(node[0], node[1])]))
                    visited.add(node[0])
        # print("Could not find a path between table {} and table {}".format(start, end))

    def _gen_from(self, candidate_tables, schema):
        if len(candidate_tables) <= 1:
            if len(candidate_tables) == 1:
                ret = "from {}".format(schema["table_names_original"][list(candidate_tables)[0]])
            else:
                ret = "from {}".format(schema["table_names_original"][0])
            # TODO: temporarily settings for select count(*)
            return {}, ret
        # print("candidate:{}".format(candidate_tables))
        table_alias_dict = {}
        uf_dict = {}
        for t in candidate_tables:
            uf_dict[t] = -1
        idx = 1
        graph = defaultdict(list)
        for acol, bcol in schema["foreign_keys"]:
            t1 = schema["column_names"][acol][0]
            t2 = schema["column_names"][bcol][0]
            graph[t1].append((t2, (acol, bcol)))
            graph[t2].append((t1, (bcol, acol)))
        candidate_tables = list(candidate_tables)
        start = candidate_tables[0]
        table_alias_dict[start] = idx
        idx += 1
        ret = "from {} as T1".format(schema["table_names_original"][start])
        try:
            for end in candidate_tables[1:]:
                if end in table_alias_dict:
                    continue
                path = self._find_shortest_path(start, end, graph)
                prev_table = start
                if not path:
                    table_alias_dict[end] = idx
                    idx += 1
                    ret = "{} join {} as T{}".format(ret, schema["table_names_original"][end],
                                                     table_alias_dict[end],
                                                     )
                    continue
                for node, (acol, bcol) in path:
                    if node in table_alias_dict:
                        prev_table = node
                        continue
                    table_alias_dict[node] = idx
                    idx += 1
                    ret = "{} join {} as T{} on T{}.{} = T{}.{}".format(ret, schema["table_names_original"][node],
                                                                        table_alias_dict[node],
                                                                        table_alias_dict[prev_table],
                                                                        schema["column_names_original"][acol][1],
                                                                        table_alias_dict[node],
                                                                        schema["column_names_original"][bcol][1])
                    prev_table = node
        except:
            traceback.print_exc()
            #print("db:{}".format(schema["db_id"]))
            # print(table["db_id"])
            return table_alias_dict, ret
        return table_alias_dict, ret

    def _gen_from_graph(self, from_graph, table):
        if len(from_graph) == 1:
            ret = "from {}".format(table["table_names_original"][list(from_graph)[0]])
            return {}, ret
        copied_table = deepcopy(from_graph)
        #print(copied_table)
        start = list(from_graph.keys())[0]
        table_alias_dict = {start: 1}
        ret = "from {} as T1".format(table["table_names_original"][start])
        while len(table_alias_dict) < len(from_graph):
            for tab_num in table_alias_dict:
                added = False
                for edge in from_graph[tab_num]:
                    if edge[2] in table_alias_dict:
                        from_graph[tab_num].remove(edge)
                        break
                    table_alias_dict[edge[2]] = len(table_alias_dict) + 1
                    added = True
                    ret += " join {} as T{} on T{}.{} = T{}.{}".format(table["table_names_original"][edge[2]],
                                                                       len(table_alias_dict),
                                                                       table_alias_dict[tab_num],
                                                                       table["column_names_original"][edge[0]][1],
                                                                       table_alias_dict[edge[2]],
                                                                       table["column_names_original"][edge[1]][1])
                    from_graph[tab_num].remove(edge)
                    break
                if added:
                    break
        return table_alias_dict, ret

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

    def load_model(self):
        dev_type = 'cuda' if self.gpu else 'cpu'
        device = torch.device(dev_type)

        # Load models
        print("Loading sel model...")
        self.sel_pred.load_state_dict(torch.load(os.path.join(self.save_dir, "sel_randn_models.dump"), map_location=device))
        print("Loading cond model...")
        self.cond_pred.load_state_dict(torch.load(os.path.join(self.save_dir, "cond_randn_models.dump"), map_location=device))
        print("Loading group model...")
        self.group_pred.load_state_dict(torch.load(os.path.join(self.save_dir, "group_randn_models.dump"), map_location=device))
        print("Loading order model...")
        self.order_pred.load_state_dict(torch.load(os.path.join(self.save_dir, "order_randn_models.dump"), map_location=device))
        if self.use_from:
            print("Loading from model...")
            self.from_pred.load_state_dict(torch.load(os.path.join(self.save_dir, "from_models.dump"), map_location=device))
            self.from_pred.bert.main_bert.load_state_dict(torch.load(os.path.join(self.save_dir, "bert_from_models.dump"), map_location=device))
            self.from_pred.bert.bert_param.load_state_dict(torch.load(os.path.join(self.save_dir, "bert_from_params.dump"), map_location=device))

    def save_model(self, acc):
        print('tot_acc:{} sel_acc:{} cond_acc:{} gby_acc:{} ody_acc:{} from_acc:{}'.format(*acc))

        if acc[1] > self.acc[0]:
            self.acc[0] = acc[1]
            print("Saving sel model...")
            torch.save(self.sel_pred.state_dict(), os.path.join(self.save_dir, 'sel_models.dump'))
        if acc[2] > self.acc[1]:
            self.acc[1] = acc[2]
            print("Saving cond model...")
            torch.save(self.cond_pred.state_dict(), os.path.join(self.save_dir, 'cond_models.dump'))
        if acc[3] > self.acc[2]:
            self.acc[2] = acc[3]
            print("Saving group model...")
            torch.save(self.group_pred.state_dict(), os.path.join(self.save_dir, 'group_models.dump'))
        if acc[4] > self.acc[3]:
            self.acc[3] = acc[4]
            print("Saving order model...")
            torch.save(self.order_pred.state_dict(), os.path.join(self.save_dir, 'order_models.dump'))
        if acc[5] > self.acc[4]:
            print("Saving from model...")
            self.acc[4] = acc[5]
            if self.from_pred:
                torch.save(self.from_pred.state_dict(), os.path.join(self.save_dir, 'from_models.dump'))
                torch.save(self.bert.main_bert.state_dict(), os.path.join(self.save_dir + "bert_from_models.dump"))
                torch.save(self.bert.bert_param.state_dict(), os.path.join(self.save_dir + "bert_from_params.dump"))


    def forward(self, input_data):
        # input
        q_seq, col_num, col_org_seq, tbl_ans, from_col_seq, history, q_emb, q_len, q_q_len, sep_embeddings, cols, q_type, selected_tbls = input_data

        B = len(q_seq)

        max_col_len = max(col_num)
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q_seq, is_list=True, is_q=True) #
        x_type_emb_var, x_type_len = self.embed_layer.gen_x_batch(q_type, is_list=True, is_q=True)  #
        col_inp_var, col_len = self.embed_layer.gen_x_batch(cols, is_list=True)

        # For from predictor
        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)
        from_score = self.from_pred([q_emb, q_len, q_q_len, hs_emb_var, hs_len, sep_embeddings], single_forward=False) if self.use_from else None

        if self.use_from:
            # Create bitmap
            col_bitmap = torch.zeros((B, max_col_len))
            col_bitmap[:, 0] = 1
            target_tbl_info = tbl_ans if self.training else selected_tbls
            for idx in range(B):
                for idx2, item in enumerate(col_org_seq[idx]):
                    if not idx2 or item[0] in target_tbl_info[idx]:
                        col_bitmap[idx][idx2] = 1.0
        else:
            col_bitmap = None

        sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_emb_var, col_bitmap)
        cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_emb_var, col_bitmap)
        group_score = self.group_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_emb_var, col_bitmap)
        order_score = self.order_pred(x_emb_var, x_len, col_inp_var, col_len, x_type_emb_var, col_bitmap)

        return sel_score, from_score, cond_score, group_score, order_score

    def loss(self, score, gt_data):
        truth_num, label = gt_data[0:2]

        sel_score, from_score, cond_score, group_score, order_score = score

        sel_num_score, sel_col_score, agg_num_score, agg_op_score = sel_score
        cond_num_score, cond_col_score, cond_op_score = cond_score
        gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = group_score
        ody_num_score, ody_col_score, ody_agg_score, ody_par_score = order_score

        B = len(truth_num)
        loss = 0
        #----------loss for from_pred -------------#
        if self.use_from:
            from_loss = self.from_pred.loss(from_score, label)
            loss += from_loss

        #----------loss for sel_pred -------------#

        # loss for sel agg # and sel agg
        for b in range(len(truth_num)):
            curr_col = truth_num[b][1][0]
            curr_col_num_aggs = 0
            gt_aggs_num = []
            for i, col in enumerate(truth_num[b][1]):
                if col != curr_col:
                    gt_aggs_num.append(curr_col_num_aggs)
                    curr_col = col
                    curr_col_num_aggs = 0
                if truth_num[b][0][i] != 0:
                    curr_col_num_aggs += 1
            gt_aggs_num.append(curr_col_num_aggs)
            # print gt_aggs_num
            data = torch.from_numpy(np.array(gt_aggs_num)) #supposed to be gt # of aggs
            if self.gpu:
                agg_num_truth_var = Variable(data.cuda())
            else:
                agg_num_truth_var = Variable(data)
            agg_num_pred = agg_num_score[b, :truth_num[b][5]] # supposed to be gt # of select columns
            loss += (self.CE(agg_num_pred, agg_num_truth_var) \
                    / len(truth_num))
            # loss for sel agg prediction
            T = 6 #num agg ops
            truth_prob = np.zeros((truth_num[b][5], T), dtype=np.float32)
            gt_agg_by_sel = []
            curr_sel_aggs = []
            curr_col = truth_num[b][1][0]
            col_counter = 0
            for i, col in enumerate(truth_num[b][1]):
                if col != curr_col:
                    gt_agg_by_sel.append(curr_sel_aggs)
                    curr_col = col
                    col_counter += 1
                    curr_sel_aggs = [truth_num[b][0][i]]
                    truth_prob[col_counter][curr_sel_aggs] = 1
                else:
                    curr_sel_aggs.append(truth_num[b][0][i])
                    truth_prob[col_counter][curr_sel_aggs] = 1
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                agg_op_truth_var = Variable(data.cuda())
            else:
                agg_op_truth_var = Variable(data)
            agg_op_prob = self.sigm(agg_op_score[b, :truth_num[b][5]])
            agg_bce_loss = -torch.mean( 3*(agg_op_truth_var * \
                    torch.log(agg_op_prob+1e-10)) + \
                    (1-agg_op_truth_var) * torch.log(1-agg_op_prob+1e-10) )
            loss += agg_bce_loss / len(truth_num)

        #Evaluate the number of select columns
        sel_num_truth = [*map(lambda x: x[5]-1, truth_num)] #might need to be the length of the set of columms
        data = torch.from_numpy(np.array(sel_num_truth))
        if self.gpu:
            sel_num_truth_var = Variable(data.cuda())
        else:
            sel_num_truth_var = Variable(data)
        loss += self.CE(sel_num_score, sel_num_truth_var)
        # Evaluate the select columns
        T = len(sel_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            truth_prob[b][truth_num[b][1]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            sel_col_truth_var = Variable(data.cuda())
        else:
            sel_col_truth_var = Variable(data)
        sel_col_prob = self.sigm(sel_col_score)
        sel_bce_loss = -torch.mean( 3*(sel_col_truth_var * \
                torch.log(sel_col_prob+1e-10)) + \
                (1-sel_col_truth_var) * torch.log(1-sel_col_prob+1e-10) )
        loss += sel_bce_loss
        #----------------loss for cond_pred--------------------#
        #cond_num_score, cond_col_score, cond_op_score = cond_score

        #Evaluate the number of conditions
        cond_num_truth = [*map(lambda x:x[2], truth_num)]
        data = torch.from_numpy(np.array(cond_num_truth))
        if self.gpu:
            cond_num_truth_var = Variable(data.cuda())
        else:
            cond_num_truth_var = Variable(data)
        loss += self.CE(cond_num_score, cond_num_truth_var)
        #Evaluate the columns of conditions
        T = len(cond_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][3]) > 0:
                truth_prob[b][list(truth_num[b][3])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            cond_col_truth_var = Variable(data.cuda())
        else:
            cond_col_truth_var = Variable(data)

        cond_col_prob = self.sigm(cond_col_score)
        bce_loss = -torch.mean( 3*(cond_col_truth_var * \
                torch.log(cond_col_prob+1e-10)) + \
                (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
        loss += bce_loss
        #Evaluate the operator of conditions
        for b in range(len(truth_num)):
            if len(truth_num[b][4]) == 0:
                continue
            data = torch.from_numpy(np.array(truth_num[b][4]))
            if self.gpu:
                cond_op_truth_var = Variable(data.cuda())
            else:
                cond_op_truth_var = Variable(data)
            cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
            # print 'cond_op_truth_var', list(cond_op_truth_var.size())
            # print 'cond_op_pred', list(cond_op_pred.size())
            loss += (self.CE(cond_op_pred, cond_op_truth_var) \
                    / len(truth_num))
        # -----------loss for group_pred -------------- #
        #gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = group_score

        # Evaluate the number of group by columns
        gby_num_truth = [*map(lambda x: x[7], truth_num)]
        data = torch.from_numpy(np.array(gby_num_truth))
        if self.gpu:
            gby_num_truth_var = Variable(data.cuda())
        else:
            gby_num_truth_var = Variable(data)
        loss += self.CE(gby_num_score, gby_num_truth_var)
        # Evaluate the group by columns
        T = len(gby_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][6]) > 0:
                truth_prob[b][list(truth_num[b][6])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            gby_col_truth_var = Variable(data.cuda())
        else:
            gby_col_truth_var = Variable(data)
        gby_col_prob = self.sigm(gby_score)
        gby_bce_loss = -torch.mean( 3*(gby_col_truth_var * \
                torch.log(gby_col_prob+1e-10)) + \
                (1-gby_col_truth_var) * torch.log(1-gby_col_prob+1e-10) )
        loss += gby_bce_loss
        loss_agg = gby_bce_loss.data.cpu().numpy()
        # Evaluate having
        having_truth = [1 if len(x[13]) == 1 else 0 for x in truth_num]
        data = torch.from_numpy(np.array(having_truth))
        if self.gpu:
            having_truth_var = Variable(data.cuda())
        else:
            having_truth_var = Variable(data)
        loss += self.CE(hv_score, having_truth_var)
        # Evaluate having col
        T = len(hv_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][13]) > 0:
                truth_prob[b][truth_num[b][13]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            hv_col_truth_var = Variable(data.cuda())
        else:
            hv_col_truth_var = Variable(data)
        hv_col_prob = self.sigm(hv_col_score)
        hv_col_bce_loss = -torch.mean( 3*(hv_col_truth_var * \
                torch.log(hv_col_prob+1e-10)) + \
                (1-hv_col_truth_var) * torch.log(1-hv_col_prob+1e-10) )
        loss += hv_col_bce_loss
        # Evaluate having agg
        T = len(hv_agg_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][12]) > 0:
                truth_prob[b][truth_num[b][12]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            hv_agg_truth_var = Variable(data.cuda())
        else:
            hv_agg_truth_var = Variable(data)
        hv_agg_prob = self.sigm(hv_agg_truth_var)
        hv_agg_bce_loss = -torch.mean( 3*(hv_agg_truth_var * \
                torch.log(hv_agg_prob+1e-10)) + \
                (1-hv_agg_truth_var) * torch.log(1-hv_agg_prob+1e-10) )
        loss += hv_agg_bce_loss
        # Evaluate having op
        T = len(hv_op_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][14]) > 0:
                truth_prob[b][truth_num[b][14]] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            hv_op_truth_var = Variable(data.cuda())
        else:
            hv_op_truth_var = Variable(data)
        hv_op_prob = self.sigm(hv_op_truth_var)
        hv_op_bce_loss = -torch.mean( 3*(hv_op_truth_var * \
                torch.log(hv_op_prob+1e-10)) + \
                (1-hv_op_truth_var) * torch.log(1-hv_op_prob+1e-10) )
        loss += hv_op_bce_loss

        # -----------loss for order_pred -------------- #
        #ody_col_score, ody_agg_score, ody_par_score = order_score

        # Evaluate the number of order by columns
        ody_num_truth = [*map(lambda x: x[10], truth_num)]
        data = torch.from_numpy(np.array(ody_num_truth))
        if self.gpu:
            ody_num_truth_var = Variable(data.cuda())
        else:
            ody_num_truth_var = Variable(data)
        loss += self.CE(ody_num_score, ody_num_truth_var)
        # Evaluate the order by columns
        T = len(ody_col_score[0])
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][9]) > 0:
                truth_prob[b][list(truth_num[b][9])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            ody_col_truth_var = Variable(data.cuda())
        else:
            ody_col_truth_var = Variable(data)
        ody_col_prob = self.sigm(ody_col_score)
        ody_bce_loss = -torch.mean( 3*(ody_col_truth_var * \
                torch.log(ody_col_prob+1e-10)) + \
                (1-ody_col_truth_var) * torch.log(1-ody_col_prob+1e-10) )
        loss += ody_bce_loss
        # Evaluate order agg assume only one
        T = 6
        truth_prob = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            if len(truth_num[b][9]) > 0:
                truth_prob[b][list(truth_num[b][8])] = 1
        data = torch.from_numpy(truth_prob)
        if self.gpu:
            ody_agg_truth_var = Variable(data.cuda())
        else:
            ody_agg_truth_var = Variable(data)
        ody_agg_prob = self.sigm(ody_agg_score)
        ody_agg_bce_loss = -torch.mean( 3*(ody_agg_truth_var * \
                torch.log(ody_agg_prob+1e-10)) + \
                (1-ody_agg_truth_var) * torch.log(1-ody_agg_prob+1e-10) )
        loss += ody_agg_bce_loss
        # Evaluate parity
        ody_par_truth = [*map(lambda x: x[11], truth_num)]
        data = torch.from_numpy(np.array(ody_par_truth))
        if self.gpu:
            ody_par_truth_var = Variable(data.cuda())
        else:
            ody_par_truth_var = Variable(data)
        loss += self.CE(ody_par_score, ody_par_truth_var)
        return loss


    def check_acc(self, pred_queries, gt_data):

        vis_info, gt_queries, from_err_seq = gt_data[2:5]

        tot_err = 0.0
        sel_err = agg_num_err = agg_op_err = sel_num_err = sel_col_err = 0.0
        cond_err = cond_num_err = cond_col_err = cond_op_err = 0.0
        gby_err = gby_num_err = gby_col_err = hv_err = hv_col_err = hv_agg_err = hv_op_err = 0.0
        ody_err = ody_num_err = ody_col_err = ody_agg_err = ody_par_err = 0.0

        # Get from clause error
        from_err = sum(from_err_seq)

        for b, (pred_qry, gt_qry, vis_data) in enumerate(zip(pred_queries, gt_queries, vis_info)):

            good = True
            tot_flag = True
            sel_flag = True
            cond_flag = True
            gby_flag = True
            ody_flag = True
            # sel
            sel_gt = gt_qry['sel']
            sel_num_gt = len(set(sel_gt))
            sel_pred = pred_qry['sel']
            sel_num_pred = pred_qry['sel_num']
            if sel_num_pred != sel_num_gt:
                sel_num_err += 1
                sel_flag = False
            if sorted(set(sel_pred)) != sorted(set(sel_gt)):
                sel_col_err += 1
                sel_flag = False

            agg_gt = gt_qry['agg']
            curr_col = gt_qry['sel'][0]
            curr_col_num_aggs = 0
            gt_aggs_num = []
            gt_sel_order = [curr_col]
            for i, col in enumerate(gt_qry['sel']):
                if col != curr_col:
                    gt_sel_order.append(col)
                    gt_aggs_num.append(curr_col_num_aggs)
                    curr_col = col
                    curr_col_num_aggs = 0
                if agg_gt[i] != 0:
                    curr_col_num_aggs += 1
            gt_aggs_num.append(curr_col_num_aggs)

            if pred_qry['agg_num'] != gt_aggs_num:
                agg_num_err += 1
                sel_flag = False

            if sorted(pred_qry['agg']) != sorted(gt_qry['agg']): # naive
                agg_op_err += 1
                sel_flag = False

            if not sel_flag:
                sel_err += 1
                good = False

            # From
            if self.use_from and from_err_seq[b]:
                good = False

            # group
            gby_gt = gt_qry['group'][:-1]
            gby_pred = pred_qry['group']
            gby_num_pred = pred_qry['gby_num']
            gby_num_gt = len(gby_gt)
            if gby_num_pred != gby_num_gt:
                gby_num_err += 1
                gby_flag = False
            if sorted(gby_pred) != sorted(gby_gt):
                gby_col_err += 1
                gby_flag = False
            gt_gby_agg = gt_qry['group'][-1][0]
            gt_gby_col = gt_qry['group'][-1][1]
            gt_gby_op = gt_qry['group'][-1][2]
            if gby_num_pred != 0 and len(gt_gby_col) != 0:
                if pred_qry['hv'] != 1:
                    hv_err += 1
                    gby_flag = False
                if pred_qry['hv_agg'] != gt_gby_agg[0]:
                    hv_agg_err += 1
                    gby_flag = False
                if pred_qry['hv_col'] != gt_gby_col[0]:
                    hv_col_err += 1
                    gby_flag = False
                if pred_qry['hv_op'] != gt_gby_op[0]:
                    hv_op_err += 1
                    gby_flag = False

            if not gby_flag:
                gby_err += 1
                good = False

            # order
            ody_gt_aggs = gt_qry['order'][0]
            ody_gt_cols = gt_qry['order'][1]
            ody_gt_par = gt_qry['order'][2]
            ody_num_cols_pred = pred_qry['ody_num']
            ody_cols_pred = pred_qry['order']
            ody_aggs_pred = pred_qry['ody_agg']
            ody_par_pred = pred_qry['parity']

            if ody_num_cols_pred != len(ody_gt_cols):
                ody_num_err += 1
                ody_flag = False
            if len(ody_gt_cols) > 0:
                if ody_cols_pred != ody_gt_cols:
                    ody_col_err += 1
                    ody_flag = False
                if ody_aggs_pred != ody_gt_aggs:
                    ody_agg_err += 1
                    ody_flag = False
                if ody_par_pred != ody_gt_par:
                    ody_par_err += 1
                    ody_flag = False

            if not ody_flag:
                ody_err += 1
                good = False

            # conds
            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['cond']
            flag = True
            if len(cond_pred) != len(cond_gt):
                flag = False
                cond_num_err += 1
                cond_flag = False
            if flag and set(x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                flag = False
                cond_col_err += 1
                cond_flag = False
            for idx in range(len(cond_pred)):
                if not flag:
                    break
                gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                    flag = False
                    cond_op_err += 1
                    cond_flag = False

            if not cond_flag:
                cond_err += 1
                good = False

            if not good:
                tot_err += 1

        # Get Acc
        acc_list = [0.0] * 6
        err_list = [tot_err, sel_err, cond_err, gby_err, ody_err, from_err]
        for i in range(6):
            acc_list[i] = len(pred_queries) - err_list[i]

        return np.array(acc_list)

    def gen_query(self, score):

        sel_score, from_score, cond_score, group_score, order_score = score
        sel_num_score, sel_col_score, agg_num_score, agg_op_score = [x.data.cpu().numpy() if x is not None else None for x in sel_score]
        cond_num_score, cond_col_score, cond_op_score = [x.data.cpu().numpy() if x is not None else None for x in cond_score]
        gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = [x.data.cpu().numpy() if x is not None else None for x in group_score]
        ody_num_score, ody_col_score, ody_agg_score, ody_par_score = [x.data.cpu().numpy() if x is not None else None for x in order_score]

        ret_queries = []
        B = len(sel_num_score)
        for b in range(B):
            cur_query = {}
            # ------------get sel predict
            sel_num_cols = np.argmax(sel_num_score[b]) + 1
            cur_query['sel_num'] = sel_num_cols
            cur_query['sel'] = np.argsort(-sel_col_score[b])[:sel_num_cols]

            agg_nums = []
            agg_preds = []
            for idx in range(sel_num_cols):
                curr_num_aggs = np.argmax(agg_num_score[b][idx])
                agg_nums.append(curr_num_aggs)
                if curr_num_aggs == 0:
                    curr_agg_ops = [0]
                else:
                    curr_agg_ops = [x for x in list(np.argsort(-agg_op_score[b][idx])) if x != 0][:curr_num_aggs]
                agg_preds += curr_agg_ops
            cur_query['agg_num'] = agg_nums
            cur_query['agg'] = agg_preds
            #----------get group by predict
            gby_num_cols = np.argmax(gby_num_score[b])
            cur_query['gby_num'] = gby_num_cols
            cur_query['group'] = np.argsort(-gby_score[b])[:gby_num_cols]
            cur_query['hv'] = np.argmax(hv_score[b])
            if gby_num_cols != 0 and cur_query['hv'] != 0:
                cur_query['hv_agg'] = np.argmax(hv_agg_score[b])
                cur_query['hv_col'] = np.argmax(hv_col_score[b])
                cur_query['hv_op'] = np.argmax(hv_op_score[b])
            else:
                cur_query['hv'] = 0
                cur_query['hv_agg'] = 0
                cur_query['hv_col'] = -1
                cur_query['hv_op'] = -1
            # --------get order by
            ody_num_cols = np.argmax(ody_num_score[b])
            cur_query['ody_num'] = ody_num_cols
            cur_query['order'] = np.argsort(-ody_col_score[b])[:ody_num_cols]
            if ody_num_cols != 0:
                cur_query['ody_agg'] = np.argmax(ody_agg_score[b])
                cur_query['parity'] = np.argmax(ody_par_score[b])
            else:
                cur_query['ody_agg'] = 0
                cur_query['parity'] = -1

            # ody_agg_preds = []
            # for idx in range(len(gt_ody[b])):           # eventually dont use gold (look at agg query generation)
            #     curr_ody_agg = np.argmax(ody_agg_score[b][idx])
            #     ody_agg_preds += curr_ody_agg
            #
            # cur_query['ody_agg'] = ody_agg_preds
            # cur_query['parity'] = np.argmax(ody_par_score[b]) - 1
            #---------get cond predict
            #cond_num_score, cond_col_score, cond_op_score = [x.data.cpu().numpy() if x is not None else None for x in cond_score]
            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(max_idxes[idx])
                cur_cond.append(np.argmax(cond_op_score[b][idx]))
                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries

    def gen_sql(self, score, gt_data):
        col_org, schema_seq, full_graphs = gt_data[5:8]

        sel_score, from_score, cond_score, group_score, order_score = score
        sel_num_score, sel_col_score, agg_num_score, agg_op_score = [x.data.cpu().numpy() if x is not None else None for x in sel_score]
        cond_num_score, cond_col_score, cond_op_score = [x.data.cpu().numpy() if x is not None else None for x in cond_score]
        gby_num_score, gby_score, hv_score, hv_col_score, hv_agg_score, hv_op_score = [x.data.cpu().numpy() if x is not None else None for x in group_score]
        ody_num_score, ody_col_score, ody_agg_score, ody_par_score = [x.data.cpu().numpy() if x is not None else None for x in order_score]

        ret_queries = []
        ret_sqls = []
        B = len(sel_num_score)

        for b in range(B):
            cur_cols = col_org[b]
            cur_query = {}
            schema = schema_seq[b]
            #for generate sql
            cur_sql = []
            cur_sel = []
            cur_conds = []
            cur_group = []
            cur_order = []
            cur_tables = defaultdict(list)

            # ------------get sel predict
            sel_num_cols = np.argmax(sel_num_score[b]) + 1
            cur_query['sel_num'] = sel_num_cols
            cur_query['sel'] = np.argsort(-sel_col_score[b])[:sel_num_cols]

            agg_nums = []
            agg_preds = []
            agg_preds_gen = []
            for idx in range(sel_num_cols):
                curr_num_aggs = np.argmax(agg_num_score[b][idx])
                agg_nums.append(curr_num_aggs)
                if curr_num_aggs == 0:
                    curr_agg_ops = [0]
                else:
                    curr_agg_ops = [x for x in list(np.argsort(-agg_op_score[b][idx])) if x != 0][:curr_num_aggs]
                agg_preds += curr_agg_ops
                agg_preds_gen.append(curr_agg_ops)
            cur_query['agg_num'] = agg_nums
            cur_query['agg'] = agg_preds
            # for gen sel


            cur_sel.append("select")
            for i, cid in enumerate(cur_query['sel']):
                aggs = agg_preds_gen[i]
                agg_num = len(aggs)
                for j, gix in enumerate(aggs):
                    if gix == 0:
                        cur_sel.append([cid, cur_cols[cid][1]])
                        cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                    else:
                        cur_sel.append(AGG_OPS[gix])
                        cur_sel.append("(")
                        cur_sel.append([cid, cur_cols[cid][1]])
                        cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                        cur_sel.append(")")
                    if j < agg_num - 1:
                        cur_sel.append(",")

                if i < sel_num_cols-1:
                    cur_sel.append(",")


            #----------get group by predict
            gby_num_cols = np.argmax(gby_num_score[b])
            cur_query['gby_num'] = gby_num_cols
            cur_query['group'] = np.argsort(-gby_score[b])[:gby_num_cols]
            cur_query['hv'] = np.argmax(hv_score[b])
            if gby_num_cols != 0 and cur_query['hv'] != 0:
                cur_query['hv_agg'] = np.argmax(hv_agg_score[b])
                cur_query['hv_col'] = np.argmax(hv_col_score[b])
                cur_query['hv_op'] = np.argmax(hv_op_score[b])
            else:
                cur_query['hv'] = 0
                cur_query['hv_agg'] = 0
                cur_query['hv_col'] = -1
                cur_query['hv_op'] = -1

            # for gen group
            if gby_num_cols > 0:
                cur_group.append("group by")
                for i, gid in enumerate(cur_query['group']):
                    cur_group.append([gid, cur_cols[gid][1]])
                    cur_tables[cur_cols[gid][0]].append([gid, cur_cols[gid][1]])
                    if i < gby_num_cols-1:
                        cur_group.append(",")
                if cur_query['hv'] != 0:
                    cur_group.append("having")
                    if cur_query['hv_agg'] != 0:
                        cur_group.append(AGG_OPS[cur_query['hv_agg']])
                        cur_group.append("(")
                        cur_group.append([cur_query['hv_col'], cur_cols[cur_query['hv_col']][1]])
                        cur_group.append(")")
                    else:
                        cur_group.append([cur_query['hv_col'], cur_cols[cur_query['hv_col']][1]])
                    cur_tables[cur_cols[cur_query['hv_col']][0]].append([cur_query['hv_col'],cur_cols[cur_query['hv_col']][1]])
                    cur_group.append(WHERE_OPS[cur_query['hv_op']])
                    cur_group.append(VALUE)

            # --------get order by
            ody_num_cols = np.argmax(ody_num_score[b])
            cur_query['ody_num'] = ody_num_cols
            cur_query['order'] = np.argsort(-ody_col_score[b])[:ody_num_cols]
            if ody_num_cols != 0:
                cur_query['ody_agg'] = np.argmax(ody_agg_score[b])
                cur_query['parity'] = np.argmax(ody_par_score[b])
            else:
                cur_query['ody_agg'] = 0
                cur_query['parity'] = -1

            # for gen order
            if ody_num_cols > 0:
                cur_order.append("order by")
                for oid in cur_query['order']:
                    if cur_query['ody_agg'] != 0:
                        cur_order.append(AGG_OPS[cur_query['ody_agg']])
                        cur_order.append("(")
                        cur_order.append([oid, cur_cols[oid][1]])
                        cur_order.append(")")
                    else:
                        cur_order.append([oid, cur_cols[oid][1]])
                    cur_tables[cur_cols[oid][0]].append([oid, cur_cols[oid][1]])

                datid = cur_query['parity']
                if datid == 0:
                    cur_order.append(DESC_ASC_LIMIT[0])
                elif datid == 1:
                    cur_order.append(DESC_ASC_LIMIT[1])
                elif datid == 2:
                    cur_order.append(DESC_ASC_LIMIT[2])
                elif datid == 3:
                    cur_order.append(DESC_ASC_LIMIT[3])

            #---------get cond predict
            #cond_num_score, cond_col_score, cond_op_score = [x.data.cpu().numpy() if x is not None else None for x in cond_score]
            cur_query['conds'] = []
            cond_num = np.argmax(cond_num_score[b])
            max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
            for idx in range(cond_num):
                cur_cond = []
                cur_cond.append(max_idxes[idx])
                cur_cond.append(np.argmax(cond_op_score[b][idx]))
                cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

            # for gen conds
            if len(cur_query['conds']) > 0:
                cur_conds.append("where")
                for i, cond in enumerate(cur_query['conds']):
                    cid, oid = cond
                    cur_conds.append([cid, cur_cols[cid][1]])
                    cur_tables[cur_cols[cid][0]].append([cid, cur_cols[cid][1]])
                    cur_conds.append(WHERE_OPS[oid])
                    cur_conds.append(VALUE)
                    if i < cond_num-1:
                        cur_conds.append("and")


            if -1 in cur_tables.keys():
                del cur_tables[-1]

            if full_graphs is None:
                table_alias_dict, ret = self._gen_from(cur_tables.keys(), schema)
            else:
                table_alias_dict, ret = self._gen_from_graph(full_graphs[b], schema)

            if len(table_alias_dict) > 0:
                col_map = {}
                for tid, aid in table_alias_dict.items():
                    for cid, col in cur_tables[tid]:
                        col_map[cid] = "t" + str(aid) + "." + col

                new_sel = []
                for s in cur_sel:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_sel.append("*")
                        elif s[0] in col_map:
                            new_sel.append(col_map[s[0]])
                    else:
                        new_sel.append(s)

                new_conds = []
                for s in cur_conds:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_conds.append("*")
                        else:
                            new_conds.append(col_map[s[0]])
                    else:
                        new_conds.append(s)

                new_group = []
                for s in cur_group:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_group.append("*")
                        else:
                            new_group.append(col_map[s[0]])
                    else:
                        new_group.append(s)

                new_order = []
                for s in cur_order:
                    if isinstance(s, list):
                        if s[0] == 0:
                            new_order.append("*")
                        else:
                            new_order.append(col_map[s[0]])
                    else:
                        new_order.append(s)

                            # for gen all sql
                cur_sql = new_sel + [ret] + new_conds + new_group + new_order
            else:
                cur_sql = []
                #try:
                cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_sel])
                if full_graphs:
                    cur_sql.extend(["from", schema["table_names_original"][list(full_graphs[b])[0]]])
                else:
                    if len(cur_tables.keys()) == 0:
                        cur_tables[0] = []
                    cur_sql.extend(["from", schema["table_names_original"][list(cur_tables.keys())[0]]])
                if len(cur_conds) > 0:
                    cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_conds])
                if len(cur_group) > 0:
                    cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_group])
                if len(cur_order) > 0:
                    cur_sql.extend([s[1] if isinstance(s, list) else s for s in cur_order])

            sql_str = " ".join(cur_sql)
            ret_sqls.append(sql_str)

        return ret_sqls

    def evaluate(self, score, gt_data):
        pred_queries = self.gen_query(score)
        acc = self.check_acc(pred_queries, gt_data)
        return acc

    def preprocess(self, batch):
        q_seq = []
        q_seq_concol = []
        col_seq = []
        col_num = []
        ans_seq = []
        query_seq = []
        gt_cond_seq = []
        vis_seq = []
        col_org_seq = []
        schema_seq = []
        from_col_seq = []
        history = [['root', 'none']] * len(batch)
        tabs = []
        cols = []
        q_type = []
        labels = []
        foreign_keys = []
        primary_keys = []
        tbl_ans = []

        for item in batch:
            # q_seq
            q_seq.append(item['question_toks'])

            # q_seq_concol
            q_seq_concol.append(item['question_tok_concol'])

            # col_seq
            db = item['db']
            col_names = [col[1] for col in db['column_names']]
            col_seq.append([x.split(' ') for x in col_names])

            # col_num
            col_num.append(len(col_names))

            # ans_seq
            ans_seq.append((item['agg'],
                           item['sel'],
                           len(item['cond']),
                           tuple(x[0] for x in item['cond']),
                           tuple(x[1] for x in item['cond']),
                           len(set(item['sel'])),
                           item['group'][:-1],  # group by columns 6
                           len(item['group']) - 1,  # number of group by columns 7
                           item['order'][0],  # order by aggregations 8
                           item['order'][1],  # order by columns 9
                           len(item['order'][1]),  # num order by columns 10
                           item['order'][2],  # order by parity 11
                           item['group'][-1][0],  # having agg 12
                           item['group'][-1][1],  # having col 13
                           item['group'][-1][2]  # having op 14
                           ))
            # query_seq
            query_seq.append(item['query_toks'])

            # gt_cond_seq
            gt_cond_seq.append([x for x in item['cond']])

            # vis_seq
            vis_seq.append((item['question'], col_names, item['query']))

            # col_org_seq
            col_org_seq.append(item['column'])

            # schema_seq
            schema_seq.append(db)

            # from_col_seq
            cols_add = []
            tab_seq = [tmp[0] for tmp in db['column_names']]
            col_type = db['column_types']
            tname_toks = [x.split(" ") for x in db['table_names']]
            for tid, col, ct in zip(tab_seq, col_names, col_type):
                col_one = [ct]
                tabn = ['all'] if tid == -1 else tname_toks[tid]
                for t in tabn:
                    if t not in col:
                        col_one.append(t)
                col_one.append(col)
                cols_add.append(col_one)
            from_col_seq.append(cols_add)

            # tabs
            tabs.append(db['table_names'])

            # cols
            cols.append(db['column_names'])

            # q_type
            q_type.append(item["question_type_concol_list"])

            # labels
            labels.append(item['join_table_dict'])

            # foreign keys
            foreign_keys.append(db['foreign_keys'])

            # primary keys
            primary_keys.append(db['primary_keys'])

        for idx, item in enumerate(batch):
            query_tok = item['query_toks']
            # Get Table names
            table_names = []
            last_key_word = ''
            for idx2 in range(len(query_tok)):
                if query_tok[idx2] in ['select', 'SELECT', 'from', 'FROM', 'where', 'WHERE', 'group', 'GROUP', 'having',
                                       'HAVING']:
                    last_key_word = query_tok[idx2]
                if query_tok[idx2] in ['from', 'FROM'] and query_tok[idx2 + 1] != '(':
                    table_names += [query_tok[idx2 + 1]]
                elif query_tok[idx2] in ['as', 'AS'] and last_key_word in ['from', 'FROM']:
                    table_names += [query_tok[idx2 - 1]]

            table_names = list(set(table_names))

            # Get Tables' designated numbers
            table_org = [n.lower() for n in item['tbl_org']]
            table_in_num = [table_org.index(name.lower()) for name in table_names]

            # Ground truth table
            tbl_ans += (table_in_num,)

        # Embedding
        if self.bert:
            q_emb, q_len, q_q_len, label, sep_embeddings = self.embed_layer.gen_bert_batch_with_table(q_seq, tabs, cols, foreign_keys, primary_keys, labels)
        else:
            q_emb = q_len = q_q_len = label = sep_embeddings = None

        q_embs = []
        q_lens = []
        q_q_lens = []
        schemas = []
        table_graph_lists = []
        full_graph_lists = []
        sep_embedding_lists = []

        for item in batch:
            labels.append(item['join_table_dict'])
            schemas.append(item['schema'])
            q_emb, q_len, q_q_len, table_graph_list, full_graph_list, sep_embeddings = self.embed_layer.gen_bert_for_eval(
                item['question_toks'], item['schema'])
            q_embs.append(q_emb)
            q_lens.append(q_len)
            q_q_lens.append(q_q_len)
            table_graph_lists.append(table_graph_list)
            full_graph_lists.append(full_graph_list)
            sep_embedding_lists.append(sep_embeddings)

        # From predictor
        if self.use_from and not self.training:
            input_data, gt_data = self.from_pred.preprocess(batch)
            scores = self.from_pred(input_data)
            from_err_seq, selected_tbls, full_graphs = self.from_pred.check_acc(scores, gt_data, batch)
        else:
            from_err_seq = [0] * len(batch)
            selected_tbls = [[0]] * len(batch)
            full_graphs = None

        # To-Do: change to dictionary?
        gt_data = (ans_seq, label, vis_seq, batch, from_err_seq, col_org_seq, schema_seq, full_graphs)
        input_data = (q_seq_concol, col_num, col_org_seq, tbl_ans, from_col_seq, history, q_embs, q_lens, q_q_lens, sep_embedding_lists, cols, q_type, selected_tbls)

        return input_data, gt_data