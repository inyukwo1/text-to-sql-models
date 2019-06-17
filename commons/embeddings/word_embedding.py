import os
import json
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from commons.embeddings.graph_utils import *

class WordEmbedding(nn.Module):
    def __init__(self, glove_path, N_word, gpu, SQL_TOK, use_bert=False,
            trainable=False, use_small=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK
        self.use_bert = use_bert
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)

        word_emb = self._load_glove(glove_path, trainable, use_small)

        if trainable:
            print("Using trainable embedding")
            self.w2i, word_emb_val = word_emb
            # trainable when using pretrained model, init embedding weights using prev embedding
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            # else use word2vec or glove
            self.word_emb = word_emb
            print("Using fixed embedding for words but trainable embedding for types")

    def _load_glove(self, file_name, load_used, use_small):
        if not load_used:
            cached_file_path = file_name + ".cache"
            if os.path.isfile(cached_file_path):
                with open(cached_file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(('Loading word embedding from %s' % file_name))
                ret = {}
                with open(file_name) as inf:
                    for idx, line in enumerate(inf):
                        if (use_small and idx >= 500):
                            break
                        info = line.strip().split(' ')
                        if info[0].lower() not in ret:
                            ret[info[0]] = np.array([float(x) for x in info[1:]])
                with open(cached_file_path, 'wb') as f:
                    pickle.dump(ret, f)
                return ret
        else:
            print ('Load used word embedding')
            with open('../alt/glove/word2idx.json') as inf:
                w2i = json.load(inf)
            with open('../alt/glove/usedwordemb.npy') as inf:
                word_emb_val = np.load(inf)
            return w2i, word_emb_val

    def word_find(self, word):
        # word = ''.join([i for i in word if i.isalpha()])
        # word = word.lower()
        return self.word_emb.get(word, np.zeros(self.N_word, dtype=np.float32))

    def gen_xc_type_batch(self, xc_type, is_col=False, is_list=False):
        B = len(xc_type)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(xc_type):
            if is_list:
                q_val = [*map(lambda x:self.w2i.get(" ".join(sorted(x)), 0), one_q)]
            else:
                q_val = [*map(lambda x:self.w2i.get(x, 0), one_q)]
            if is_col:
                val_embs.append(q_val)  #<BEG> and <END>
                val_len[i] = len(q_val)
            else:
                val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)
        val_tok_array = np.zeros((B, max_len), dtype=np.int64)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_tok_array[i,t] = val_embs[i][t]
        val_tok = torch.from_numpy(val_tok_array)
        if self.gpu:
            val_tok = val_tok.cuda()
        val_tok_var = Variable(val_tok)
        val_inp_var = self.embedding(val_tok_var)

        return val_inp_var, val_len


    def gen_x_batch(self, q, is_list=False, is_q=False):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(q):
            if self.trainable:
                q_val = [*map(lambda x:self.w2i.get(x, 0), one_q)]
            elif not is_list:
                q_val = [*map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q)]
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        tmp = self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32))
                        tmp = list(tmp.tolist())
                        tmp = np.array(tmp) if tmp else np.zeros(self.N_word, dtype=np.float32)
                        emb_list.append(tmp)
                        #emb_list.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32))) 
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))
            if self.trainable:
                val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            elif not is_list or is_q:
                val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                val_embs.append(q_val)
                val_len[i] = len(q_val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_x_q_bert_batch(self, q):
        tokenized_q = []
        q_len = np.zeros(len(q), dtype=np.int64)
        for idx, one_q in enumerate(q):
            tokenized_one_q = self.bert_tokenizer.tokenize(" ".join(one_q))
            indexed_one_q = self.bert_tokenizer.convert_tokens_to_ids(tokenized_one_q)
            tokenized_q.append(indexed_one_q)
            q_len[idx] = len(indexed_one_q)
        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        tokenized_q = torch.LongTensor(tokenized_q)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
        return tokenized_q, q_len

    def encode_one_q_with_bert(self, one_q, table_name, table_cols, parent_tables, foreign_keys, primary_keys, table_graph):
        input_q = "[CLS] " + " ".join(one_q)
        one_q_q_len = len(self.bert_tokenizer.tokenize(input_q))
        for table_num in table_graph:
            input_q += " [SEP] " + table_name[table_num]
        # table_names = [tables[idx][table_num] for table_num in generated_graph]
        # input_q += " ".join(table_names)
        col_name_dict = {}
        for table_num in table_graph:
            col_name_dict[table_num] = []
        renamed_table_cols = deepcopy(table_cols)
        for col_idx, [par_tab, col_name] in enumerate(renamed_table_cols):
            if par_tab in table_graph:
                if col_idx in table_graph[par_tab]:
                    if col_idx in primary_keys:
                        for f, p in foreign_keys:
                            if parent_tables[f] in table_graph and f in table_graph[parent_tables[f]] and p == col_idx:
                                _, col_name = renamed_table_cols[f]
                        for f, p in foreign_keys:
                            if parent_tables[f] in table_graph and f in table_graph[parent_tables[f]] and p == col_idx:
                                renamed_table_cols[f][1] = col_name
                    col_name_dict[par_tab].append(col_name)
                else:
                    col_name_dict[par_tab].append(col_name)
        col_name_list = [l for k, l in col_name_dict.items()]
        col_name_len_list = [len(l) for l in col_name_list]
        sep_embeddings = list(range(len(table_graph)))
        for k_idx, k in enumerate(table_graph):
            for cidx in range(max(col_name_len_list)):
                l = col_name_dict[k]
                if cidx < len(l):
                    input_q += " [SEP] " + l[cidx]
                    sep_embeddings.append(k_idx)

        tokenozed_one_q = self.bert_tokenizer.tokenize(input_q)
        indexed_one_q = self.bert_tokenizer.convert_tokens_to_ids(tokenozed_one_q)

        sep_embeddings_per_loc = []
        cur_sep_cnt = -1
        for token_idx, token in enumerate(tokenozed_one_q):
            if token == '[SEP]':
                cur_sep_cnt += 1
                sep_embeddings_per_loc.append(sep_embeddings[cur_sep_cnt])
            else:
                sep_embeddings_per_loc.append(-1)
        return one_q_q_len, indexed_one_q, sep_embeddings_per_loc

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)
        #TODO: what is the diff bw name_len and col_len?
        name_inp_var, name_len = self.str_list_to_batch(names)

        return name_inp_var, name_len, col_len


    def gen_agg_batch(self, q):
        B = len(q)
        ret = []
        agg_ops = ['none', 'maximum', 'minimum', 'count', 'total', 'average']
        for b in range(B):
            if self.trainable:
                ct_val = map(lambda x:self.w2i.get(x, 0), agg_ops)
            else:
                ct_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), agg_ops)
            ret.append(ct_val)

        agg_emb_array = np.zeros((B, 6, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(ret[i])):
                agg_emb_array[i,t,:] = ret[i][t]
        agg_inp = torch.from_numpy(agg_emb_array)
        if self.gpu:
            agg_inp = agg_inp.cuda()
        agg_inp_var = Variable(agg_inp)

        return agg_inp_var


    def str_list_to_batch(self, str_list):
        """get a list var of wemb of words in each column name in current bactch"""
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


    def gen_x_history_batch(self, history):
        B = len(history)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_history in enumerate(history):
            history_val = []
            for item in one_history:
                #col
                if isinstance(item, list) or isinstance(item, tuple):
                    emb_list = []
                    ws = item[0].split() + item[1].split()
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_find(w))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        history_val.append(emb_list[0])
                    else:
                        history_val.append(sum(emb_list) / float(ws_len))
                #ROOT
                elif isinstance(item,str):
                    if item == "ROOT":
                        item = "root"
                    elif item == "asc":
                        item = "ascending"
                    elif item == "desc":
                        item = "descending"
                    if item in (
                    "none", "select", "from", "where", "having", "limit", "intersect", "except", "union", 'not',
                    'between', '=', '>', '<', 'in', 'like', 'is', 'exists', 'root', 'ascending', 'descending'):
                        history_val.append(self.word_find(item))
                    elif item == "orderBy":
                        history_val.append((self.word_find("order") +
                                            self.word_find("by")) / 2)
                    elif item == "groupBy":
                        history_val.append((self.word_find("group") +
                                            self.word_find("by")) / 2)
                    elif item in ('>=', '<=', '!='):
                        history_val.append((self.word_find(item[0]) +
                                            self.word_find(item[1])) / 2)
                elif isinstance(item,int):
                    history_val.append(self.word_find(AGG_OPS[item]))
                else:
                    print(("Warning: unsupported data type in history! {}".format(item)))

            val_embs.append(history_val)
            val_len[i] = len(history_val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len

    def gen_bert_batch_with_table(self, q, tables, table_cols, foreign_keys, primary_keys, labels):
        tokenized_q = []
        q_len = []
        q_q_len = []
        anses = []
        sep_embeddings = []
        for idx, one_q in enumerate(q):
            parent_tables = []
            for t, c in table_cols[idx]:
                parent_tables.append(t)

            if random.randint(0, 100) < 7:
                true_graph = 1.
                generated_graph = str_graph_to_num_graph(labels[idx])
            else:
                true_graph = 0.
                generated_graph = generate_random_graph_generate(len(tables[idx]), parent_tables, foreign_keys[idx])
                if graph_checker(generated_graph, labels[idx], foreign_keys[idx], primary_keys[idx]):
                    true_graph = 1.
            anses.append(true_graph)

            one_q_q_len, indexed_one_q, one_sep_embeddings \
                = self.encode_one_q_with_bert(one_q, tables[idx], table_cols[idx], parent_tables, foreign_keys[idx], primary_keys[idx], generated_graph)
            q_q_len.append(one_q_q_len)
            tokenized_q.append(indexed_one_q)
            q_len.append(len(indexed_one_q))
            sep_embeddings.append(one_sep_embeddings)

        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        tokenized_q = torch.LongTensor(tokenized_q)
        anses = torch.tensor(anses)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
            anses = anses.cuda()
        return tokenized_q, q_len, q_q_len, anses, sep_embeddings

    def gen_bert_for_eval(self, one_q, one_tables, one_cols, foreign_keys, primary_keys):
        tokenized_q = []
        parent_nums = []
        sep_embeddings = []

        for par_tab, _ in one_cols:
            parent_nums.append(par_tab)
        table_graph_lists = []
        for tab in range(len(one_tables)):
            table_graph_lists += list(generate_four_hop_path_from_seed(tab, parent_nums, foreign_keys))

        simple_graph_lists = []
        for graph in table_graph_lists:
            new_graph = deepcopy(graph)
            for k in new_graph:
                for idx, l in enumerate(new_graph[k]):
                    new_graph[k][idx] = l[0]
            simple_graph_lists.append(new_graph)
        B = len(table_graph_lists)
        q_len = []
        q_q_len = []
        for b in range(B):

            one_q_q_len, indexed_one_q, one_sep_embeddings \
                = self.encode_one_q_with_bert(one_q, one_tables, one_cols, parent_nums, foreign_keys, primary_keys,
                                                simple_graph_lists[b])
            q_q_len.append(one_q_q_len)
            tokenized_q.append(indexed_one_q)
            q_len.append(len(indexed_one_q))
            sep_embeddings.append(one_sep_embeddings)

        max_len = max(q_len)
        for tokenized_one_q in tokenized_q:
            tokenized_one_q += [0] * (max_len - len(tokenized_one_q))
        tokenized_q = torch.LongTensor(tokenized_q)
        if self.gpu:
            tokenized_q = tokenized_q.cuda()
        return tokenized_q, q_len, q_q_len, simple_graph_lists, table_graph_lists, sep_embeddings

    def gen_word_list_embedding(self,words,B):
        val_emb_array = np.zeros((B,len(words), self.N_word), dtype=np.float32)
        for i,word in enumerate(words):
            if len(word.split()) == 1:
                emb = self.word_find(word)
            else:
                word = word.split()
                emb = (self.word_find(word[0]) + self.word_find(word[1]))/2
            for b in range(B):
                val_emb_array[b,i,:] = emb
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def gen_x_q_batch(self, q):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(q):
            q_val = []
            for ws in one_q:
                q_val.append(self.word_find(ws))

            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
            val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len