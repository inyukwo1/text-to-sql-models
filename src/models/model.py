# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : model.py
# @Software: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from src.beam import Beams, ActionInfo
from src.dataset import Batch
from src.models import nn_utils
from src.models.bert_rat import BERT_RAT
from src.models.basic_model import BasicModel
from src.models.pointer_net import PointerNet
from src.rule import semQL as define_rule
from transformers import *
from src.models.ra_transformer import RATransformerEncoder
from preprocess.relation_types import RELATION_LIST

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased', 768),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]


class IRNet(BasicModel):
    
    def __init__(self, args, grammar):
        super(IRNet, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer
        self.use_sentence_features = args.sentence_features

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor
        if args.bert != -1:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[args.bert]
            args.hidden_size = dim
            args.col_embed_size = dim
            args.embed_size = dim
            args.att_vec_size = dim
        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size // 2, bidirectional=True,
                                    batch_first=True)


        input_dim = args.action_embed_size + \
                    args.att_vec_size + \
                    args.type_embed_size
        # previous action
        # input feeding
        # pre type embedding

        self.lf_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        self.att_sketch_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(9, args.col_embed_size)
        self.tab_type = nn.Linear(5, args.col_embed_size)
        self.sketch_encoder = nn.LSTM(args.action_embed_size, args.action_embed_size // 2, bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        self.att_project = nn.Linear(args.hidden_size + args.type_embed_size, args.hidden_size)

        self.N_embed = nn.Embedding(len(define_rule.N._init_grammar()), args.action_embed_size)

        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.q_att = nn.Linear(args.hidden_size, args.embed_size)

        self.column_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.col_embed_size, args.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, args.col_embed_size, attention_type=args.column_att)
        if args.bert != -1:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[args.bert]
            self.rat_encoder = RATransformerEncoder(dim, 8, len(RELATION_LIST), 2048)
        self.without_bert_params = list(self.parameters(recurse=True))
        if args.bert != -1:
            model_class, tokenizer_class, pretrained_weight, dim = MODELS[args.bert]
            # self.transformer_encoder = model_class.from_pretrained(pretrained_weight)
            self.bert_rat = BERT_RAT(len(RELATION_LIST))
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
            # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[table]", "[column]", "[value]"]})
            self.transformer_dim = dim
            self.col_lstm = torch.nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
            self.tab_lstm = torch.nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
            args.hidden_size = dim
            args.col_embed_size = dim

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)
        
    def forward(self, examples):
        args = self.args
        # now should implement the examples
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask

        if args.bert == -1:
            src_encodings, (last_state, last_cell) = self.encode(batch.src_sents, batch.src_sents_len, None)

            src_encodings = self.dropout(src_encodings)

            table_embedding = self.gen_x_batch(batch.table_sents)
            src_embedding = self.gen_x_batch(batch.src_sents)
            schema_embedding = self.gen_x_batch(batch.table_names)
            # get emb differ
            embedding_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=table_embedding,
                                                     table_unk_mask=batch.table_unk_mask)

            schema_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=schema_embedding,
                                                  table_unk_mask=batch.schema_token_mask)

            tab_ctx = (src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
            schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

            table_embedding = table_embedding + tab_ctx

            schema_embedding = schema_embedding + schema_ctx

            col_type = self.input_type(batch.col_hot_type)

            col_type_var = self.col_type(col_type)

            tab_type = self.input_type(batch.tab_hot_type)

            tab_type_var = self.tab_type(tab_type)

            table_embedding = table_embedding + col_type_var

            schema_embedding = schema_embedding + tab_type_var
        else:
            src_encodings, table_embedding, schema_embedding, last_cell = self.transformer_encode(batch, examples)
            # src_lens = [len(one) for one in src_encodings]
            # table_lens = [len(one) for one in table_embedding]
            # schema_lens = [len(one) for one in schema_embedding]
            #
            # cat_embeddings = [one_src + one_table + one_schema for one_src, one_table, one_schema in zip(src_encodings, table_embedding, schema_embedding)]
            # cat_lens = [len(one) for one in cat_embeddings]
            #
            # for b in range(len(cat_embeddings)):
            #     cat_embeddings[b] += [torch.zeros_like(cat_embeddings[b][0])] * (max(cat_lens) - len(cat_embeddings[b]))
            #     cat_embeddings[b] = torch.stack(cat_embeddings[b])
            # cat_embeddings = torch.stack(cat_embeddings)
            # concat_encodings = torch.cat((last_cell.unsqueeze(1), cat_embeddings), dim=1)
            # rat_encodings = concat_encodings
            # concat_encodings = concat_encodings.transpose(0, 1)
            # rat_encodings = self.rat_encoder(concat_encodings, batch.relation).transpose(0, 1)
            #
            # last_cell = rat_encodings[:,0,:]
            # src_encodings = []
            # table_embedding = []
            # schema_embedding = []
            # for b, (src_len, table_len, schema_len) in enumerate(zip(src_lens, table_lens, schema_lens)):
            #     src_encodings.append(rat_encodings[b, 1:1 + src_len, :])
            #     table_embedding.append(rat_encodings[b, 1+src_len:1+src_len+table_len, :])
            #     schema_embedding.append(rat_encodings[b, 1+src_len+table_len:1+src_len+table_len+schema_len,:])
            # for b in range(len(src_lens)):
            #     if src_lens[b] < max(src_lens):
            #         padding = torch.zeros((max(src_lens) - src_lens[b], args.hidden_size), dtype=src_encodings[b].dtype, device=src_encodings[b].device)
            #         src_encodings[b] = torch.cat((src_encodings[b], padding), dim=0)
            #     if table_lens[b] < max(table_lens):
            #         padding = torch.zeros((max(table_lens) - table_lens[b], args.hidden_size), dtype=table_embedding[b].dtype, device=table_embedding[b].device)
            #         table_embedding[b] = torch.cat((table_embedding[b], padding), dim=0)
            #     if schema_lens[b] < max(schema_lens):
            #         padding = torch.zeros((max(schema_lens) - schema_lens[b], args.hidden_size), dtype=schema_embedding[b].dtype, device=schema_embedding[b].device)
            #         schema_embedding[b] = torch.cat((schema_embedding[b], padding), dim=0)
            #
            # src_encodings = torch.stack(src_encodings)
            # table_embedding = torch.stack(table_embedding)
            # schema_embedding = torch.stack(schema_embedding)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec
        action_probs = [[] for _ in examples]

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        zero_type_embed = Variable(self.new_tensor(args.type_embed_size).zero_())

        sketch_attention_history = list()

        for t in range(batch.max_sketch_num):
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.sketch_decoder_lstm.input_size).zero_(),
                             requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, example in enumerate(examples):

                    if t < len(example.sketch):
                        # get the last action
                        # This is the action embedding
                        action_tm1 = example.sketch[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order]:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        else:
                            print(action_tm1, 'only for sketch')
                            quit()
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, example in enumerate(examples):
                    if t < len(example.sketch):
                        action_tm = example.sketch[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstm,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)
            sketch_attention_history.append(att_t)

            # get the Root possibility
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            for e_id, example in enumerate(examples):
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]

        h_tm1 = dec_init_vec

        for t in range(batch.max_action_num):
            if t == 0:
                # x = self.lf_begin_vec.unsqueeze(0).repeat(len(batch), 1)
                x = Variable(self.new_tensor(len(batch), self.lf_decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order,
                                                ]:

                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]

                        else:
                            if isinstance(action_tm1, define_rule.C):
                                a_tm1_embed = self.column_rnn_input(table_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.column_rnn_input(schema_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.A):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            else:
                                print(action_tm1, 'not implement')
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                # tgt t-1 action type
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm = example.tgt_actions[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)
            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
            else:
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)

            weights.data.masked_fill_(batch.table_token_mask, -float('inf'))

            column_attention_weights = F.softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask, -float('inf'))
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask, -float('inf'))

            table_weights = F.softmax(table_weights, dim=-1)
            # now get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        table_appear_mask[e_id, action_t.id_c] = 1
                        table_enable[e_id] = action_t.id_c
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [torch.stack(action_probs_i, dim=0).log().sum() for action_probs_i in action_probs], dim=0)

        return [sketch_prob_var, lf_prob_var]

    def transformer_encode(self, batch: Batch, examples):
        B = len(batch)
        sentences = batch.src_sents
        col_sets = batch.table_sents
        table_sets = batch.table_names_iter

        questions = []
        question_lens = []
        word_start_end_batch = []
        col_start_end_batch = []
        tab_start_end_batch = []
        col_types = []
        for b in range(B):
            word_start_ends = []
            question = "[CLS]"
            for word in sentences[b]:
                start = len(self.tokenizer.tokenize(question))
                for one_word in word:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                word_start_ends.append((start, end))
            col_start_ends = []
            for cols in col_sets[b]:
                start = len(self.tokenizer.tokenize(question))
                question += " [SEP]"
                for one_word in cols:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                col_start_ends.append((start, end))
            tab_start_ends = []
            for tabs in table_sets[b]:
                start = len(self.tokenizer.tokenize(question))
                question += " [SEP]"
                for one_word in tabs:
                    question += " " + one_word
                end = len(self.tokenizer.tokenize(question))
                tab_start_ends.append((start, end))
            if end >= self.tokenizer.max_len:
                print("xxxxxxxxxx")
                continue
            col_types.append(batch.col_hot_type[b])
            question_lens.append(end)
            questions.append(question)
            word_start_end_batch.append(word_start_ends)
            col_start_end_batch.append(col_start_ends)
            tab_start_end_batch.append(tab_start_ends)
            for st, ed in word_start_ends:
                for insert_id in range(st + 1, ed):
                    examples[b].relation = np.insert(examples[b].relation, insert_id, examples[b].relation[st], axis=0)
            for st, ed in col_start_ends:
                for insert_id in range(st + 1, ed):
                    examples[b].relation = np.insert(examples[b].relation, insert_id, examples[b].relation[st], axis=0)
            for st, ed in tab_start_ends:
                for insert_id in range(st + 1, ed):
                    examples[b].relation = np.insert(examples[b].relation, insert_id, examples[b].relation[st], axis=0)
            for st, ed in word_start_ends:
                for insert_id in range(st + 1, ed):
                    examples[b].relation = np.insert(examples[b].relation, insert_id, examples[b].relation[:,st], axis=1)
            for st, ed in col_start_ends:
                for insert_id in range(st + 1, ed):
                    examples[b].relation = np.insert(examples[b].relation, insert_id, examples[b].relation[:,st], axis=1)
            for st, ed in tab_start_ends:
                for insert_id in range(st + 1, ed):
                    examples[b].relation = np.insert(examples[b].relation, insert_id, examples[b].relation[:,st], axis=1)
        relation_matrix = [examples[b].relation for b in range(B)]
        max_len = max([len(item) for item in relation_matrix])
        relation_matrix_padded = torch.zeros(len(relation_matrix), max_len, max_len, dtype=torch.long)
        for b_idx in range(len(relation_matrix)):
            length = len(relation_matrix[b_idx])
            relation_matrix_padded[b_idx, :length, :length] = torch.tensor(relation_matrix[b_idx])
        if torch.cuda.is_available():
            relation_matrix_padded = relation_matrix_padded.cuda()
        for idx, question_len in enumerate(question_lens):
            questions[idx] = questions[idx] + (" " + self.tokenizer.pad_token) * (max(question_lens) - question_len)
        encoded_questions = [self.tokenizer.encode(question, add_special_tokens=False) for question in questions]
        encoded_questions = torch.tensor(encoded_questions)
        if torch.cuda.is_available():
            encoded_questions = encoded_questions.cuda()
        embedding = self.bert_rat(encoded_questions, relation_matrix_padded)[0]
        src_encodings = []
        table_embedding = []
        schema_embedding = []
        for b in range(len(questions)):
            one_q_encodings = []
            for st, ed in word_start_end_batch[b]:
                sum_tensor = torch.zeros_like(embedding[b][st])
                for i in range(st, ed):
                    sum_tensor = sum_tensor + embedding[b][i]
                sum_tensor = sum_tensor / (ed - st)
                one_q_encodings.append(sum_tensor)
            src_encodings.append(one_q_encodings)
            one_col_encodings = []
            for st, ed in col_start_end_batch[b]:
                inputs = embedding[b, st:ed].unsqueeze(0)
                lstm_out = self.col_lstm(inputs)[0].view(ed - st, 2, self.transformer_dim // 2)
                col_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                one_col_encodings.append(col_encoding)
            table_embedding.append(one_col_encodings)
            one_tab_encodings = []
            for st, ed in tab_start_end_batch[b]:
                inputs = embedding[b, st:ed].unsqueeze(0)
                lstm_out = self.tab_lstm(inputs)[0].view(ed - st, 2, self.transformer_dim // 2)
                tab_encoding = torch.cat((lstm_out[-1, 0], lstm_out[0, 1]))
                one_tab_encodings.append(tab_encoding)
            schema_embedding.append(one_tab_encodings)
        # return src_encodings, table_embedding, schema_embedding, embedding[:,0,:]

        max_src_len = max([len(one_q_encodings) for one_q_encodings in src_encodings])
        max_col_len = max([len(one_col_encodings) for one_col_encodings in table_embedding])
        max_tab_len = max([len(one_tab_encodings) for one_tab_encodings in schema_embedding])
        for b in range(len(questions)):
            src_encodings[b] += [torch.zeros_like(src_encodings[b][0])] * (max_src_len - len(src_encodings[b]))
            src_encodings[b] = torch.stack(src_encodings[b])
            table_embedding[b] += [torch.zeros_like(table_embedding[b][0])] * (max_col_len - len(table_embedding[b]))
            table_embedding[b] = torch.stack(table_embedding[b])
            schema_embedding[b] += [torch.zeros_like(schema_embedding[b][0])] * (max_tab_len - len(schema_embedding[b]))
            schema_embedding[b] = torch.stack(schema_embedding[b])
        src_encodings = torch.stack(src_encodings)
        table_embedding = torch.stack(table_embedding)
        schema_embedding = torch.stack(schema_embedding)

        col_type = self.input_type(col_types)
        col_type_var = self.col_type(col_type)
        table_embedding = table_embedding + col_type_var

        return src_encodings, table_embedding, schema_embedding, embedding[:,0,:]

    def parse(self, examples, beam_size=5):
        """
        one example a time
        :param examples:
        :param beam_size:
        :return:
        """
        batch = Batch([examples], self.grammar, cuda=self.args.cuda)
        if self.args.bert == -1:
            src_encodings, (last_state, last_cell) = self.encode(batch.src_sents, batch.src_sents_len, None)

            src_encodings = self.dropout(src_encodings)

            table_embedding = self.gen_x_batch(batch.table_sents)
            src_embedding = self.gen_x_batch(batch.src_sents)
            schema_embedding = self.gen_x_batch(batch.table_names)
            # get emb differ
            embedding_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=table_embedding,
                                                     table_unk_mask=batch.table_unk_mask)

            schema_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=schema_embedding,
                                                  table_unk_mask=batch.schema_token_mask)

            tab_ctx = (src_encodings.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
            schema_ctx = (src_encodings.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)

            table_embedding = table_embedding + tab_ctx

            schema_embedding = schema_embedding + schema_ctx

            col_type = self.input_type(batch.col_hot_type)

            col_type_var = self.col_type(col_type)

            tab_type = self.input_type(batch.tab_hot_type)

            tab_type_var = self.tab_type(tab_type)

            table_embedding = table_embedding + col_type_var

            schema_embedding = schema_embedding + tab_type_var
        else:
            src_encodings, table_embedding, schema_embedding, last_cell = self.transformer_encode(batch, [examples])
            # src_lens = [len(one) for one in src_encodings]
            # table_lens = [len(one) for one in table_embedding]
            # schema_lens = [len(one) for one in schema_embedding]
            #
            # cat_embeddings = [one_src + one_table + one_schema for one_src, one_table, one_schema in
            #                   zip(src_encodings, table_embedding, schema_embedding)]
            # cat_lens = [len(one) for one in cat_embeddings]
            #
            # for b in range(len(cat_embeddings)):
            #     cat_embeddings[b] += [torch.zeros_like(cat_embeddings[b][0])] * (max(cat_lens) - len(cat_embeddings[b]))
            #     cat_embeddings[b] = torch.stack(cat_embeddings[b])
            # cat_embeddings = torch.stack(cat_embeddings)
            # concat_encodings = torch.cat((last_cell.unsqueeze(1), cat_embeddings), dim=1)
            # rat_encodings = concat_encodings
            # concat_encodings = concat_encodings.transpose(0, 1)
            # rat_encodings = self.rat_encoder(concat_encodings, batch.relation).transpose(0, 1)
            #
            # last_cell = rat_encodings[:, 0, :]
            # src_encodings = []
            # table_embedding = []
            # schema_embedding = []
            # for b, (src_len, table_len, schema_len) in enumerate(zip(src_lens, table_lens, schema_lens)):
            #     src_encodings.append(rat_encodings[b, 1:1 + src_len, :])
            #     table_embedding.append(rat_encodings[b, 1 + src_len:1 + src_len + table_len, :])
            #     schema_embedding.append(
            #         rat_encodings[b, 1 + src_len + table_len:1 + src_len + table_len + schema_len, :])
            # for b in range(len(src_lens)):
            #     if src_lens[b] < max(src_lens):
            #         padding = torch.zeros((max(src_lens) - src_lens[b], args.hidden_size), dtype=src_encodings[b].dtype,
            #                               device=src_encodings[b].device)
            #         src_encodings[b] = torch.cat((src_encodings[b], padding), dim=0)
            #     if table_lens[b] < max(table_lens):
            #         padding = torch.zeros((max(table_lens) - table_lens[b], args.hidden_size),
            #                               dtype=table_embedding[b].dtype, device=table_embedding[b].device)
            #         table_embedding[b] = torch.cat((table_embedding[b], padding), dim=0)
            #     if schema_lens[b] < max(schema_lens):
            #         padding = torch.zeros((max(schema_lens) - schema_lens[b], args.hidden_size),
            #                               dtype=schema_embedding[b].dtype, device=schema_embedding[b].device)
            #         schema_embedding[b] = torch.cat((schema_embedding[b], padding), dim=0)
            #
            # src_encodings = torch.stack(src_encodings)
            # table_embedding = torch.stack(table_embedding)
            # schema_embedding = torch.stack(schema_embedding)

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings)
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_cell)
        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=True)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(beams)
            exp_src_enconding = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(hyp_num,
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           1),
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           2))
            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_enconding,
                                             exp_src_encodings_sketch_linear, self.sketch_decoder_lstm,
                                             self.sketch_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                action_class = hyp.get_availableClass()
                if action_class in [define_rule.Root1,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.N,
                                    define_rule.Order]:
                    possible_productions = self.grammar.get_production(action_class)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]
                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    raise RuntimeError('No right action class')

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]
                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']
                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        # now get the sketch result
        completed_beams.sort(key=lambda hyp: -hyp.score)
        if len(completed_beams) == 0:
            return [[], []]

        sketch_actions = completed_beams[0].actions
        # sketch_actions = examples.sketch

        padding_sketch = self.padding_sketch(sketch_actions)

        batch_table_dict = batch.col_table_dict

        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(beams)

            # expand value
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(hyp_num,
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         1),
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         2))
            exp_table_embedding = table_embedding.expand(hyp_num, table_embedding.size(1),
                                                         table_embedding.size(2))

            exp_schema_embedding = schema_embedding.expand(hyp_num, schema_embedding.size(1),
                                                           schema_embedding.size(2))


            table_appear_mask = batch.table_appear_mask
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        a_tm1_embed = self.column_rnn_input(table_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.T):
                        a_tm1_embed = self.column_rnn_input(schema_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                             self.lf_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            table_appear_mask_val = torch.from_numpy(table_appear_mask)

            if self.args.cuda: table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
                # weights = weights + self.col_attention_out(exp_embedding_differ).squeeze()
            else:
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)
            # weights.data.masked_fill_(exp_col_pred_mask, -float('inf'))

            column_selection_log_prob = F.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)
            # table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0), src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask, -float('inf'))

            table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask, -float('inf'))

            table_weights = F.log_softmax(table_weights, dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.A:
                    possible_productions = self.grammar.get_production(define_rule.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': define_rule.A, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_selection_log_prob[hyp_id, col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': define_rule.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                                  'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']
                if action_type_str == define_rule.C:
                    col_id = hyp_meta_entry['col_id']
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = hyp_meta_entry['t_id']
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]

                beams = new_beams
                t += 1
            else:
                break

        completed_beams.sort(key=lambda hyp: -hyp.score)

        return [completed_beams, sketch_actions]

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = F.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = F.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

