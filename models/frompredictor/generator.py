import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Set
from models.frompredictor.constituency_tree import ConstituencyTree, SubtreeEncoder
from models.frompredictor.schema_encoder import SchemaEncoder
from models.frompredictor.ontology import Ontology
from commons.utils import run_lstm
from commons.embeddings.graph_utils import *
from commons.embeddings.bert_container import BertContainer
from commons.embeddings.word_embedding import WordEmbedding
from commons.utils import run_lstm, col_tab_name_encode, seq_conditional_weighted_num, SIZE_CHECK
from datasets.sql import *
import nltk
from pytorch_pretrained_bert import BertModel


def truncated_normal_(tensor, mean=0., std=1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class GeneratorWrapper:
    def __init__(self, H_PARAM):
        self.generator = Generator(H_PARAM)
        self.acc_num = 1
        self.acc = 99999

        self.N_h = H_PARAM['N_h']
        self.N_depth = H_PARAM['N_depth']
        self.toy = H_PARAM['toy']
        self.gpu = H_PARAM['gpu']

        self.B_word = H_PARAM['B_WORD']
        self.N_word = H_PARAM['N_WORD']

        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        self.embed_layer = WordEmbedding(H_PARAM['glove_path'].format(self.B_word, self.N_word),
                                         self.N_word, gpu=self.gpu, SQL_TOK=self.SQL_TOK, use_bert=True, use_small=H_PARAM["toy"])

        self.save_dir = H_PARAM['save_dir']

    def load_model(self):
        if self.toy:
            return
        print('Loading from model...')
        dev_type = 'cuda' if self.gpu else 'cpu'
        device = torch.device(dev_type)

        self.generator.load_state_dict(torch.load(os.path.join(self.save_dir, "gen_models.dump"), map_location=device))
        self.generator.bert.load_state_dict(torch.load(os.path.join(self.save_dir, "genbert_from_models.dump"), map_location=device))

    def save_model(self, acc):
        print('tot_loss:{}'.format(acc), flush=True)

        if acc < self.acc:
            self.acc = acc
            torch.save(self.generator.state_dict(), os.path.join(self.save_dir, 'gen_models.dump'))
            torch.save(self.generator.bert.state_dict(), os.path.join(self.save_dir + "genbert_from_models.dump"))

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.bert_optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.bert_optimizer.step()

    def _select_tab_cols(self, schema, attention, label: Ontology):
        max_val, max_arg = torch.max(attention, dim=1)
        max_arg_numpy = max_arg.data.cpu().numpy()
        ontology = Ontology()
        selected_tabs = set()
        selected_cols = set()
        for entity_idx in max_arg_numpy:
            if entity_idx < schema.tab_num():
                ontology.tables.add(entity_idx)
                selected_tabs.add(entity_idx)
            elif entity_idx < schema.tab_num() + schema.col_num_except_star():
                col_id = entity_idx - schema.tab_num() + 1
                ontology.cols.add(col_id)
                selected_cols.add(col_id)

        tab_ids, col_ids = ontology.prune(schema)
        loss_tab_ids = label.tables - tab_ids
        loss_col_ids = label.cols - col_ids

        return selected_tabs, selected_cols, loss_tab_ids, loss_col_ids, ontology

    def loss(self, score, gt_data, subtrees):
        # TODO - [experiments]
        #          loss for only selected entity-subtree pairs <-
        #          loss for selected entity-subtree & gold
        #          loss for all without bridge entities
        #          loss for all

        # Shadowing
        # score: List[(subtree x entity)]
        # 1) 뽑아야되는데 아무도 안뽑음 -> 다같이 뽑게 함 (except 뭔가 뽑은 애)- TODO - [experiments] 어떤 값을 ground truth로 쓸 것인가?
        # 2) 안뽑아야되는거 -> 로스
        # 4) 뽑아야되는데 잘 뽑음 -> 더 잘 뽑게 로스 및 모르는거는 로스 x
        # TODO - 다음과 같은 로스를 주고 싶다 1) subtree 중 하나만 선택하도록 하는 loss 2) entity 중 하나만 선택하도록 하는 loss
        labels, schemas, questions, sqls = gt_data
        B = len(score)
        losses = []
        accs = []
        assert B == len(schemas)
        for b in range(B):
            attention = score[b]
            schema = schemas[b]
            label = labels[b]
            # gold = torch.zeros_like(attention)
            # for tab_id in label.tables:
            #     gold[tab_id] = 1.
            # for col_id in label.cols:
            #     gold[col_id - 1 + schema.tab_num()] = 1.
            # losses.append(F.binary_cross_entropy(attention, gold.detach()))
            # print(label)
            # entity_num = list(attention.size())[0]
            # print_att = attention.data.cpu().numpy()
            # gold = gold.cpu()
            # gold = gold.numpy()
            # for i in range(entity_num):
            #     print("   att[{}: {}]".format(i, print_att[i]))
            #     print("   gld[{}: {}]".format(i, gold[i]))
            #
            # continue

            selected_tabs, selected_cols, loss_tab_ids, loss_col_ids, ontology =\
                self._select_tab_cols(schema, attention, label)
            gold = torch.clone(attention).detach()
            subtree_num, entity_num = attention.size()
            max_val, max_arg = torch.max(attention, dim=1)
            max_arg_numpy = max_arg.data.cpu().numpy()
            assert entity_num == schema.tab_num() + schema.col_num_except_star() + 1
            # loss_applied = [[False] * entity_num] * subtree_num
            for tab_id in range(schema.tab_num()):
                assert tab_id < schema.tab_num()
                if tab_id not in selected_tabs:
                    for subtree_idx in range(subtree_num):
                        # if max_arg_numpy[subtree_idx] == entity_num - 1: # no-entity
                            gold[subtree_idx, tab_id] = 1.0
                            # loss_applied[subtree_idx][tab_id] = True
            for col_id in range(schema.col_num()):
                assert col_id < schema.col_num()
                if col_id not in selected_cols:
                    for subtree_idx in range(subtree_num):
                        # if max_arg_numpy[subtree_idx] == entity_num - 1:
                            gold[subtree_idx, col_id - 1 + schema.tab_num()] = 1.0
                            # loss_applied[subtree_idx][col_id - 1 + schema.tab_num()] = True
            for subtree_idx in range(subtree_num):
                for tab_id in range(schema.tab_num()):
                    entity_idx = tab_id
                    if tab_id not in label.tables:
                        gold[subtree_idx, entity_idx] = 0.
                        # gold[subtree_idx, entity_num - 1] = 1.
                        # loss_applied[subtree_idx][entity_idx] = True
                for col_id in range(schema.col_num()):
                    entity_idx = col_id - 1 + schema.tab_num()
                    if col_id not in label.cols:
                        gold[subtree_idx, entity_idx] = 0.
                        # gold[subtree_idx, entity_num - 1] = 1.
                        # loss_applied[subtree_idx][entity_idx] = True
            # for subtree_idx in range(subtree_num):
            #     for entity_idx in range(entity_num):
            #         if not loss_applied[subtree_idx][entity_idx]:
            #             attention[subtree_idx, entity_idx] = 0.11111
            #             gold[subtree_idx, entity_idx] = 0.11111
            for subtree_idx in range(subtree_num):
                selected = max_arg_numpy[subtree_idx]
                if selected < schema.tab_num():
                    if selected in loss_tab_ids:
                        gold[subtree_idx, selected] = 1.
                elif selected < schema.tab_num() + schema.col_num_except_star():
                    col_id = selected - schema.tab_num() + 1
                    if col_id in loss_col_ids:
                        gold[subtree_idx, col_id - 1 + schema.tab_num()] = 1.

            losses.append(F.binary_cross_entropy(attention, gold.detach()))
            accs.append(1. if ontology == label else 0.)
            if not self.generator.training:
                print("ACC: {}".format(ontology == label))
                print(questions[b])
                print(sqls[b])
                print(schema)
                print(" ")
                print(ontology)
                print(" ")
                print(label)
                print(" ")
                print_att = attention.data.cpu().numpy()
                gold = gold.cpu()
                gold = gold.numpy()
                for i in range(subtree_num):
                    print(subtrees[b][i].words)
                    for j in range(entity_num):
                        print("   att[{}: {}]".format(j, print_att[i, j]))
                        print("   gld[{}: {}]".format(j, gold[i, j]), flush=True)
        return sum(losses), sum(accs)

    def evaluate(self, score, gt_data, batch=None, log=False):
        pass

    def preprocess(self, batch):
        q_seq = []
        ontologies = []
        schemas = []
        sqls = []
        trees = []

        for item in batch:
            ontologies.append(item['ontology'])
            q_seq.append(item['nltk_question_toks'])
            schemas.append(item['schema'])
            sqls.append(item['query'])
            trees.append(item["parsetree"])

        q_embs, q_len, q_ranges_list, table_ranges_list, column_ranges_list, new_schemas, parse_trees, new_ontologies, new_questions, new_sqls = self.embed_layer.gen_for_generator(q_seq, schemas, ontologies, sqls, trees)

        input_data = q_embs, q_len, q_ranges_list, table_ranges_list, column_ranges_list, new_schemas, parse_trees
        gt_data = (new_ontologies, new_schemas, new_questions, new_sqls)

        # return None, None
        return input_data, gt_data

    def forward(self, input_data):
        q_embs, q_len, q_ranges_list, table_ranges_list, column_ranges_list, new_schemas, parse_trees = input_data
        return self.generator.forward(q_embs, q_len, q_ranges_list, table_ranges_list, column_ranges_list, new_schemas, parse_trees)

    def train(self, mode=True):
        self.generator.train(mode)

    def eval(self):
        self.generator.train(False)

    def generator_params(self):
        for param in self.generator.constituency_encoder.parameters():
            yield param
        for param in self.generator.schema_encoder.parameters():
            yield param

    def bert_params(self):
        for param in self.generator.bert.parameters():
            yield param


class Generator(nn.Module):
    def __init__(self, H_PARAM):
        super(Generator, self).__init__()
        self.acc_num = 1
        self.acc = 99999

        self.N_h = H_PARAM['N_h']
        self.N_depth = H_PARAM['N_depth']
        self.toy = H_PARAM['toy']
        self.gpu = H_PARAM['gpu']

        self.B_word = H_PARAM['B_WORD']
        self.N_word = H_PARAM['N_WORD']

        self.save_dir = H_PARAM['save_dir']
        self.bert = BertModel.from_pretrained('bert-large-cased')

        self.encoded_num = H_PARAM['encoded_num'] #1024
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        self.constituency_encoder = SubtreeEncoder(H_PARAM)
        self.schema_encoder = SchemaEncoder(H_PARAM)
        self.tagset = set()
        # TODO order-effective chooser
        # TODO pop with containing subtrees

        if self.gpu:
            self.cuda(0)

    def forward(self, q_embs, q_len,
                 q_ranges_list: List[List[Tuple[int, int]]],
                 table_ranges_list: List[List[Tuple[int, int]]],
                 column_ranges_list: List[List[Tuple[int, int]]], schemas: List[Schema], parse_trees: List[nltk.Tree], single_forward=False):
        B = len(q_len)
        q_enc = self.bert(q_embs, output_all_encoded_layers=False)[0]
        batch_attentions = []
        batch_subtrees = []
        for b in range(B):
            word_vectors = []
            for word_st, word_ed in q_ranges_list[b]:
                # TODO - [experiments]
                #          sum <-
                #          avg
                #          lstm
                word_vectors.append(torch.sum(q_enc[b, word_st:word_ed, :], dim=0))
            constituency_tree = ConstituencyTree(parse_trees[b], word_vectors)
            encoded_subtree = []
            subtrees = []
            for subtree in constituency_tree.gen_subtrees():
                encoded_subtree.append(self.constituency_encoder(subtree))
                subtrees.append(subtree)
            batch_subtrees.append(subtrees)

            table_vectors = []
            for table_st, table_ed in table_ranges_list[b]:
                table_vectors.append(torch.sum(q_enc[b, table_st:table_ed, :], dim=0))
            col_vectors = []
            for col_st, col_ed in column_ranges_list[b]:
                col_vectors.append(torch.sum(q_enc[b, col_st:col_ed, :], dim=0))
            encoded_tables, encoded_cols, no_entry = self.schema_encoder(schemas[b], table_vectors, col_vectors)
            encoded_subtree = torch.stack(encoded_subtree)
            encoded_entity = torch.cat((encoded_tables, encoded_cols, no_entry.unsqueeze(0)), 0)
            attention = torch.mm(encoded_subtree, encoded_entity.transpose(0, 1))
            # (subtree x entity)
            attention = torch.softmax(attention, dim=1)
            # attention = torch.nn.functional.sigmoid(attention)
            # attention = torch.sum(attention, dim=0)
            # attention = torch.sigmoid(attention)
            batch_attentions.append(attention)

        return batch_attentions, batch_subtrees

