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
from src.models.basic_model import BasicModel
from src.models.pointer_net import PointerNet
from src.rule import semQL as define_rule
from tree_lstm import Tree, BatchedTree, TreeLSTM
from src.rule import lf
from random import randrange, sample
from src.models.emtable_tree import EmtableTree, EmtableNode


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

        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size // 2, bidirectional=True,
                                    batch_first=True)

        input_dim = args.action_embed_size + \
                    args.att_vec_size  + \
                    args.type_embed_size
        # previous action
        # input feeding
        # pre type embedding

        self.hidden_size = args.hidden_size

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

        self.col_type = nn.Linear(4, args.col_embed_size)
        self.sketch_encoder = nn.LSTM(args.action_embed_size, args.action_embed_size // 2, bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id) + 1, args.action_embed_size)
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

        self.tree_lstm = TreeLSTM(args.hidden_size, args.hidden_size, 0.3, 'n_ary', 8)

        self.outer = nn.Sequential(
            nn.Linear(args.hidden_size * 3, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )

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

        table_embedding = table_embedding + col_type_var

        # TODO attention

        batched_tree, gold, emptabletrees = self.batch_to_punked_trees(batch, table_embedding, schema_embedding)
        encoded_tree = self.tree_lstm(batched_tree).get_hidden_state()

        mix1 = encoded_tree[:,0,:] - last_state
        mix2 = encoded_tree[:,0,:] * last_state
        mix3 = mix1.abs()

        mix = torch.cat((mix1, mix2, mix3), dim=-1)
        out = self.outer(mix).squeeze()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, gold)


        ###logging
        loging_out = torch.sigmoid(out)
        for b, tree in enumerate(emptabletrees):
            print("TREE: {}".format(tree))
            print("GOLD: {}".format(gold[b]))
            print("OUT: {}".format(loging_out[b]))
            print("")

        return torch.sum(loss)

    def batch_to_punked_trees(self, batch: Batch, col_embeddings, tab_embeddings):
        def punk_tree(self, root, col_embeddings, tab_embeddings):
            tree = Tree(self.hidden_size)
            def traverse_1(self, node):
                node.punk = False
                node_num = 1
                for child in node.children:
                    node_num += traverse_1(self, child)
                return node_num

            node_num = traverse_1(self, root)
            if node_num != 0:
                punk_num = randrange(node_num)
            else:
                punk_num = 0
            selected_punks = sample(range(node_num), punk_num)

            def traverse_2(self, parent, node, node_id, selected_punks):
                new_parent = node
                if node_id in selected_punks:
                    node.punk = True
                    if parent and parent.punk:
                        parent.children.remove(node)
                        parent.children += node.children
                        new_parent = parent

                node_id += 1
                for child in node.children:
                    node_id = traverse_2(self, new_parent, child, node_id, selected_punks)
                return node_id
            traverse_2(self, None, root, 0, selected_punks)

            def traverse_3(self, node, id):
                for child in node.children:
                    if child.punk:
                        new_id = tree.add_node(parent_id=id, tensor=self.production_embed.weight[-1])
                    else:
                        if isinstance(child, define_rule.C):
                            new_id = tree.add_node(parent_id=id, tensor=col_embeddings[child.id_c])
                        elif isinstance(child, define_rule.T):
                            new_id = tree.add_node(parent_id=id, tensor=tab_embeddings[child.id_c])
                        else:
                            new_id = tree.add_node(parent_id=id, tensor=self.production_embed.weight[self.grammar.prod2id[child.production]])
                    traverse_3(self, child, new_id)
            if root.punk:
                root_id = tree.add_node(parent_id=None, tensor=self.production_embed.weight[-1])
            else:
                root_id = tree.add_node(parent_id=None, tensor=self.production_embed.weight[self.grammar.prod2id[root.production]])
            traverse_3(self, root, root_id)

            emptable_tree = EmtableTree()
            emptable_tree.root.is_empty = root.punk
            emptable_tree.root.rule_type = type(root)
            if root.punk:
                emptable_tree.root.action = None
            else:
                emptable_tree.root.action = root
                emptable_tree.empty_nodes = set()

            def traverse_4(node, emptable_node):
                for child in node.children:
                    if child.punk:
                        is_empty = True
                        rule_type = type(child)
                        action = None
                    else:
                        is_empty = False
                        rule_type = type(child)
                        action = child
                    new_node = EmtableNode(is_empty, rule_type, action, node)
                    if is_empty:
                        emptable_tree.empty_nodes.add(new_node)
                    emptable_node.children.append(new_node)
                    traverse_4(child, new_node)
            traverse_4(root, emptable_tree.root)

            return tree, emptable_tree

        def example_to_punked_tree(self, example, col_embeddings, tab_embeddings):
            # root = lf.build_tree(example.truth_actions)
            return punk_tree(self, example.truth_actions[0], col_embeddings, tab_embeddings)

        def make_random_negative_tree(self, example, col_embeddings, tab_embeddings):
            # TODO consider - do we have to make similar negatives?
            root = define_rule.Root1(id_c=randrange(4))
            def random_make_tree(node):
                for action in node.get_next_action():

                    if action == define_rule.Root1:
                        max_c = 4
                    elif action == define_rule.Root:
                        max_c = 6
                    elif action == define_rule.N:
                        max_c = 5
                    elif action == define_rule.C:
                        max_c = example.col_num
                    elif action == define_rule.T:
                        max_c = example.table_len
                    elif action == define_rule.A:
                        max_c = 6
                    elif action == define_rule.Sel:
                        max_c = 1
                    elif action == define_rule.Filter:
                        max_c = 20
                    elif action == define_rule.Sup:
                        max_c = 2
                    elif action == define_rule.Order:
                        max_c = 2
                    else:
                        assert False

                    new_action = action(randrange(max_c), parent=node)
                    root.add_children(new_action)
                    random_make_tree(new_action)
            random_make_tree(root)
            # TODO more accurate check if new tree is same with original tree
            origin_root = example.truth_actions[0]
            origin_actions = []
            def gather_action(origin_actions, origin_root):
                origin_actions.append(origin_root.production)
                for child in origin_root.children:
                    gather_action(origin_actions, child)
            gather_action(origin_actions, origin_root)
            def check_if_exists(origin_actions, root):
                if not root.punk and root.production not in origin_actions:
                    return True
                for child in root.children:
                    if check_if_exists(origin_actions, child):
                        return True
                return False
            tree, emptabletree = punk_tree(self, root, col_embeddings, tab_embeddings)
            if check_if_exists(origin_actions, root):
                return tree, emptabletree
            return None, None


        gold = []
        trees = []
        emptabletrees = []
        for b in range(len(batch)):
            if randrange(100) < 50:
                tree, emptabletree = example_to_punked_tree(self, batch.examples[b], col_embeddings[b], tab_embeddings[b])
                gold.append(1.)
            else:
                tree = None
                emptabletree = None
                while tree is None:
                    tree, emptabletree = make_random_negative_tree(self, batch.examples[b], col_embeddings[b], tab_embeddings[b])
                gold.append(0.)
            trees.append(tree)
            emptabletrees.append(emptabletree)

        gold = torch.Tensor(gold)
        if torch.cuda.is_available():
            gold = gold.cuda()
        return BatchedTree(trees), gold, emptabletrees

    def parse(self, example, beam_size=5):
        batch = Batch([example], self.grammar, cuda=self.args.cuda)

        table_appear_mask = batch.table_appear_mask


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

        table_embedding = table_embedding + col_type_var

        table_embedding = table_embedding.squeeze(0)
        schema_embedding = schema_embedding.squeeze(0)

        # TODO attention
        selected_tree = EmtableTree()
        while True:
            print("SELECTED : {}".format(selected_tree))
            possible_trees = selected_tree.possible_next_trees(example.table_len, example.col_num)

            if not possible_trees:
                break
            trees = []
            for possible_tree in possible_trees:
                tree = Tree(self.hidden_size)
                def traverse(self, node, id):
                    for child in node.children:
                        if child.is_empty:
                            new_id = tree.add_node(parent_id=id, tensor=self.production_embed.weight[-1])
                        else:
                            if child.rule_type == define_rule.C:
                                new_id = tree.add_node(parent_id=id, tensor=table_embedding[child.action.id_c])
                            elif child.rule_type == define_rule.T:
                                new_id = tree.add_node(parent_id=id, tensor=schema_embedding[child.action.id_c])
                            else:
                                new_id = tree.add_node(parent_id=id, tensor=self.production_embed.weight[
                                self.grammar.prod2id[child.action.production]])
                        traverse(self, child, new_id)

                if possible_tree.root.is_empty:
                    root_id = tree.add_node(parent_id=None, tensor=self.production_embed.weight[-1])
                else:
                    root_id = tree.add_node(parent_id=None, tensor=self.production_embed.weight[
                        self.grammar.prod2id[possible_tree.root.action.production]])
                traverse(self, possible_tree.root, root_id)
                trees.append(tree)
            batched_tree = BatchedTree(trees)
            encoded_tree = self.tree_lstm(batched_tree).get_hidden_state()

            mix1 = encoded_tree[:, 0, :] - last_state
            mix2 = encoded_tree[:, 0, :] * last_state
            mix3 = mix1.abs()

            mix = torch.cat((mix1, mix2, mix3), dim=-1)
            out = self.outer(mix).squeeze()
            max_tree = np.argmax(out.data.cpu().numpy())

            for en, tree in enumerate(possible_trees):
                print("POSSIBLE: {} / {}".format(tree, out.data.cpu().numpy()[en]))
            print("")
            print("")
            selected_tree = possible_trees[max_tree]
        print("TOTAL SELECTED: {}".format(selected_tree))
        print("\n\n\n\n")

        return selected_tree

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

