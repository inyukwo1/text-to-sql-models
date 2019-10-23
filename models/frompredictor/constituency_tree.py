import torch
import torch.nn
import nltk
import benepar
import itertools
import functools
from typing import List, Tuple
from collections import OrderedDict
from torch.nn import GRUCell
from tree_lstm import Tree, TreeLSTM, BatchedTree


class ConstituencyNode:
    def __init__(self, parent, tag, level):
        self.parent: ConstituencyNode = parent
        self.children = []
        self.tag = tag
        self.level = level

    def update_parent_and_pop_self(self, new_tensor):
        self.parent.tensor = self.parent.tensor + new_tensor
        self.parent.children.remove(self)


# TODO make test!
class ConstituencySubtree:
    def __init__(self, leaves: List[Tuple], words: List[str]):
        self.leaves = leaves
        self.words = words

    def get_one_leaf(self) -> int:
        max_level_node_getter = lambda x, y: x if x[1].level > y[1].level else y
        max_lev = -1
        max_idx = -1
        for idx, leaf in enumerate(self.leaves):
            if max_lev < leaf[1].level:
                max_lev = leaf[1].level
                max_idx = idx
        return max_idx

    def update_leaf(self, origin_leaf_idx, new_tensor):
        _, node = self.leaves[origin_leaf_idx]
        for idx, (ex_tensor, ex_node) in enumerate(self.leaves):
            if node.parent is ex_node:
                self.leaves[idx] = (ex_tensor + new_tensor, ex_node)
                self.leaves.pop(origin_leaf_idx)
                return
        self.leaves.append((new_tensor, node.parent))
        self.leaves.pop(origin_leaf_idx)

    def only_root(self):
        return len(self.leaves) == 1


class ConstituencyTree:
    def __init__(self, tree: nltk.Tree, sentence_tensors: List[torch.Tensor]):
        assert len(tree.leaves()) == len(sentence_tensors)
        self.root: ConstituencyNode = ConstituencyNode(None, tree.label(), 0)
        self.leaves = []
        self.words = []

        def search_children_and_add(parent: ConstituencyNode, tree: nltk.Tree, level):
            for child in tree:
                if isinstance(child, str):
                    tensor = sentence_tensors[len(self.leaves)]
                    self.leaves.append((tensor, parent))
                    self.words.append(child)
                else:
                    new_node = ConstituencyNode(parent, child.label(), level + 1)
                    search_children_and_add(new_node, child, level + 1)

        search_children_and_add(self.root, tree, 0)

    def gen_subtrees(self):
        # TODO - [experiments]
        #          LEAVE_NUM = 1
        #          LEAVE_NUM = 2
        #          LEAVE_NUM = 3 <--
        #          ...
        LEAVE_NUM = 3
        leave_len = len(self.leaves)
        for leav_num in range(1, LEAVE_NUM):
            for selected_leaves_id in itertools.combinations(range(leave_len), leav_num):
                selected_leaves = [self.leaves[i] for i in selected_leaves_id]
                selected_words = [self.words[i] for i in selected_leaves_id]
                yield ConstituencySubtree(list(selected_leaves), selected_words)


class SubtreeEncoder(torch.nn.Module):
    def __init__(self, H_PARAM):
        super(SubtreeEncoder, self).__init__()
        # TODO - [experiments]
        #          just sum <--
        #          using only .one gru
        #          various gru
        #          something else..(search tree-encoder)?
        tags = {'JJR', 'CD', '<STOP>', 'JJ', 'FW', 'NP', 'MD', 'FRAG', 'SINV', 'ADJP', 'WHADJP', 'VB', 'CC', 'PRT', 'NX', ':', 'WHNP', 'IN', 'NN', 'VBP', "''", '$', 'WHPP', 'JJS', 'SQ', 'TO', 'RP', 'RBS', 'UCP', 'PP', 'PRP$', 'PRN', 'WDT', 'NNS', 'RBR', 'X', 'CONJP', 'SBAR', '``', 'NNP', 'VBG', 'WP', 'DT', 'PRP', 'SBARQ', ',', '#', 'EX', 'WHADVP', 'RB', 'NNPS', 'VP', 'ADVP', 'VBD', '-LRB-', 'WP$', '-RRB-', 'UH', '.', 'WRB', 'INTJ', 'S', 'VBZ', 'QP', 'VBN', 'POS', 'PDT'}
        self.param_dict = torch.nn.ParameterDict()
        for st_tag in tags:
            st_tag = st_tag.replace(".", "dot")
            for ed_tag in tags:
                ed_tag = ed_tag.replace(".", "dot")
                new_param = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
                new_param.requires_grad = True
                self.param_dict[st_tag + "_" + ed_tag] = new_param

        new_param = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        new_param.requires_grad = True
        self.param_dict["root"] = new_param
        self.n_h = H_PARAM["encoded_num"]
        self.tree_lstm = TreeLSTM(self.n_h, self.n_h, 0.3, cell_type='n_ary', n_ary=5)


    def to_batched_tree(self, subtree: ConstituencySubtree):
        tree = Tree(self.n_h)
        root_id = tree.add_node(None, self.param_dict["root"])
        added_nodes = {}
        while not subtree.only_root():
            leaf_idx = subtree.get_one_leaf()
            tensor, leaf = subtree.leaves[leaf_idx]
            childrens_ids = []
            for child in leaf.children:
                if child in added_nodes:
                    childrens_ids.append(added_nodes[child])
            new_id = tree.add_node_bottom_up(childrens_ids, tensor)
            added_nodes[leaf] = new_id
            if leaf.parent in added_nodes:
                tree.add_link(new_id, added_nodes[leaf.parent])
            leaf_tag = leaf.tag.replace(".", "dot")
            parent_tag = leaf.parent.tag.replace(".", "dot")
            tag_tensor = self.param_dict[leaf_tag + "_" + parent_tag]
            subtree.update_leaf(leaf_idx, tag_tensor)
        tensor, leaf = subtree.leaves[0]
        childrens_ids = []
        for child in leaf.children:
            if child in added_nodes:
                childrens_ids.append(added_nodes[child])

        new_id = tree.add_node_bottom_up(childrens_ids, tensor)
        tree.add_link(new_id, root_id)
        return BatchedTree([tree])


    def forward(self, subtree: ConstituencySubtree):
        batched_tree = self.to_batched_tree(subtree)
        return self.tree_lstm.forward(batched_tree).get_hidden_state().squeeze(0)[0]
