import torch
import torch.nn
import nltk
import benepar
import itertools
import functools
from typing import List, Tuple
from collections import OrderedDict
from torch.nn import GRUCell


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
    def __init__(self, leaves: List[Tuple[torch.Tensor, ConstituencyNode]], words: List[str]):
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
                    tensor = sentence_tensors[len(self.leaves)].unsqueeze(0)
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
        #          using only one gru
        #          various gru
        #          something else..(search tree-encoder)?
        tags = {'JJR', 'CD', '<STOP>', 'JJ', 'FW', 'NP', 'MD', 'FRAG', 'SINV', 'ADJP', 'WHADJP', 'VB', 'CC', 'PRT', 'NX', ':', 'WHNP', 'IN', 'NN', 'VBP', "''", '$', 'WHPP', 'JJS', 'SQ', 'TO', 'RP', 'RBS', 'UCP', 'PP', 'PRP$', 'PRN', 'WDT', 'NNS', 'RBR', 'X', 'CONJP', 'SBAR', '``', 'NNP', 'VBG', 'WP', 'DT', 'PRP', 'SBARQ', ',', '#', 'EX', 'WHADVP', 'RB', 'NNPS', 'VP', 'ADVP', 'VBD', '-LRB-', 'WP$', '-RRB-', 'UH', '.', 'WRB', 'INTJ', 'S', 'VBZ', 'QP', 'VBN', 'POS', 'PDT'}
        self.gru_cell = GRUCell(H_PARAM["encoded_num"], H_PARAM["encoded_num"])
        self.param_dict = torch.nn.ParameterDict()
        for st_tag in tags:
            st_tag = st_tag.replace(".", "dot")
            for ed_tag in tags:
                ed_tag = ed_tag.replace(".", "dot")
                new_param = torch.nn.Parameter(torch.randn(1, H_PARAM["encoded_num"]))
                new_param.requires_grad = True
                self.param_dict[st_tag + "_" + ed_tag] = new_param
        self.outer = torch.nn.Sequential(
            torch.nn.Linear(H_PARAM["encoded_num"], H_PARAM["encoded_num"]),
            torch.nn.Sigmoid()
        )

    def forward(self, subtree: ConstituencySubtree):
        # just sum
        # summed = 0
        # leaves = []
        # for leav in subtree.leaves:
        #     leaves.append(leav[0])
        # leaves = torch.stack(leaves)
        # return torch.sum(self.outer(leaves), dim=0).squeeze()
        # various gru
        while not subtree.only_root():
            leaf_idx = subtree.get_one_leaf()
            leaf_tensor, leaf_node = subtree.leaves[leaf_idx]
            leaf_tag = leaf_node.tag
            parent_tag = leaf_node.parent.tag
            leaf_tag = leaf_tag.replace(".", "dot")
            parent_tag = parent_tag.replace(".", "dot")
            new_tensor = self.gru_cell(self.param_dict[leaf_tag + "_" + parent_tag], leaf_tensor)
            subtree.update_leaf(leaf_idx, new_tensor)
        return self.outer(subtree.leaves[0][0]).squeeze(0)
