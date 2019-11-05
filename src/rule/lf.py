# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""
import copy
import json

import numpy as np

from src.rule import semQL as define_rule
from src.rule.semQL import C, T, Root1

def build_tree(lf):
    prev_root = None
    while lf:
        root = lf.pop(0)
        assert isinstance(root, define_rule.Root1)
        column = lf.pop(0)
        assert isinstance(column, define_rule.C)
        if len(lf) == 0:
            table = None
        else:
            table = lf.pop(0)
            if not isinstance(table, define_rule.T):
                lf.insert(0, table)
                table = None

        root.add_children(column)
        column.set_parent(root)
        if table is not None:
            column.add_children(table)
            table.set_parent(column)
        if prev_root:
            prev_root.add_children(root)
            root.set_parent(prev_root)
        prev_root = root
    verify(prev_root)
    # eliminate_parent(root)


def eliminate_parent(node):
    for child in node.children:
        eliminate_parent(child)
    node.children = list()


def verify(node):
    if isinstance(node, C) and len(node.children) > 0:
        table = node.children[0]
        assert table is None or isinstance(table, T)
    if isinstance(node, T):
        return
    children_num = len(node.children)
    if isinstance(node, Root1):
        if node.id_c == 0:
            assert children_num == 2
        else:
            assert children_num == 1
    for child in node.children:
        assert child.parent == node
        verify(child)


def label_matrix(lf, matrix, node):
    nindex = lf.index(node)
    for child in node.children:
        if child not in lf:
            continue
        index = lf.index(child)
        matrix[nindex][index] = 1
        label_matrix(lf, matrix, child)


def build_adjacency_matrix(lf, symmetry=False):
    _lf = list()
    for rule in lf:
        if isinstance(rule, C) or isinstance(rule, T):
            continue
        _lf.append(rule)
    length = len(_lf)
    matrix = np.zeros((length, length,))
    label_matrix(_lf, matrix, _lf[0])
    if symmetry:
        matrix += matrix.T
    return matrix


if __name__ == '__main__':
    with open(r'..\data\train.json', 'r') as f:
        data = json.load(f)
    for d in data:
        rule_label = [eval(x) for x in d['rule_label'].strip().split(' ')]
        print(d['question'])
        print(rule_label)
        build_tree(copy.copy(rule_label))
        adjacency_matrix = build_adjacency_matrix(rule_label, symmetry=True)
        print(adjacency_matrix)
        print('===\n\n')
