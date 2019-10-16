from copy import deepcopy
from src.rule import semQL as define_rule
import itertools


def partitioning(some_list, partition_cardinal):
    bars = [0] * (partition_cardinal - 1)
    def make_partition_using_bars(some_list, bars):
        partition = []
        bar_st = 0
        for bar in bars:
            partition.append(some_list[bar_st:bar])
        if bars:
            partition.append(some_list[bars[-1]:])
        else:
            partition.append(some_list)
        return partition

    def gen_next_bar_partition(bars, cur_idx, max_bar_pos):
        if cur_idx == len(bars):
            yield bars
            return
        for i in range(bars[cur_idx], max_bar_pos + 1):
            bars[cur_idx:] = [i] * (len(bars) - cur_idx)
            yield from gen_next_bar_partition(bars, cur_idx + 1, max_bar_pos)
    for new_bar in gen_next_bar_partition(bars, 0, len(some_list)):
        yield make_partition_using_bars(some_list, new_bar)


def all_possible_actions(tab_num, col_num):
    for rule in [define_rule.Root1, define_rule.Root, define_rule.Filter, define_rule.Order, define_rule.Sup, define_rule.Sel, define_rule.N, define_rule.A]:
        for id_c in range(rule.id_c_num):
            yield rule(id_c)
    for tab in range(tab_num):
        yield define_rule.T(tab)
    for col in range(col_num):
        yield define_rule.C(col)


class EmtableNode:
    def __init__(self, is_empty=True, rule_type=None, action=None, parent=None):
        self.is_empty = is_empty
        self.rule_type = rule_type
        self.action = action
        self.parent = parent
        self.children = []


class EmtableTree:
    def __init__(self):
        self.root = EmtableNode(True, define_rule.Root1)
        self.empty_nodes = {self.root}

    def possible_next_trees(self, table_num, col_num):
        possible_trees = []
        for empty_node in self.empty_nodes:
            # top-down
            if empty_node.rule_type == define_rule.C:
                max_id_c = col_num
            elif empty_node.rule_type == define_rule.T:
                max_id_c = table_num
            else:
                max_id_c = empty_node.rule_type.id_c_num
            for id_c in range(max_id_c):
                action = empty_node.rule_type(id_c)
                empty_node.is_empty = False
                empty_node.action = action
                self.empty_nodes.remove(empty_node)

                child_rules = action.get_child_rules()
                for partition in partitioning(empty_node.children, len(child_rules)):
                    def gen_empty_tree(possible_trees, child_rules, current_tree, partition, handling_part, empty_node, new_children):
                        if handling_part == len(child_rules):
                            empty_node.children = new_children
                            possible_trees.append(deepcopy(current_tree))
                            return
                        try:
                            if len(partition[handling_part]) == 1 and partition[handling_part][0].rule_type == child_rules[handling_part]:
                                new_children[handling_part] = partition[handling_part][0]
                                gen_empty_tree(possible_trees, child_rules, current_tree, partition, handling_part + 1, empty_node, new_children)
                        except Exception as e:
                            print("x")
                        new_empty = EmtableNode(is_empty=True, rule_type=child_rules[handling_part], action=None, parent=empty_node)
                        new_children[handling_part] = new_empty
                        new_empty.children = partition[handling_part]
                        current_tree.empty_nodes.add(new_empty)
                        gen_empty_tree(possible_trees, child_rules, current_tree, partition, handling_part + 1, empty_node, new_children)
                        current_tree.empty_nodes.remove(new_empty)
                    gen_empty_tree(possible_trees, child_rules, self, partition, 0, empty_node, [None] * len(child_rules))
                empty_node.is_empty = True
                empty_node.action = None
                self.empty_nodes.add(empty_node)
            # mid-out
            for action in all_possible_actions(table_num, col_num):
                new_node = EmtableNode(False, type(action), action, empty_node)
                new_node.children = empty_node.children
                empty_node.children = [new_node]
                self.empty_nodes.add(new_node)

                child_rules = action.get_child_rules()
                for partition in partitioning(new_node.children, len(child_rules)):
                    def gen_empty_tree(possible_trees, child_rules, current_tree, partition, handling_part, new_node, new_children):
                        if handling_part == len(child_rules):
                            new_node.children = new_children
                            possible_trees.append(deepcopy(current_tree))
                            return
                        if len(partition[handling_part]) == 1 and partition[handling_part][0].rule_type == child_rules[handling_part]:
                            new_children[handling_part] = partition[handling_part][0]
                            gen_empty_tree(possible_trees, child_rules, current_tree, partition, handling_part + 1, new_node, new_children)
                        new_empty = EmtableNode(is_empty=True, rule_type=child_rules[handling_part], action=None, parent=new_node)
                        new_children[handling_part] = new_empty
                        new_empty.children = partition[handling_part]
                        current_tree.empty_nodes.add(new_empty)
                        gen_empty_tree(possible_trees, child_rules, current_tree, partition, handling_part + 1, new_node, new_children)
                        current_tree.empty_nodes.remove(new_empty)
                    gen_empty_tree(possible_trees, child_rules, self, partition, 0, new_node, [None] * len(child_rules))

                self.empty_nodes.remove(new_node)
            # add children
            for action in all_possible_actions(table_num, col_num):
                new_node = EmtableNode(False, type(action), action, empty_node)
                self.empty_nodes.add(new_node)
                for child_rule in action.get_child_rules():
                    new_node.children.append(EmtableNode(True, None, None, new_node))
                for child_idx in range(len(empty_node.children) + 1):
                    empty_node.children.insert(child_idx, new_node)
                    possible_trees.append(deepcopy(self))
                    empty_node.children.remove(new_node)
                self.empty_nodes.remove(new_node)

        # TODO remove unreachable tree





