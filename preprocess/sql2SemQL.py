# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : sql2SemQL.py
# @Software: PyCharm
"""

import argparse
import json
import sys

import copy
from utils import load_dataSets

sys.path.append("..")
from src.rule.semQL import Root1, Root, N, C, T, Sel, Filter


class Parser:
    def __init__(self):
        self.copy_selec = None
        self.sel_result = []
        self.colSet = set()

    def _init_rule(self):
        self.copy_selec = None
        self.colSet = set()

    def _parse_root(self, sql):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False
        if 'orderBy' in sql['sql']:
            sql['sql']['orderby'] = sql['sql']['orderBy']
        if 'groupBy' in sql['sql']:
            sql['sql']['groupby'] = sql['sql']['groupBy']

        if sql['sql']['limit'] == None:
            use_sup = False

        if sql['sql']['orderby'] == []:
            use_ord = False
        elif sql['sql']['limit'] != None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] != [] or \
                        sql['sql']['having'] != []:
            use_fil = True

        if use_fil and (use_sup or use_ord):
            result = [Root(0)]
            select = sql['sql']['select'][1]
            self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderby'][1][0][1][1]]))
            result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderby'][1][0][1][1]])))
            if sql['sql']['orderby'][1][0][1][1] == 0:
                result.append(self._parser_column0(sql, select))
            else:
                result.append(T(sql['col_table'][sql['sql']['orderby'][1][0][1][1]]))
            return result, ['FILTER', 'SEL']
        elif use_sup or use_ord:
            result = [Root(2)]
            select = sql['sql']['select'][1]
            self.colSet.add(sql['col_set'].index(sql['names'][sql['sql']['orderby'][1][0][1][1]]))
            result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderby'][1][0][1][1]])))
            if sql['sql']['orderby'][1][0][1][1] == 0:
                result.append(self._parser_column0(sql, select))
            else:
                result.append(T(sql['col_table'][sql['sql']['orderby'][1][0][1][1]]))
            return result, ['SEL']
        elif use_fil:
            return [Root(1)], ['FILTER', 'SEL']
        else:
            return [Root(3)], ['SEL']

    def _parser_column0(self, sql, select):
        """
        Find table of column '*'
        :return: T(table_id)
        """
        if len(sql['sql']['from']['table_units']) == 1:
            return T(sql['sql']['from']['table_units'][0][1])
        else:
            table_list = []
            for tmp_t in sql['sql']['from']['table_units']:
                if type(tmp_t[1]) == int:
                    table_list.append(tmp_t[1])
            table_set, other_set = set(table_list), set()
            for sel_p in select:
                if sel_p[1][1][1] != 0:
                    other_set.add(sql['col_table'][sel_p[1][1][1]])

            if len(sql['sql']['where']) == 1:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
            elif len(sql['sql']['where']) == 3:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
            elif len(sql['sql']['where']) == 5:
                other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
                other_set.add(sql['col_table'][sql['sql']['where'][4][2][1][1]])
            table_set = table_set - other_set
            if len(table_set) == 1:
                return T(list(table_set)[0])
            elif len(table_set) == 0 and sql['sql']['groupby'] != []:
                return T(sql['col_table'][sql['sql']['groupby'][0][1]])
            else:
                question = sql['question']
                self.sel_result.append(question)
                print('column * table error')
                return T(sql['sql']['from']['table_units'][0][1])

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        result.append(Sel(0))
        result.append(N(len(select) - 1))

        for sel in select:
            self.colSet.add(sql['col_set'].index(sql['names'][sel[1][1][1]]))
            result.append(C(sql['col_set'].index(sql['names'][sel[1][1][1]])))
            # now check for the situation with *
            if sel[1][1][1] == 0:
                result.append(self._parser_column0(sql, select))
            else:
                result.append(T(sql['col_table'][sel[1][1][1]]))
            if not self.copy_selec:
                self.copy_selec = [copy.deepcopy(result[-2]), copy.deepcopy(result[-1])]

        return result, None

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []
        # check the where
        if sql['sql']['where'] != [] and sql['sql']['having'] != []:
            result.append(Filter(0))

        if sql['sql']['where'] != []:
            # check the not and/or
            if len(sql['sql']['where']) == 1:
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
            elif len(sql['sql']['where']) == 3:
                result.append(Filter(0))
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
            else:
                if sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                elif sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'or':
                    result.append(Filter(0))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                elif sql['sql']['where'][1] == 'or' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(0))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                else:
                    result.append(Filter(0))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql['names'], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql['names'], sql))

        # check having
        if sql['sql']['having'] != []:
            result.extend(self.parse_one_condition(sql['sql']['having'][0], sql['names'], sql))
        return result, None

    def parse_one_condition(self, sql_condit, names, sql):
        result = []
        # check if V(root)
        nest_query = True
        if type(sql_condit[3]) != dict:
            nest_query = False

        if sql_condit[0] == True:
            if sql_condit[1] == 9:
                # not like only with values
                fil = Filter(1)
            elif sql_condit[1] == 8:
                # not in with Root
                fil = Filter(2)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else:
            # check for Filter (<,=,>,!=,between, >=,  <=, ...)
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if nest_query == False:
                    fil = Filter(1)
                else:
                    fil = Filter(2)
            elif sql_condit[1] == 9:
                fil = Filter(1)
            elif sql_condit[1] == 8:
                fil = Filter(2)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")

        result.append(fil)
        self.colSet.add(sql['col_set'].index(sql['names'][sql_condit[2][1][1]]))
        result.append(C(sql['col_set'].index(sql['names'][sql_condit[2][1][1]])))
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql_condit[2][1][1]]))

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
            nest_query['col_set'] = sql['col_set']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['keys'] = sql['keys']
            result.extend(self.parser(nest_query))

        return result

    def _parse_step(self, state, sql):

        if state == 'ROOT':
            return self._parse_root(sql)

        if state == 'SEL':
            return self._parse_select(sql)

        elif state == 'FILTER':
            return self._parse_filter(sql)

        else:
            raise NotImplementedError("Not the right state")

    def full_parse(self, query):
        sql = query['sql']
        nest_query = {}
        nest_query['names'] = query['names']
        nest_query['query_toks_no_value'] = ""
        nest_query['col_table'] = query['col_table']
        nest_query['col_set'] = query['col_set']
        nest_query['table_names'] = query['table_names']
        nest_query['question'] = query['question']
        nest_query['query'] = query['query']
        nest_query['keys'] = query['keys']

        if sql['intersect']:
            results = [Root1(0)]
            nest_query['sql'] = sql['intersect']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['union']:
            results = [Root1(0)]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['except']:
            results = [Root1(0)]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        results = [Root1(1)]
        results.extend(self.parser(query))

        return results

    def parser(self, query):
        stack = ["ROOT"]
        result = []
        while len(stack) > 0:
            state = stack.pop()
            step_result, step_state = self._parse_step(state, query)
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()

    parser = Parser()

    # loading dataSets
    datas, table = load_dataSets(args)
    processed_data = []

    for i, d in enumerate(datas):
        if len(datas[i]['sql']['select'][1]) > 5:
            continue
        r = parser.full_parse(datas[i])
        datas[i]['rule_label'] = " ".join([str(x) for x in r])
        processed_data.append(datas[i])

    print('Finished %s datas and failed %s datas' % (len(processed_data), len(datas) - len(processed_data)))
    with open(args.output, 'w', encoding='utf8') as f:
        f.write(json.dumps(processed_data))

