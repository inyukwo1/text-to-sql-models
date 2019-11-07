# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : data_process.py
# @Software: PyCharm
"""
import json
import argparse
import nltk
import os
import pickle
import sqlite3
from utils import symbol_filter, re_lemma, fully_part_header, group_header, partial_header, num2year, group_symbol, group_values, group_digital, group_db
from utils import AGG, wordnet_lemmatizer
from utils import load_dataSets
from pattern.en import lemma

def process_datas(datas, args):
    """
    :param datas:
    :param args:
    :return:
    """
    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        english_IsA = pickle.load(f)

    db_values = dict()

    with open('../data/tables.json') as f:
        schema_tables = json.load(f)
    schema_dict = dict()
    for one_schema in schema_tables:
        schema_dict[one_schema["db_id"]] = one_schema
        schema_dict[one_schema["db_id"]]["only_cnames"] = [c_name.lower() for tid, c_name in one_schema["column_names_original"]]
    # copy of the origin question_toks
    for d in datas:
        if 'origin_question_toks' not in d:
            d['origin_question_toks'] = d['question_toks']

    for entry in datas:
        db_id = entry['db_id']
        if db_id not in db_values:
            conn = sqlite3.connect("../data/database/{}/{}.sqlite".format(db_id, db_id))
            # conn.text_factory = bytes
            cursor = conn.cursor()

            schema = {}

            # fetch table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [str(table[0].lower()) for table in cursor.fetchall()]

            # fetch table info
            for table in tables:
                cursor.execute("PRAGMA table_info({})".format(table))
                schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
            col_value_set = dict()
            for table in tables:
                for col in schema[table]:
                    cursor.execute("SELECT \"{}\" FROM \"{}\"".format(col, table))
                    col_idx = schema_dict[db_id]["only_cnames"].index(col)
                    col = entry["names"][col_idx]
                    value_set = set()
                    try:
                        for val in cursor.fetchall():
                            if isinstance(val[0], str):
                                value_set.add(str(val[0].lower()))
                                value_set.add(lemma(str(val[0].lower())))

                    except:
                        print("not utf8 value")
                    if col in col_value_set:
                        col_value_set[col] |= value_set
                    else:
                        col_value_set[col] = value_set
            db_values[db_id] = col_value_set


        entry['question_toks'] = symbol_filter(entry['question_toks'])
        origin_question_toks = symbol_filter([x for x in entry['origin_question_toks'] if x.lower() != 'the'])
        question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in entry['question_toks'] if x.lower() != 'the']

        entry['question_toks'] = question_toks

        table_names = []
        table_names_pattern = []

        for y in entry['table_names']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))
            x = [re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        header_toks = []
        header_toks_list = []

        header_toks_pattern = []
        header_toks_list_pattern = []

        for y in entry['col_set']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        num_toks = len(question_toks)
        idx = 0
        tok_concol = []
        type_concol = []
        nltk_result = nltk.pos_tag(question_toks)
        while idx < num_toks:

            # check for aggregation
            end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
            if agg:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["agg"])
                idx = end_idx
                continue

            if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MORE'])
                idx += 1
                continue

            if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MOST'])
                idx += 1
                continue

            # fully header
            end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for table
            end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["table"])
                idx = end_idx
                continue

            # check for column
            end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for partial column
            end_idx, tname = partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["col"])
                idx = end_idx
                continue



            # string match for Time Format
            if num2year(question_toks[idx]):
                question_toks[idx] = 'year'
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

            def get_concept_result(toks, graph):
                for begin_id in range(0, len(toks)):
                    for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                        tmp_query = "_".join(toks[begin_id:r_ind])
                        if tmp_query in graph:
                            mi = graph[tmp_query]
                            for col in entry['col_set']:
                                if col in mi:
                                    return col

            end_idx, symbol = group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            end_idx, values, col = group_db(origin_question_toks, idx, num_toks, db_values[db_id])
            if values:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["db", col])
                idx = end_idx
                continue

            result = group_digital(question_toks, idx)
            if result is True:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["value"])
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            tok_concol.append([question_toks[idx]])
            type_concol.append(['NONE'])
            idx += 1
            continue

        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol
        entry['nltk_pos'] = nltk_result

    return datas


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()
    args.conceptNet = './conceptNet'

    # loading dataSets
    datas, table = load_dataSets(args)

    # process datasets
    process_result = process_datas(datas, args)

    with open(args.output, 'w') as f:
        json.dump(datas, f)


