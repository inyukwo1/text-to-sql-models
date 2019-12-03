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
from relational_schema import RelationalSchema
from relation_types import RELATION_TYPE
import numpy
import re

def create_relation_matrix(data, schema, skip_q_indices):
    # question token - question token relation
    q_q_relation = parse_q_q_relation(data)
    q_q = numpy.array(q_q_relation)
    q_q = numpy.delete(q_q, skip_q_indices, 0)
    q_q = numpy.delete(q_q, skip_q_indices, 1)

    # question token - schema entity relation
    q_s_relation = parse_q_s_relation(data, schema)
    q_s = numpy.array(q_s_relation)
    q_s = numpy.delete(q_s, skip_q_indices, 0)

    # schema entity - question token relation
    s_q_relation = parse_s_q_relation(data, schema)
    s_q = numpy.array(s_q_relation)
    s_q = numpy.delete(s_q, skip_q_indices, 1)

    # schema entity - schema entity relation
    s_s_relation = schema.get_entity_relations()
    s_s = numpy.array(s_s_relation)

    # Concatenate
    tmp1 = numpy.concatenate((q_q, s_q))
    tmp2 = numpy.concatenate((q_s, s_s))
    relation_matrix = numpy.concatenate((tmp1, tmp2), 1)

    return relation_matrix


def parse_q_s_relation(data, schema):
    # need to fix this
    relations = []
    relations += [[RELATION_TYPE['cls_c']] * len(schema.get_col_names()) + [RELATION_TYPE['cls_t']] * len(schema.get_table_names())]
    for idx_1 in range(len(data['question_toks'])):
        tmp = []
        linked_entities = data['schema_linking'][idx_1]
        for idx_2 in range(schema.get_entity_num()):
            entity = schema.get_entity_original_name(idx_2)
            if entity in linked_entities:
                assert 'TABLE' in linked_entities[0] or 'COLUMN' in linked_entities[0]
                s1 = 'qt' if 'TABLE' in linked_entities[0] else 'qc'
                assert 'EXACT' in linked_entities[0] or 'PARTIAL' in linked_entities[0]
                s2 = 'exact' if 'EXACT' in linked_entities[0] else 'partial'
                key = '_'.join([s1, s2])
            else:
                assert 'qt_no' in RELATION_TYPE and 'qc_no' in RELATION_TYPE, 'key changed'
                key = 'qt_no' if '.' in entity else 'qc_no'
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def parse_s_q_relation(data, schema):
    # need to fix this
    relations = []
    for idx_1 in range(schema.get_entity_num()):
        tmp = []
        entity = schema.get_entity_original_name(idx_1)
        if idx_1 < len(schema.get_col_names()):
            tmp.append(RELATION_TYPE['c_cls'])
        else:
            tmp.append(RELATION_TYPE['t_cls'])
        for idx_2 in range(len(data['question_toks'])):
            linked_entities = data['schema_linking'][idx_2]
            if entity in linked_entities:
                assert 'TABLE' in linked_entities[0] or 'COLUMN' in linked_entities[0]
                s1 = 'tq' if 'TABLE' in linked_entities[0] else 'cq'
                assert 'EXACT' in linked_entities[0] or 'PARTIAL' in linked_entities[0]
                s2 = 'exact' if 'EXACT' in linked_entities[0] else 'partial'
                key = '_'.join([s1, s2])
            else:
                assert 'tq_no' in RELATION_TYPE and 'cq_no' in RELATION_TYPE, 'key changed'
                key = 'tq_no' if '.' in entity else 'cq_no'
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def parse_q_q_relation(data):
    # need to fix this
    relations = []
    relations += [[RELATION_TYPE['cls_cls']] + [RELATION_TYPE['cls_q']] * len(data['question_toks'])]
    question_length = len(data['question_toks'])
    for idx_1 in range(question_length):
        tmp = [RELATION_TYPE['q_cls']]
        for idx_2 in range(question_length):
            key = 'qq_' + str(max(min(idx_1-idx_2, 2), -2))
            tmp += [RELATION_TYPE[key]]
        relations += [tmp]
    return relations


def exact_match(s_idx, q_toks, schema_entities, keys, tmp_list):
    for endIdx in range(len(q_toks), s_idx, -1):
        sub_toks = q_toks[s_idx:endIdx]
        sub_toks = ' '.join(sub_toks)
        for entity_idx in range(len(schema_entities)):
            entity = schema_entities[entity_idx]
            if sub_toks == ' '.join(entity):
                # Insert match item to list
                for idx in range(s_idx, endIdx):
                    tmp_list[idx] += [keys[entity_idx]]

        if tmp_list[s_idx]:
            # Insert Type
            typ = '[EXACT COLUMN]' if '.' in tmp_list[s_idx][0] else '[EXACT TABLE]'
            for idx in range(s_idx, endIdx):
                tmp_list[idx].insert(0, typ)
            return endIdx+1
    return None


def partial_match(s_idx, q_toks, schema_entities, keys, tmp_list):
    for endIdx in range(len(q_toks), s_idx, -1):
        sub_toks = q_toks[s_idx:endIdx]
        for entity_idx in range(len(schema_entities)):
            entity = schema_entities[entity_idx]
            tmp_set = set(sub_toks) | set(entity)
            if tmp_set == set(entity):
                # Insert match item to list
                for idx in range(s_idx, endIdx):
                    tmp_list[idx] += [keys[entity_idx]]

        if tmp_list[s_idx]:
            # Insert Type
            typ = '[PARTIAL COLUMN]' if '.' in tmp_list[s_idx][0] else '[PARTIAL TABLE]'
            for idx in range(s_idx, endIdx):
                tmp_list[idx].insert(0, typ)
            return endIdx+1
    return None


def get_symbol(s_idx, q_toks):
    if q_toks[s_idx] == "'":
        for e_idx in range(s_idx+1, len(q_toks)):
            if q_toks[e_idx] == "'":
                return e_idx+1
    return None


def is_digit(value):
    value = value.replace(':', '').replace('.', '').replace('-', '').replace('$', '')
    return value.isdigit()


def search_knowledge_graph(toks, knowledge_graph):
    for s_idx in range(len(toks)):
        for e_idx in range(len(toks), -1, -1):
            sub_tok = '_'.join(toks[s_idx:e_idx])
            if sub_tok in knowledge_graph:
                return knowledge_graph[sub_tok]
    return None


def get_value_idx(s_idx, q_toks):
    if q_toks[s_idx] == "'":
        for e_idx in range(s_idx+1, len(q_toks)):
            if q_toks[e_idx] == "'":
                return e_idx+1
    return None


def match(value, schema_entities, s_idx, e_idx, tmp_list, is_table):
    tmp = [entity for entity in schema_entities if value == entity]
    if tmp:
        typ = '[EXACT TABLE]' if is_table else '[EXACT COLUMN]'
        tmp.insert(0, typ)
        for idx in range(s_idx, e_idx):
            tmp_list[idx] += tmp
    return tmp


def schema_linking(schema, question_toks, english_RelatedTo, english_IsA):
    # To-Do: Some words get worse after lemmatization. Need to fix it
    # Lemmatize Question
    lemmatized_question = [wordnet_lemmatizer.lemmatize(word) for word in question_toks]

    q_len = len(question_toks)
    assert q_len == len(lemmatized_question)

    # Lemmatize Tables (1.to singular, 2.to present tense)
    lemmatized_tables_1 = []
    lemmatized_tables_2 = []
    for table in schema.get_table_names():
        tmp1 = [wordnet_lemmatizer.lemmatize(x.lower()) for x in table.split(' ')]
        tmp2 = [lemma(x.lower()) if lemma(x.lower()) else x.lower() for x in table.split(' ')]
        lemmatized_tables_1 += [tmp1]
        lemmatized_tables_2 += [tmp2]

    # Table Keys
    table_keys = [table for table in schema.get_table_names()]

    # Lemmatize Columns (1.to singular 2.to present tense)
    lemmatized_columns_1 = []
    lemmatized_columns_2 = []
    for column in schema.get_col_names():
        tmp1 = [wordnet_lemmatizer.lemmatize(x.lower()) for x in column.split(' ')]
        tmp2 = [lemma(x.lower()) if lemma(x.lower()) else x.lower() for x in column.split(' ')]
        lemmatized_columns_1 += [tmp1]
        lemmatized_columns_2 += [tmp2]

    # Column Keys
    column_keys = \
        ['.'.join([schema.get_table_name(schema.get_parent_table_id(id)), column]) for id, column in schema.get_col_items()]

    # Match with schema
    idx = 0
    schema_annotation = [[] for _ in range(q_len)]
    while idx < q_len:
        # Exact-Match with col
        r_idx = exact_match(idx, lemmatized_question, lemmatized_columns_1, column_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue
        r_idx = exact_match(idx, lemmatized_question, lemmatized_columns_2, column_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue

        # Exact-Match with table
        r_idx = exact_match(idx, lemmatized_question, lemmatized_tables_1, table_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue
        r_idx = exact_match(idx, lemmatized_question, lemmatized_tables_2, table_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue

        # Partial-Match with col
        r_idx = partial_match(idx, lemmatized_question, lemmatized_columns_1, column_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue
        r_idx = partial_match(idx, lemmatized_question, lemmatized_columns_2, column_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue

        # Partial-Match with table
        r_idx = partial_match(idx, lemmatized_question, lemmatized_tables_1, table_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue
        r_idx = partial_match(idx, lemmatized_question, lemmatized_tables_2, table_keys, schema_annotation)
        if r_idx:
            idx = r_idx
            continue

        # Match with knowledge Graph
        r_idx = get_value_idx(idx, question_toks)
        if r_idx:
            s_idx = idx
            idx = r_idx
            # Search with Is-a relation
            result = search_knowledge_graph(question_toks[idx:r_idx], english_IsA)
            if result:
                if match(result, lemmatized_columns_1, s_idx, r_idx, schema_annotation, False):
                    continue
                if match(result, lemmatized_columns_2, s_idx, r_idx, schema_annotation, False):
                    continue
                if match(result, lemmatized_tables_1, s_idx, r_idx, schema_annotation, True):
                    continue
                if match(result, lemmatized_tables_2, s_idx, r_idx, schema_annotation, True):
                    continue
            # Search with Related-to relation
            result = search_knowledge_graph(question_toks[idx:r_idx], english_RelatedTo)
            if result:
                if match(result, lemmatized_columns_1, s_idx, r_idx, schema_annotation, False):
                    continue
                if match(result, lemmatized_columns_2, s_idx, r_idx, schema_annotation, False):
                    continue
                if match(result, lemmatized_tables_1, s_idx, r_idx, schema_annotation, True):
                    continue
                if match(result, lemmatized_tables_2, s_idx, r_idx, schema_annotation, True):
                    continue
            continue
        idx += 1

    # Convert Schema_annotation to None
    schema_annotation = [anno if anno else '[NONE]' for anno in schema_annotation]

    return schema_annotation



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

    with open(args.table_path) as f:
        schema_tables = json.load(f)
    schema_dict = dict()
    for one_schema in schema_tables:
        schema_dict[one_schema["db_id"]] = one_schema
        schema_dict[one_schema["db_id"]]["only_cnames"] = [c_name.lower() for tid, c_name in one_schema["column_names_original"]]
        schema = RelationalSchema()
        schema.import_from_spider(one_schema)
        schema_dict[one_schema["db_id"]]["schema_object"] = schema
    # copy of the origin question_toks
    for d in datas:
        if 'origin_question_toks' not in d:
            d['origin_question_toks'] = d['question_toks']

    for entry in datas:
        db_id = entry['db_id']
        if db_id not in db_values:
            schema_json = schema_dict[db_id]
            primary_foreigns = set()
            for f, p in schema_json["foreign_keys"]:
                primary_foreigns.add(f)
                primary_foreigns.add(p)

            conn = sqlite3.connect("../data/wikitablequestions/database/{}/{}.sqlite".format(db_id, db_id))
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
                    col_idx = schema_json["only_cnames"].index(col)
                    if col_idx in primary_foreigns and schema_json["column_types"][col_idx] == "number":
                        continue
                    cursor.execute("SELECT \"{}\" FROM \"{}\"".format(col, table))
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
        origin_question_toks = [x.lower() for x in re.findall(r"[^,.():;\"`?! ]+|[,.():;\"?!]", entry['question'].replace("'", " ' "))]
        question_toks = [wordnet_lemmatizer.lemmatize(x) for x in origin_question_toks]

        entry['question_toks'] = origin_question_toks

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
        linking_result = []
        nltk_result = nltk.pos_tag(question_toks)
        while idx < num_toks:

            # fully header
            end_idx, header, header_indices = fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                linking_result.append(("qc_exact", header_indices))
                idx = end_idx
                continue

            # check for table
            end_idx, tname, header_indices = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["table"])
                linking_result.append(("qt_exact", header_indices))
                idx = end_idx
                continue

            # check for column
            end_idx, header, header_indices = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                linking_result.append(("qc_exact", header_indices))
                idx = end_idx
                continue

            # check for partial column
            end_idx, tname, headers, header_indices = partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["col"] + headers)
                linking_result.append(("qc_partial", header_indices))
                idx = end_idx
                continue
            # check for aggregation
            end_idx, agg, _ = group_header(question_toks, idx, num_toks, AGG)
            if agg:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["agg"])
                linking_result.append(("none",))
                idx = end_idx
                continue

            if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MORE'])
                linking_result.append(("none",))
                idx += 1
                continue

            if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MOST'])
                linking_result.append(("none",))
                idx += 1
                continue

            # string match for Time Format
            if num2year(question_toks[idx]):
                # question_toks[idx] = 'year'
                end_idx, header, header_indices = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    linking_result.append(("qc_exact", header_indices))

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
                    linking_result.append(("none",))
                    type_concol.append([pro_result])
                    pro_result = "NONE"

                idx = end_idx
                continue

            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx]]
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
                    linking_result.append(("none",))

                idx = end_idx
                continue

            end_idx, values, cols, col_indices = group_db(origin_question_toks, idx, num_toks, db_values[db_id])
            if end_idx == idx + 1 and (nltk_result[idx][1] == 'VBZ'
                                       or nltk_result[idx][1] == 'IN'
                                       or nltk_result[idx][1] == 'CC'
                                       or nltk_result[idx][1] == 'DT'
                                       or origin_question_toks[idx] == "'"
                                       or (nltk_result[idx][1] == 'VBP' and origin_question_toks[idx] == 'are')
                                       or (nltk_result[idx][1] == 'VBP' and origin_question_toks[idx] == 'do')
                                       or (nltk_result[idx][1] == 'VBP' and origin_question_toks[idx] == 'doe')
                                       or (nltk_result[idx][1] == 'VBP' and origin_question_toks[idx] == 'does')):
                tok_concol.append([origin_question_toks[idx]])
                type_concol.append(['NONE'])
                linking_result.append(("none",))
                idx += 1
                continue
            if values:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["db"] + cols)
                linking_result.append(("qc_db", col_indices))

                idx = end_idx
                continue

            result = group_digital(question_toks, idx)
            if result is True:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["value"])
                linking_result.append(("none",))
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            tok_concol.append([origin_question_toks[idx]])
            linking_result.append(("none",))
            type_concol.append(['NONE'])
            idx += 1
            continue

        assert len(tok_concol) == len(linking_result)

        # make relation
        relation_matrix = [[0] * (1 + len(tok_concol) + len(entry['col_set']) + len(entry["table_names"])) for _ in range((1 + len(tok_concol) + len(entry['col_set']) + len(entry["table_names"])))]
        relation_matrix[0][0] = RELATION_TYPE['cls_cls']
        for relation_ed in range(1, 1 + len(linking_result)):
            relation_matrix[0][relation_ed] = RELATION_TYPE['cls_q']
        for relation_ed in range(1 + len(linking_result), 1 + len(linking_result) + len(entry['col_set'])):
            relation_matrix[0][relation_ed] = RELATION_TYPE['cls_c']
        for relation_ed in range(1 + len(linking_result) + len(entry['col_set']), 1 + len(linking_result) + len(entry['col_set']) + len(entry['table_names'])):
            relation_matrix[0][relation_ed] = RELATION_TYPE['cls_t']
        for relation_st in range(len(linking_result)):
            st_idx = 1 + relation_st
            relation_matrix[st_idx][0] = RELATION_TYPE['q_cls']
            for relation_ed in range(relation_st, len(linking_result)):
                ed_idx = 1 + relation_ed
                if relation_st - relation_ed <= -2:
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qq_-2']
                    relation_matrix[ed_idx][st_idx] = RELATION_TYPE['qq_2']
                elif relation_st - relation_ed == -1:
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qq_-1']
                    relation_matrix[ed_idx][st_idx] = RELATION_TYPE['qq_1']
                elif relation_st - relation_ed == 0:
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qq_0']
            for relation_ed in range(len(entry['col_set'])):
                ed_idx = 1 + len(linking_result) + relation_ed
                relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qc_no']
                relation_matrix[ed_idx][st_idx] = RELATION_TYPE['cq_no']
            for relation_ed in range(len(entry['table_names'])):
                ed_idx = 1 + len(linking_result) + len(entry['col_set']) + relation_ed
                relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qt_no']
                relation_matrix[ed_idx][st_idx] = RELATION_TYPE['tq_no']
            if linking_result[relation_st][0] == 'qc_exact':
                for col_idx in linking_result[relation_st][1]:
                    ed_idx = 1 + len(linking_result) + col_idx
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qc_exact']
                    relation_matrix[ed_idx][st_idx] = RELATION_TYPE['cq_exact']
            if linking_result[relation_st][0] == 'qc_partial':
                for col_idx in linking_result[relation_st][1]:
                    ed_idx = 1 + len(linking_result) + col_idx
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qc_partial']
                    relation_matrix[ed_idx][st_idx] = RELATION_TYPE['cq_partial']
            if linking_result[relation_st][0] == 'qc_db':
                for col_idx in linking_result[relation_st][1]:
                    ed_idx = 1 + len(linking_result) + col_idx
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qc_db']
                    relation_matrix[ed_idx][st_idx] = RELATION_TYPE['cq_db']
            if linking_result[relation_st][0] == 'qt_exact':
                for tab_idx in linking_result[relation_st][1]:
                    ed_idx = 1 + len(linking_result) + len(entry['col_set']) + tab_idx
                    relation_matrix[st_idx][ed_idx] = RELATION_TYPE['qt_exact']
                    relation_matrix[ed_idx][st_idx] = RELATION_TYPE['tq_exact']
        schema = schema_dict[db_id]
        for col_idx_1 in range(len(schema["column_names"])):
            par_id_1, col_name_1 = schema["column_names"][col_idx_1]
            col_set_idx_1 = entry['col_set'].index(col_name_1)
            relation_matrix_idx_1 = 1 + len(linking_result) + col_set_idx_1
            relation_matrix[relation_matrix_idx_1][0] = RELATION_TYPE['c_cls']
            for col_idx_2 in range(len(schema["column_names"])):
                par_id_2, col_name_2 = schema["column_names"][col_idx_2]
                col_set_idx_2 = entry['col_set'].index(col_name_2)
                relation_matrix_idx_2 = 1 + len(linking_result) + col_set_idx_2
                if col_set_idx_1 == col_set_idx_2:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['cc_identical']
                elif par_id_1 == par_id_2:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['cc_sibling']
                elif (col_idx_1, col_idx_2) in schema['foreign_keys']:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['cc_foreign_primary']
                elif (col_idx_2, col_idx_1) in schema['foreign_keys']:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['cc_primary_foreign']
                else:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['cc_etc']
            for tab_idx in range(len(entry['table_names'])):
                relation_matrix_idx_2 = 1 + len(linking_result) + len(entry['col_set']) + tab_idx
                if par_id_1 == tab_idx and col_idx_1 in schema['primary_keys']:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['ct_primary_child']
                elif par_id_1 == tab_idx:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['ct_child']
                else:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['ct_etc']

        for tab_idx in range(len(entry['table_names'])):
            relation_matrix_idx_1 = 1 + len(linking_result) + len(entry['col_set']) + tab_idx
            relation_matrix[relation_matrix_idx_1][0] = RELATION_TYPE['t_cls']
            for col_idx_2 in range(len(schema["column_names"])):
                par_id_2, col_name_2 = schema["column_names"][col_idx_2]
                col_set_idx_2 = entry['col_set'].index(col_name_2)
                relation_matrix_idx_2 = 1 + len(linking_result) + col_set_idx_2
                if par_id_2 == tab_idx and col_idx_2 in schema['primary_keys']:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tc_primary_child']
                elif par_id_2 == tab_idx:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tc_child']
                else:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tc_etc']
            for tab_idx_2 in range(len(entry['table_names'])):
                relation_matrix_idx_2 = 1 + len(linking_result) + len(entry['col_set']) + tab_idx_2
                if tab_idx == tab_idx_2:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tt_identical']
                    continue
                for f, p in schema['foreign_keys']:
                    f_par, _ = schema['column_names'][f]
                    p_par, _ = schema['column_names'][p]
                    if f_par == tab_idx and p_par == tab_idx_2:
                        if relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] == RELATION_TYPE['tt_reversed']:
                            relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tt_both']
                        else:
                            relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tt_foreign']
                    if p_par == tab_idx and f_par == tab_idx_2:
                        if relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] == RELATION_TYPE['tt_foreign']:
                            relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tt_both']
                        else:
                            relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tt_reversed']

                if relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] == 0:
                    relation_matrix[relation_matrix_idx_1][relation_matrix_idx_2] = RELATION_TYPE['tt_etc']


        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol
        entry['nltk_pos'] = nltk_result
        schema_annotation = schema_linking(schema_dict[db_id]["schema_object"], entry['question_toks'], english_RelatedTo, english_IsA)

        entry['schema_linking'] = schema_annotation
        entry['relation_matrix'] = relation_matrix
        assert len(entry['relation_matrix']) == 1 + len(tok_concol) + len(entry['col_set']) + len(entry["table_names"])

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
        json.dump(datas, f, indent=4)


