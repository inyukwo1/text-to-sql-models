# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import json
import time
import pickle

import copy
import numpy as np
import os
import torch
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from src.dataset import Example
from src.rule import lf
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

wordnet_lemmatizer = WordNetLemmatizer()


def idx2seq(seq, indices, cur_idx):
    if indices[cur_idx]:
        cur = indices[cur_idx]
        new_seq = []
        for item in cur:
            new_seq += idx2seq(seq, indices, item)
        return [seq[cur_idx]] + new_seq
    else:
        return [seq[cur_idx]]


def get_terminal_idx(seq, parent, begin_idx):
    '''
    :param seq: List (list of SemQL in left-to-right order)
    :param parent: Action (parent action of current action)
    :param begin_idx: Int (idx of current action in seq)
    :return: Int (terminal idx of the current action with respect to seq)
    '''
    cur = seq[begin_idx]
    assert str(parent) == str(cur.parent)

    if cur.children:
        children_idx = []
        for child in cur.children:
            target_idx = children_idx[-1] + 1 if children_idx else begin_idx + 1
            children_idx += [get_terminal_idx(seq, cur, target_idx)]
        return children_idx[-1]
    else:
        return begin_idx


def seq2idx(seq):
    '''
    :param seq: List (list of SemQL in left-to-right order)
    :return: List (list of list containing indices as a tree)
    '''
    indices = []
    for idx, cur in enumerate(seq):
        children_idx = [idx+1] if cur.children else []
        children_idx += [get_terminal_idx(seq, cur, children_idx[-1]) + 1 for children_num in range(1, len(cur.children))]
        indices += [children_idx]

    return indices

def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}

    cache_name = file_name.replace('txt', 'pkl')
    if os.path.isfile(cache_name) and not use_small:
        with open(cache_name, 'rb') as cache_file:
            ret = pickle.load(cache_file)
    else:
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
        with open(cache_name, 'wb') as cache_file:
            pickle.dump(ret, cache_file)
    return ret

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

def get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, tab_set_type, tab_set_iter, sql):

    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            try:
                tab_set_type[tab_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['[table]'] + question_arg[count_q]
            except:
                print(tab_set_iter, question_arg[count_q])
                raise RuntimeError("not in tab set")
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                for idx in [idx for idx, value in enumerate(col_set_iter) if value == question_arg[count_q]]:
                    col_set_type[idx][1] = 5
                question_arg[count_q] = ['[column]'] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['[value]'] + question_arg[count_q]
        elif t == 'db':
            one_hot_type[count_q][6] = 1
            c_cand = [wordnet_lemmatizer.lemmatize(v).lower() for v in t_q[1].split(" ")]
            question_arg[count_q] = ['[db]'] + question_arg[count_q]
            for idx in [idx for idx, value in enumerate(col_set_iter) if value == c_cand]:
                col_set_type[idx][4] = 5
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:

                        for idx in [idx for idx, value in enumerate(col_set_iter) if value == col_probase]:
                            col_set_type[idx][2] = 5
                        question_arg[count_q] = ['[value]'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    for idx in [idx for idx, value in enumerate(col_set_iter) if value == col_probase]:
                        col_set_type[idx][3] += 1

def process(sql, table):

    process_dict = {}

    origin_sql = sql['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]

    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    tab_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['table_names']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 7))

    col_set_type = np.zeros((len(col_set_iter), 5))
    tab_set_type = np.zeros((len(table_names), 5))

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['tab_set_type'] = tab_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['tab_set_iter'] = tab_set_iter
    process_dict['relation_matrix'] = sql['relation_matrix']

    return process_dict

def is_valid(rule_label, col_table_dict, sql):
    try:
        lf.build_tree(copy.copy(rule_label))
    except:
        print(rule_label)

    flag = False
    for r_id, rule in enumerate(rule_label):
        if type(rule) == C:
            try:
                assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
            except:
                flag = True
                print(sql['question'])
    return flag is False


def to_batch_seq(sql_data, table_data, idxes, st, ed,
                 is_train=True):
    """

    :return:
    """
    examples = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        table = table_data[sql['db_id']]

        process_dict = process(sql, table)

        for c_id, col_ in enumerate(process_dict['col_set_iter']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in col_:
                    process_dict['col_set_type'][c_id][0] += 1

        for t_id, tab_ in enumerate(process_dict['table_names']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in tab_:
                    process_dict['tab_set_type'][t_id][0] += 1

        schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                       process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'],
                       process_dict['tab_set_type'], process_dict['table_names'],
                       sql)

        col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)
        table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

        process_dict['col_set_iter'][0] = ['count', 'number', 'many']

        rule_label = None
        if 'rule_label' in sql and is_train:
            # handle the subquery on From cause
            if 'from' in sql['rule_label']:
                continue
            rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]
            if is_valid(rule_label, col_table_dict=col_table_dict, sql=sql) is False:
                continue

        example = Example(
            src_sent=process_dict['question_arg'],
            col_num=len(process_dict['col_set_iter']),
            vis_seq=(sql['question'], process_dict['col_set_iter'], sql['query']),
            tab_cols=process_dict['col_set_iter'],
            tab_iter=process_dict['tab_set_iter'],
            sql=sql['query'],
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_set_type'],
            tab_hot_type=process_dict['tab_set_type'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            col_table_dict=col_table_dict,
            cols=process_dict['tab_cols'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict['col_set_type'],
            tgt_actions=rule_label,
            relation=process_dict['relation_matrix']
        )
        example.sql_json = copy.deepcopy(sql)
        example.db_id = sql['db_id']
        examples.append(example)

    if is_train:
        examples.sort(key=lambda e: -len(e.src_sent))
        return examples
    else:
        return examples

def epoch_train(model, optimizer, bert_optimizer, batch_size, sql_data, table_data,
                args, epoch=0, loss_epoch_threshold=20, sketch_loss_coefficient=0.2):
    model.train()
    # shuffle
    new_sql_data = []
    if args.bert != -1:
        for sql in sql_data:
            if sql["db_id"] != "baseball_1":
                new_sql_data.append(sql)

    sql_data = new_sql_data
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    for st in tqdm(range(0, len(sql_data), batch_size)):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed)
        optimizer.zero_grad()
        if bert_optimizer:
            bert_optimizer.zero_grad()

        score = model.forward(examples)
        if score[0] is None:
            continue
        loss_sketch = -score[0]
        loss_lf = -score[1]

        loss_sketch = torch.mean(loss_sketch)
        loss_lf = torch.mean(loss_lf)

        if epoch > loss_epoch_threshold:
            loss = loss_lf + sketch_loss_coefficient * loss_sketch
        else:
            loss = loss_lf + loss_sketch

        loss.backward()
        if args.clip_grad > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        if bert_optimizer:
            bert_optimizer.step()
        cum_loss += loss.data.cpu().numpy()*(ed - st)
    return cum_loss / len(sql_data)

def epoch_acc(model, batch_size, sql_data, table_data, beam_size=3):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0

    json_datas = []
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        examples = to_batch_seq(sql_data, table_data, perm, st, ed,
                                                        is_train=False)
        for example in examples:
            try:
                results_all = model.parse(example, beam_size=beam_size)
                results = results_all[0]
                list_preds = []

                pred = " ".join([str(x) for x in results[0].actions])
                for x in results:
                    list_preds.append(" ".join(str(x.actions)))
            except Exception as e:
                print('Error: {}'.format(e))
                results_all = [[], []]
                pred = ""

            simple_json = example.sql_json['pre_sql']

            simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
            simple_json['model_result'] = pred

            simple_json['col_set_type'] = example.col_hot_type
            simple_json['tab_set_type'] = example.tab_hot_type
            json_datas.append(simple_json)
        st = ed
    return json_datas

import random
def eval_acc(preds, sqls, log=False):
    sketch_correct, best_correct = 0, 0
    perm = list(range(len(preds)))
    random.shuffle(perm)
    for q_idx in range(len(preds)):
        perm_idx = perm[q_idx]
        pred = preds[perm_idx]
        sql = sqls[perm_idx]
        if pred['model_result'] == sql['rule_label']:
            best_correct += 1

        elif log:
            print("IDX: {}, origin IDX: {}".format(q_idx, perm_idx))
            print("DB ID: {}".format(sql['db_id']))
            print("QUESTION: {}".format(sql['question_arg']))
            print("QUESTION: {}".format(sql['question_arg_type']))
            print("QUERY: {}".format(sql["query"]))
            print("tables: {}".format(list(zip(range(len(sql["table_names"])), sql["table_names"]))))
            print("{}".format(pred["tab_set_type"].transpose((1, 0))))
            print("col set: {}".format(list(zip(range(len(sql["col_set"])), sql["col_set"]))))
            print("{}".format(pred["col_set_type"].transpose((1, 0))))
            print("GOLD: {}".format(sql['rule_label']))
            print("PRED: {}".format(pred['model_result']))
            print("\n\n")

        tmp = ' '.join([t for t in pred['rule_label'].split(' ') if t.split('(')[0] not in ['A', 'C', 'T']])
        if pred['sketch_result'] == tmp:
            sketch_correct += 1

    return best_correct / len(preds), sketch_correct / len(preds)


def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    table_data_new = {table['db_id']: table for table in table_data}

    if use_small:
        return sql_data[:80], table_data_new
    else:
        return sql_data, table_data_new


def load_dataset(dataset_dir, use_small=False):
    print("Loading from datasets...")

    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train_rat.json")
    DEV_PATH = os.path.join(dataset_dir, "dev_rat.json")
    with open(TABLE_PATH) as inf:
        print("Loading data from %s"%TABLE_PATH)
        table_data = json.load(inf)

    train_sql_data, train_table_data = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data = load_data_new(DEV_PATH, table_data, use_small=use_small)

    return train_sql_data, train_table_data, val_sql_data, val_table_data


def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

def init_log_checkpoint_path(args):
    save_path = args.save
    dir_name = save_path + str(int(time.time()))
    save_path = os.path.join(os.path.curdir, 'saved_model', dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path
