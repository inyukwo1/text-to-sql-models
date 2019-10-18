import torch
from src import args as arg
from src import utils
from src.models.model import IRNet
from src.rule import semQL
from preprocess.utils import symbol_filter, re_lemma, fully_part_header, group_header, partial_header, num2year, group_symbol, group_values, group_digital
from preprocess.utils import AGG, wordnet_lemmatizer
from src.utils import process, schema_linking, get_col_table_dict, get_table_colNames
from src.dataset import Example
from src.rule.sem_utils import *
from sem2SQL import transform
import nltk
import os
import pickle
import re
from flask_restful import Resource, reqparse, Api
from flask import Flask
from flask_cors import CORS

with open('preprocess/conceptNet/english_RelatedTo.pkl', 'rb') as f:
    english_RelatedTo = pickle.load(f)

with open('preprocess/conceptNet/english_IsA.pkl', 'rb') as f:
    english_IsA = pickle.load(f)

def preprocess_data(entry):
    if 'origin_question_toks' not in entry:
        entry['origin_question_toks'] = entry['question_toks']
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


def get_runner(model_path):

    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)

    grammar = semQL.Grammar()
    sql_data, table_data, val_sql_data,\
    val_table_data= utils.load_dataset(args.dataset, use_small=args.toy)

    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    print('load pretrained model from %s'% (model_path))
    pretrained_model = torch.load(model_path,
                                     map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    model.word_emb = utils.load_word_emb(args.glove_embed_path)
    model.eval()

    def runner(db_id, nl_string):
        table = val_table_data[db_id]
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]

        entry = {}
        entry["question"] = nl_string
        entry["question_toks"] = re.findall(r"[^,.:;\"`?! ]+|[,.:;\"?!]", nl_string.replace("'", " '")) # TODO split by special separator
        entry['names'] = table['schema_content']
        entry['table_names'] = table['table_names']
        entry['col_set'] = table['col_set']
        entry['col_table'] = table['col_table']
        keys = {}
        for kv in table['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in table['primary_keys']:
            keys[id_k] = id_k
        entry['keys'] = keys
        preprocess_data(entry)

        process_dict = process(entry, table)

        for c_id, col_ in enumerate(process_dict['col_set_iter']):
            for q_id, ori in enumerate(process_dict['q_iter_small']):
                if ori in col_:
                    process_dict['col_set_type'][c_id][0] += 1

        schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                       process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'], entry)

        col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], entry)
        table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

        process_dict['col_set_iter'][0] = ['count', 'number', 'many']

        rule_label = None

        example = Example(
            src_sent=process_dict['question_arg'],
            col_num=len(process_dict['col_set_iter']),
            vis_seq=(entry['question'], process_dict['col_set_iter'], None),
            tab_cols=process_dict['col_set_iter'],
            sql=None,
            one_hot_type=process_dict['one_hot_type'],
            col_hot_type=process_dict['col_set_type'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            col_table_dict=col_table_dict,
            cols=process_dict['tab_cols'],
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            tokenized_src_sent=process_dict['col_set_type'],
            tgt_actions=rule_label
        )
        example.sql_json = copy.deepcopy(entry)

        results_all = model.parse(example, beam_size=5)
        results = results_all[0]
        list_preds = []
        try:
            pred = " ".join([str(x) for x in results[0].actions])
            for x in results:
                list_preds.append(" ".join(str(x.actions)))
        except Exception as e:
            # print('Epoch Acc: ', e)
            # print(results)
            # print(results_all)
            pred = ""

        simple_json = example.sql_json['pre_sql']
        simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
        simple_json['model_result'] = pred
        simple_json["query"] = None

        print(simple_json)

        alter_not_in_one_entry(simple_json, table)
        alter_inter_one_entry(simple_json)
        alter_column0_one_entry(simple_json)

        try:
            result = transform(simple_json, table)
        except Exception as e:
            result = transform(simple_json, table,
                               origin='Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)')
        return result[0]

    return runner


runner = get_runner("saved_model/best_model.model")


class Service(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('db_id', required=True, type=str)
            parser.add_argument('question', required=True, type=str)
            args = parser.parse_args()
            print("done well")
            return {'result': runner(args["db_id"], args["question"])}
        except Exception as e:
            print("done not well")
            return {'result': str(e)}


app = Flask('irnet service')
CORS(app)
api = Api(app)
api.add_resource(Service, '/service')

if __name__ == "__main__":
    # runner("dog_kennels", "Which professionals have done at least two treatments? List the professional's id, role, and first name.")
    app.run(host='141.223.199.148', port=5000, debug=False)