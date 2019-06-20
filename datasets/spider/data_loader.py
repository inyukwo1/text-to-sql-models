import os
import json
import tqdm
import random
from datasets.schema import Schema
from models.frompredictor.ontology import Ontology

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


class DataLoader():
    def __init__(self, batch_size):
        self.schemas = {}
        self.dbs = {}
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.eval_len = None
        self.train_len = None

        self.batch_size = batch_size

        self._data_path = './datasets/spider/data/'
        self._table_path = os.path.join(self._data_path, 'tables.json')
        self._train_path = os.path.join(self._data_path, 'train_type.json')
        self._dev_path = os.path.join(self._data_path, 'dev_type.json')
        self._test_path = os.path.join(self._data_path, 'dev_type.json')
        self._init()

    def _init(self):
        # Load Table Data
        with open(self._table_path, encoding='utf-8') as f:
            table_data = json.load(f)

        # Load Schema
        for table in table_data:
            schema = Schema()
            schema.import_from_spider(table)
            self.schemas[schema.db_id] = schema
            self.dbs[schema.db_id] = table

    def _get(self, data):
        data_len = len(data)
        for i in tqdm.tqdm(range(0, data_len, self.batch_size)):
            yield(data[i:min(i+self.batch_size, data_len)])

    def _load(self, path, load_option=None):
        if load_option:
            root, history, train_dev, component = load_option
            return json.load(open(os.path.join(self._data_path, "{}/{}_{}_{}_dataset.json".format(root, history, train_dev, component))))

        sql_list = []
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            sql_tmp = {}
            schema = self.schemas[item['db_id']]
            db = self.dbs[item['db_id']]

            # Info
            sql_tmp['question'] = item['question']
            sql_tmp['question_toks'] = item['question_toks']
            sql_tmp['question_tok_concol'] = item['question_tok_concol']
            sql_tmp['question_type_concol_list'] = item['question_type_concol_list']
            sql_tmp['query'] = item['query']
            sql_tmp['query_toks'] = item['query_toks']
            sql_tmp['sql'] = item['sql']
            sql_tmp['from'] = item['sql']['from']
            sql_tmp['column'] = db['column_names']

            sql_tmp['schema'] = schema
            # DB info
            sql_tmp['db_id'] = item['db_id']
            sql_tmp['db'] = db
            sql_tmp['tbl'] = db['table_names']
            sql_tmp['foreign_keys'] = db['foreign_keys']
            sql_tmp['primary_keys'] = db['primary_keys']

            # GOLD Values
            sql_tmp['agg'] = []
            sql_tmp['sel'] = []
            sql_tmp['cond'] = []
            sql_tmp['conj'] = []
            sql_tmp['group'] = []
            sql_tmp['order'] = []

            # For From predictor
            sql_tmp['join_table_dict'] = []
            sql_tmp['history'] = ['none']

            # Parse AGG and SEL
            gt_sel = item['sql']['select'][1][:3]

            for tup in gt_sel:
                sql_tmp['agg'].append(tup[0])
                sql_tmp['sel'].append(tup[1][1][1])

            # Parse Conditions [[col, op, value], [col, op, value], ...]
            gt_cond = item['sql']['where']

            if gt_cond:
                conds = gt_cond[0::2] # Get conditions (without conjunction operator)
                for cond in conds:
                    curr_cond = [cond[2][1][1]] # column
                    curr_cond += [cond[1]] # Operator
                    curr_cond += [[cond[3], cond[4]]] if cond[4] else [cond[3]] # Values (two values if two)
                    sql_tmp['cond'].append(curr_cond)

            # Parse Conjunctions
            sql_tmp['conj'] = gt_cond[1::2]

            # Parse GROUP BY
            sql_tmp['group'] = [x[1] for x in item['sql']['groupBy']] # Assume only one groupby

            having_cond = []
            if item['sql']['having']:
                gt_having = item['sql']['having'][0]  # currently only do first having condition
                having_cond.append([gt_having[2][1][0]])  # aggregator
                having_cond.append([gt_having[2][1][1]])  # column
                having_cond.append([gt_having[1]])  # operator
                having_cond += [[gt_having[3], gt_having[4]]] if gt_having[4] else [gt_having[3]] # Values (two if two)
            else:
                having_cond = [[], [], []]

            sql_tmp['group'].append(having_cond)  # GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

            # Parse order by / limit
            order_aggs = []
            order_cols = []
            order_par = 4
            gt_order = item['sql']['orderBy']
            limit = item['sql']['limit']

            if gt_order:
                order_aggs = [x[1][0] for x in gt_order[1][:1]]  # limit to 1 order by
                order_cols = [x[1][1] for x in gt_order[1][:1]]
                if limit != None:
                    order_par = 0 if gt_order[0] == 'asc' else 1
                else:
                    order_par = 2 if gt_order[0] else 3

            sql_tmp['order'] = [order_aggs, order_cols, order_par]  # GOLD for ORDER [[[agg], [col], [dat]], []]

            # Parse intersect/except/union
            sql_tmp['special'] = 0
            if item['sql']['intersect'] is not None:
                sql_tmp['special'] = 1
            elif item['sql']['except'] is not None:
                sql_tmp['special'] = 2
            elif item['sql']['union'] is not None:
                sql_tmp['special'] = 3

            # Parse join_table_dict
            join_table_dict = dict()
            table_list = [table[1] for table in item["sql"]["from"]["table_units"]]
            for i in range(1):
                for table_num in table_list:
                    if type(table_num) is dict:
                        #print("WRONG2")  # nested query is in from clause # TODO handle needed
                        break
                    join_table_dict[table_num] = set()

                join_conds = item["sql"]["from"]["conds"]
                join_cols_list = []
                for cond in join_conds:
                    if cond != 'and':
                        join_cols_list.append(cond[2][1][1])
                        join_cols_list.append(cond[3][1])

                for col in join_cols_list:
                    parent_table = schema.get_parent_table_id(col)
                    if parent_table not in join_table_dict:
                        #print("WRONG111111")  # syntaxsqlnet bug - parsing bug # TODO handle needed
                        break
                    else:
                        join_table_dict[parent_table].add(col)
                for table_unit in join_table_dict:
                    join_table_dict[table_unit] = list(join_table_dict[table_unit])

            sql_tmp['join_table_dict'] = join_table_dict
            ontology = Ontology()
            ontology.import_from_sql(item['sql'])
            sql_tmp['ontology'] = ontology

            sql_list.append(sql_tmp)
        return sql_list


    def shuffle(self):
        if self.train_data:
            random.shuffle(self.train_data)
        if self.dev_data:
            random.shuffle(self.dev_data)
        if self.test_data:
            random.shuffle(self.test_data)

    def get_train(self):
        return self._get(self.train_data)

    def get_train_len(self):
        return len(self.train_data)

    def get_eval(self):
        return self._get(self.dev_data) if self.dev_data else self._get(self.test_data)

    def get_eval_len(self):
        return len(self.dev_data) if self.dev_data else len(self.test_data)

    def load_data(self, type, load_option=None):
        print('Loading {} data'.format(type))
        if type == 'test':
            self.test_data = self._load(self._test_path, load_option)
            self.eval_len = len(self.test_data)
        elif type == 'train':
            self.train_data = self._load(self._train_path, load_option)
            self.dev_data = self._load(self._dev_path, load_option)
            self.train_len = len(self.train_data)
            self.eval_len = len(self.dev_data)
        else:
            print('Wrong type')
            exit(-1)

