import os
import json

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')

class Dataloader():
    def __init__(self, data_path, use_small=False):
        self.schemas = {}
        self.use_small = use_small
        self.table_path = os.path.join(data_path, 'tables.json')
        self.train_path = os.path.join(data_path, 'train.json')
        self.dev_path = os.path.join(data_path, 'dev.json')
        self.test_path = os.path.join(data_path, 'dev.json')

    def get_data(self):
        print('Loading from datasets...')

        # Load Table Data
        with open(self.table_path, encoding='utf-8') as f:
            table_data = json.load(f)

        # Load Schema
        for table in table_data:
            self.schemas[table['db_id']] = table
            del self.schemas[table['db_id']]['db_id']

        # Load NLQ, SQL Data
        train_sql_data = self.process(self.train_path)
        dev_sql_data = self.process(self.dev_path)
        test_sql_data = self.process(self.train_path)

        return train_sql_data, dev_sql_data, test_sql_data, self.schemas

    def process(self, path):
        sql_list = []

        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            sql_tmp = {}
            db = self.schemas[item['db_id']]

            # Info
            sql_tmp['question'] = item['question']
            sql_tmp['question_tok'] = item['question_tok']
            sql_tmp['query'] = item['query']
            sql_tmp['query_tok'] = item['query_tok']
            sql_tmp['column'] = item['column_names_original']
            sql_tmp['from'] = item['sql']['from']

            # DB info
            sql_tmp['db_id'] = item['db_id']
            sql_tmp['tbl'] = db['table_names_original']
            sql_tmp['f_keys'] = db['foreign_keys']

            # GOLD Values
            sql_tmp['agg'] = []
            sql_tmp['sel'] = []
            sql_tmp['cond'] = []
            sql_tmp['conj'] = []
            sql_tmp['group'] = []
            sql_tmp['order'] = []

            # Parse for AGG and SEL
            gt_sel = item['sql']['select'][1][:3]

            for tup in gt_sel:
                sql_tmp['agg'].append(tup[0])
                sql_tmp['sel'].append(tup[1][1][1])

            # Parse Conditions [[col, op, value], [col, op, value], ...]
            gt_cond = item['sql']['where']

            if gt_cond:
                conds = gt_cond[0::2] # Get conditions (without conjunction operator)
                for cond in conds:
                    curr_cond = cond[2][1][1] # column
                    curr_cond += [cond[1]] # Operator
                    curr_cond += [[cond[3], cond[4]]] if cond[4] else [cond[3]] # Values (two values if two)
                    sql_tmp['cond'].append(curr_cond)

            # Parse Conjunctions
            sql_tmp['conj'] = gt_cond[1::2]

            # Parse GROUP BY
            sql_tmp['group'] = [x[1] for x in item['sql']['groupby']] # Assume only one groupby

            having_cond = []
            if item['sql']['having']:
                gt_having = item['sql']['having'][0]  # currently only do first having condition
                having_cond.append([gt_having[2][1][0]])  # aggregator
                having_cond.append([gt_having[2][1][1]])  # column
                having_cond.append([gt_having[1]])  # operator
                having_cond += [[gt_hav]]
                if gt_having[4] is not None:
                    having_cond.append([gt_having[3], gt_having[4]])
                else:
                    having_cond.append(gt_having[3])
            else:
                having_cond = [[], [], []]
            sql_tmp['group'].append(having_cond)  # GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

            # Parse order by / limit
            order_aggs = []
            order_cols = []
            sql_tmp['order'] = []
            order_par = 4
            gt_order = item['sql']['orderby']
            limit = item['sql']['limit']
            if len(gt_order) > 0:
                order_aggs = [x[1][0] for x in gt_order[1][:1]]  # limit to 1 order by
                order_cols = [x[1][1] for x in gt_order[1][:1]]
                if limit != None:
                    if gt_order[0] == 'asc':
                        order_par = 0
                    else:
                        order_par = 1
                else:
                    if gt_order[0] == 'asc':
                        order_par = 2
                    else:
                        order_par = 3

            sql_tmp['order'] = [order_aggs, order_cols, order_par]  # GOLD for ORDER [[[agg], [col], [dat]], []]

            # Parse intersect/except/union
            sql_tmp['special'] = 0
            if item['sql']['intersect'] is not None:
                sql_tmp['special'] = 1
            elif item['sql']['except'] is not None:
                sql_tmp['special'] = 2
            elif item['sql']['union'] is not None:
                sql_tmp['special'] = 3

            sql_list.append(sql_tmp)

        return sql_list

