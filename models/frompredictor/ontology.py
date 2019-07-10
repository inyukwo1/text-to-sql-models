from random import choice, randint, sample
from typing import List
from datasets.schema import Schema
from nltk import WordNetLemmatizer


class Ontology:
    def __init__(self):
        self.tables = set()
        self.cols = set()

    def __str__(self):
        return "ONTOLOGY:: [tables: {} cols: {}]".format(self.tables, self.cols)

    def import_from_sql(self, sql):
        def find_all_table_nums(sql):
            if isinstance(sql, list):
                for item in sql:
                    yield from find_all_table_nums(item)
            if isinstance(sql, dict):
                for keyword in sql:
                    if keyword == "table_units":
                        for _, table_num in sql["table_units"]:
                            if not isinstance(table_num, int):
                                yield from find_all_table_nums(table_num)
                            else:
                                yield table_num
                    else:
                        yield from find_all_table_nums(sql[keyword])

        def find_all_col_nums(sql):
            def yield_for_col_unit(col_unit):
                if not col_unit:
                    return False
                agg_id, col_id, isDistinct = col_unit
                yield col_id

            def yield_for_val_unit(val_unit):
                unit_op, col_unit1, col_unit2 = val_unit
                yield from yield_for_col_unit(col_unit1)
                yield from yield_for_col_unit(col_unit2)

            if not sql:
                return
            for cond in sql["from"]["conds"]:
                if isinstance(cond, str):
                    continue
                not_op, op_id, val_unit, val1, val2 = cond
                yield from yield_for_val_unit(val_unit)
                agg_id, col_id, isDistinct = val1
                yield col_id

            for _, table_num in sql["from"]["table_units"]:
                if isinstance(table_num, dict):
                    yield from find_all_col_nums(table_num)
                    continue
            for val_unit in sql["select"][1]:
                agg_id, val_unit = val_unit
                yield from yield_for_val_unit(val_unit)
            for cond in sql["where"]:
                if isinstance(cond, str):
                    continue
                not_op, op_id, val_unit, val1, val2 = cond
                yield from yield_for_val_unit(val_unit)
                if isinstance(val1, dict):
                    yield from find_all_col_nums(val1)
            for col_unit in sql["groupBy"]:
                yield from yield_for_col_unit(col_unit)
            for cond in sql["having"]:
                if isinstance(cond, str):
                    continue
                not_op, op_id, val_unit, val1, val2 = cond
                yield from yield_for_val_unit(val_unit)
                if isinstance(val1, dict):
                    yield from find_all_col_nums(val1)
                if isinstance(val1, dict):
                    yield from find_all_col_nums(val1)
            if sql["orderBy"]:
                for val_unit in sql["orderBy"][1]:
                    yield from yield_for_val_unit(val_unit)

            yield from find_all_col_nums(sql["except"])
            yield from find_all_col_nums(sql["intersect"])
            yield from find_all_col_nums(sql["union"])

        self.tables = set(find_all_table_nums(sql))
        self.cols = set(find_all_col_nums(sql)) - {0}

    def random_from_schema(self, schema):
        self.tables.add(schema.get_random_table_id())
        while randint(0, 100) < 50:
            pivot_table = choice(list(self.tables))
            neighbor_tables = schema.get_neighbor_table_ids(pivot_table)
            if neighbor_tables:
                selected_neighbor_table = choice(neighbor_tables)
                foreign_primary_keys = schema.get_foreign_primary_keys_with_two_tables(pivot_table,
                                                                                       selected_neighbor_table)
                selected_foreign, selected_primary = choice(foreign_primary_keys)
                self.tables.add(selected_neighbor_table)
                self.cols.add(selected_foreign)
                self.cols.add(selected_primary)

        max_col_num = min(len(self.tables) * 6, 10)
        col_num = choice(range(0, max_col_num))

        possible_cols = []
        for table_id in self.tables:
            possible_cols += schema.get_child_col_ids(table_id)
        col_num = min(col_num, len(possible_cols))
        self.cols |= set(sample(possible_cols, col_num))

    def is_same(self, ontology: 'Ontology'):
        if self.tables == ontology.tables and self.cols == ontology.cols:
            return True
        return False

    def make_hint_vector(self, schema: Schema, words: List[str]):
        hint = [0.] * len(words)
        words = [schema._lemmatizer.lemmatize(word) for word in words]
        for table_id in self.tables:
            table_name = schema.get_table_name(table_id)
            for n_gram in range(1, 4):
                for st in range(0, len(words) - n_gram):
                    if table_name == ' '.join(words[st:st + n_gram]):
                        hint[st:st + n_gram] = [1.] * n_gram

        for col_id in self.cols:
            col_name = schema.get_col_name(col_id)
            for n_gram in range(1, 4):
                for st in range(0, len(words) - n_gram):
                    if col_name == ' '.join(words[st:st + n_gram]):
                        hint[st:st + n_gram] = [1.] * n_gram
        return hint


