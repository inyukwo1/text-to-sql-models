import random


class Schema:
    def __init__(self):
        self.db_id = ""
        self._table_names = dict()
        self._table_names_original = dict()
        self._col_names = dict()
        self._col_names_original = dict()
        self._col_types = dict()
        self._col_parent = dict()
        self._foreign_primary_pairs  = []
        self._primary_keys = []

    def import_from_spider(self, spider_schema):
        self.db_id = spider_schema["db_id"]
        for tab_num, tab_name in enumerate(spider_schema["table_names"]):
            self._table_names[tab_num] = tab_name
        for tab_num, tab_name_original in enumerate(spider_schema["table_names_original"]):
            self._table_names_original[tab_num] = tab_name_original
        for col_num, (par_tab, col_name) in enumerate(spider_schema["column_names"]):
            # if par_tab != -1:
                self._col_names[col_num] = col_name
                self._col_parent[col_num] = par_tab

        for col_num, (par_tab, col_name_original) in enumerate(spider_schema["column_names_original"]):
            # if par_tab != -1:
                self._col_names_original[col_num] = col_name_original

        for idx in range(1, len(spider_schema['column_types'])):
            self._col_types[idx-1] = spider_schema['column_types'][idx]
        self._foreign_primary_pairs = spider_schema["foreign_keys"]
        self._primary_keys = spider_schema["primary_keys"]

    def get_parent_table_id(self, col_id):
        return self._col_parent[col_id]

    def get_random_table_id(self):
        table_id_list = list(self._table_names)
        return random.choice(table_id_list)

    def get_foreign_primary_pairs(self):
        return self._foreign_primary_pairs

    def is_primary_key(self, col_id):
        return col_id in self._primary_keys

    def get_all_table_ids(self):
        return list(self._table_names)

    def get_table_name(self, table_id):
        if table_id == -1:
            return '*'
        return self._table_names[table_id]

    def get_table_names(self):
        return self._table_names.values()

    def get_original_table_name(self, table_id):
        if table_id == -1:
            return '*'
        return self._table_names_original[table_id]

    def get_original_table_names(self):
        return self._table_names_original

    def get_col_name(self, col_id):
        return self._col_names[col_id]

    def get_col_names(self):
        return self._col_names.values()

    def get_child_col_ids(self, table_id):
        col_ids = []
        for col_id in self._col_parent:
            if self.get_parent_table_id(col_id) == table_id:
                col_ids.append(col_id)
        return col_ids

    def get_col_type(self, col_id):
        return self._col_types[col_id]

    def is_foreign_key(self, col_id):
        return col_id in [fk for fk, _ in self._foreign_primary_pairs]

    def get_table_id(self, table_name):
        return list(self._table_names_original.keys())[list(self._table_names_original.values()).index(table_name)]

    def get_col_id(self, parent_id, col_name):
        for key, item in self._col_names_original.items():
            if item == col_name and self._col_parent[key] == parent_id:
                return key
        raise RuntimeError('Should Not Reach Here')

    def get_col_items(self):
        return self._col_names.items()

    def get_table_items(self):
        return self._table_names.items()

    def is_neighbor_table(self, table_id_1, table_id_2):
        for c_id_1, c_id_2 in self.get_foreign_primary_pairs():
            t_id_1 = self.get_parent_table_id(c_id_1)
            t_id_2 = self.get_parent_table_id(c_id_2)
            if [t_id_1, t_id_2] == [table_id_1, table_id_2] or [t_id_2, t_id_1] == [table_id_1, table_id_2]:
                return True
        return False

    def get_table_relation_types(self, table_id_1, table_id_2):
        relation = None
        for c_id_1, c_id_2 in self.get_foreign_primary_pairs():
            t_id_1 = self.get_parent_table_id(c_id_1)
            t_id_2 = self.get_parent_table_id(c_id_2)
            if [t_id_1, t_id_2] == [table_id_1, table_id_2]:
                relation = 'tt_reversed' if not relation else 'tt_both'
            if [t_id_2, t_id_1] == [table_id_1, table_id_2]:
                relation = 'tt_foreign' if not relation else 'tt_both'
        return relation
