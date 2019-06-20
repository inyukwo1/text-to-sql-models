import random


class Schema:
    def __init__(self):
        self.db_id = ""
        self._table_names = dict()
        self._table_names_original = dict()
        self._col_names = dict()
        self._col_names_original = dict()
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
            if par_tab != -1:
                self._col_names[col_num] = col_name
                self._col_parent[col_num] = par_tab
        for col_num, (par_tab, col_name_original) in enumerate(spider_schema["column_names_original"]):
            if par_tab != -1:
                self._col_names_original[col_num] = col_name_original
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
        return self._table_names[table_id]

    def get_col_name(self, col_id):
        return self._col_names[col_id]

    def get_child_col_ids(self, table_id):
        col_ids = []
        for col_id in self._col_parent:
            if self.get_parent_table_id(col_id) == table_id:
                col_ids.append(col_id)
        return col_ids