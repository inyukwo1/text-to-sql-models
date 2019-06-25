import random
import sqlite3
from typing import List
from nltk import WordNetLemmatizer


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
        self._col_contents = dict()
        self._lemmatizer = WordNetLemmatizer()

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

    def import_contents(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.text_factory = str
        cursor = conn.cursor()
        for col_id, col_orig_name in self._col_names_original.items():
            table_id = self._col_parent[col_id]
            table_orig_name = self._table_names_original[table_id]
            cursor.execute("SELECT \"{}\" FROM \"{}\"".format(col_orig_name, table_orig_name))
            self._col_contents[col_id] = []
            while True:
                try:
                    content = cursor.fetchone()
                    if not content:
                        break
                    content = content[0]
                    if isinstance(content, str):
                        self._col_contents[col_id].append(content)
                except:
                    continue

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

    def get_neighbor_table_ids(self, table_id):
        neighbor_table_ids = set()
        for f, p in self._foreign_primary_pairs:
            foreign_parent = self.get_parent_table_id(f)
            primary_parent = self.get_parent_table_id(p)
            if table_id == foreign_parent:
                neighbor_table_ids.add(primary_parent)
            if table_id == primary_parent:
                neighbor_table_ids.add(foreign_parent)
        return list(neighbor_table_ids)

    def has_content(self, col_id, word: List[str]):
        contents = self._col_contents[col_id]
        if len(word) == 1:
            single = True
        else:
            single = False
        word = ' '.join(word)
        word_2 = ''.join(word)
        for content in contents:
            content = str(content)
            if content.lower() == word.lower():
                return True
            if content.lower() == word_2.lower():
                return True
            if single:
                if self._lemmatizer.lemmatize(content) == self._lemmatizer.lemmatize(word):
                    return True
        return False
