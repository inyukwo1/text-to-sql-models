import random
import sqlite3
from typing import List
from nltk import WordNetLemmatizer
from copy import deepcopy

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
            self._col_contents[col_id] = set()
            while True:
                try:
                    content = cursor.fetchone()
                    if not content:
                        break
                    content = content[0]
                    if isinstance(content, str):
                        self._col_contents[col_id].add(content)
                except:
                    continue

    def col_num(self):
        return len(self._col_names) + 1

    def tab_num(self):
        return len(self._table_names)

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

    def get_all_col_ids(self):
        return list(self._col_names)

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

    def get_foreign_primary_keys_with_two_tables(self, table_id_1, table_id_2):
        foreign_primary_keys = []
        for f, p in self._foreign_primary_pairs:
            foreign_parent = self.get_parent_table_id(f)
            primary_parent = self.get_parent_table_id(p)
            if table_id_1 == foreign_parent and table_id_2 == primary_parent:
                foreign_primary_keys.append([f, p])
            elif table_id_1 == primary_parent and table_id_2 == foreign_parent:
                foreign_primary_keys.append([p, f])
        return foreign_primary_keys

    def has_content(self, col_id, word: List[str]):
        contents = self._col_contents[col_id]
        if len(word) == 1:
            single = True
        else:
            single = False
        word = ' '.join(word)
        word_2 = ''.join(word)
        if word in contents:
            return True
        if word_2 in contents:
            return True
        if single:
            if self._lemmatizer.lemmatize(word) in contents:
                return True
        return False

    def make_linking(self, words: List[str]):
        def lemmatize(word):
            new_word = []
            for tok in word.split(' '):
                new_word.append(self._lemmatizer.lemmatize(tok))
            return ' '.join(new_word)

        words = [self._lemmatizer.lemmatize(word) for word in words]
        table_names = [(table_id, lemmatize(self._table_names[table_id])) for table_id in self._table_names]
        col_names = [(col_id, lemmatize(self._col_names[col_id])) for col_id in self._col_names]
        done_checking = [False] * len(words)
        linking = dict()
        col_linking = dict()
        for col_id in self._col_names:
            col_linking[col_id] = "NONE"
        tab_linking = dict()
        for tab_id in self._table_names:
            tab_linking[tab_id] = "NONE"
        q_ranges = set()
        for n_gram in range(6, 0, -1):
            for st in range(0, len(words) - n_gram):
                if True in done_checking[st:st+n_gram]:
                    continue
                subword = ' '.join(words[st:st+n_gram])
                for (tab_id, tab_name) in table_names:
                    if subword == tab_name:
                        # table exact matching
                        linking[st] = "TABLE"
                        done_checking[st:st + n_gram] = [True] * n_gram
                        tab_linking[tab_id] = "EXACT"
                        q_ranges.add((st, st+n_gram))
                    elif set(subword.split(' ')).issubset(set(tab_name.split(' '))):
                        # table partial matching
                        linking[st] = "TABLE"
                        done_checking[st:st + n_gram] = [True] * n_gram
                        tab_linking[tab_id] = "PARTIAL"
                        q_ranges.add((st, st+n_gram))

                for (col_id, col_name) in col_names:
                    if subword == col_name:
                        # column exact matching
                        linking[st] = "COLUMN"
                        done_checking[st:st + n_gram] = [True] * n_gram
                        col_linking[col_id] = "EXACT"
                        q_ranges.add((st, st+n_gram))
                    elif set(subword.split(' ')).issubset(set(col_name.split(' '))):
                        # column partial matching
                        linking[st] = "COLUMN"
                        done_checking[st:st + n_gram] = [True] * n_gram
                        col_linking[col_id] = "PARTIAL"
                        q_ranges.add((st, st+n_gram))

                if subword[0] == "'" and subword[-1] == "'":
                    linking[st] = "VALUE"
                    done_checking[st:st + n_gram] = [True] * n_gram
                    q_ranges.add((st, st+n_gram))

        q_ranges = list(q_ranges)

        return linking, tab_linking, col_linking, q_ranges
