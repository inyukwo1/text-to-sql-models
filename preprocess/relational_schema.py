from schema import Schema
from relation_types import RELATION_TYPE


class RelationalSchema(Schema):
    def __init__(self):
        super(RelationalSchema, self).__init__()
        self._entity_relations = []
        self._entity_id_original = dict()
        self._entity_id = dict()

    def _construct_entity_dic(self, col_dic, table_dic):
        tmp_dic = dict()
        idx_cnt = 0

        # Column
        for col_id, col_name in col_dic.items():
            parent_id = self.get_parent_table_id(col_id)
            parent = self.get_original_table_name(parent_id)
            key = '.'.join([parent, col_name])
            assert key not in tmp_dic.keys()
            tmp_dic[key] = idx_cnt
            idx_cnt += 1
        # Table
        for table_id, table_name in table_dic.items():
            tmp_dic[table_name] = idx_cnt
            idx_cnt += 1

        return tmp_dic

    def import_from_spider(self, spider_schema):
        super().import_from_spider(spider_schema)

        assert len(self._col_names) == len(self._col_names_original)
        assert len(self._col_names) == len(self._col_parent)
        assert len(self._table_names) == len(self._table_names_original)

        # Create dictionary for entity idx (key: name, value: idx in relation matrix)
        self._entity_id_original = self._construct_entity_dic(self._col_names_original, self._table_names_original)
        self._entity_id = self._construct_entity_dic(self._col_names, self._table_names)

        # Get relation matrix
        '''
        Matrix of identical row and column:
            (column1_1, column1_2, ... column1_N, column2_N, ... columnK_N, table_1, table_2, ... table_M)

        need dictionary for knowing index of table and columns for the matrix.
        need to index the dictionary with table.column (due to repeated names in columns)
        '''
        self._entity_relations = [[[0] for _ in range(len(self._entity_id_original))] for _ in range(len(self._entity_id_original))]

        for key_1, idx_1 in self._entity_id_original.items():
            is_table_1 = '.' not in key_1
            if not is_table_1 and key_1.split('.')[0] == '*':
                parent_id_1 = None
                item_id_1 = 0
            else:
                parent_id_1 = None if is_table_1 else self.get_table_id(key_1.split('.')[0])
                item_id_1 = self.get_table_id(key_1) if is_table_1 else self.get_col_id(parent_id_1, key_1.split('.')[1])

            for key_2, idx_2 in self._entity_id_original.items():
                is_table_2 = '.' not in key_2
                if not is_table_2 and key_2.split('.')[0] == '*':
                    parent_id_2 = None
                    item_id_2 = 0
                else:
                    parent_id_2 = None if is_table_2 else self.get_table_id(key_2.split('.')[0])
                    item_id_2 = self.get_table_id(key_2) if is_table_2 else self.get_col_id(parent_id_2, key_2.split('.')[1])

                # Find Relation Type
                if is_table_1:
                    if is_table_2:
                        # Table - Table
                        if item_id_1 == item_id_2:
                            relation_type = RELATION_TYPE['tt_identical']
                        # Need to elaborate this
                        elif self.is_neighbor_table(item_id_1, item_id_2):
                            relation = self.get_table_relation_types(item_id_1, item_id_2)
                            assert relation in RELATION_TYPE, 'unknown table-table relation!'
                            relation_type = RELATION_TYPE[relation]
                        else:
                            relation_type = RELATION_TYPE['tt_etc']
                    else:
                        # Table - Column
                        if self.get_parent_table_id(item_id_2) == item_id_1 and self.is_primary_key(item_id_2):
                            relation_type = RELATION_TYPE['tc_primary_child']
                        elif self.get_parent_table_id(item_id_2) == item_id_1:
                            relation_type = RELATION_TYPE['tc_child']
                        else:
                            relation_type = RELATION_TYPE['tc_etc']
                else:
                    if is_table_2:
                        # Column - Table
                        if self.get_parent_table_id(item_id_1) == item_id_2 and self.is_primary_key(item_id_1):
                            relation_type = RELATION_TYPE['ct_primary_child']
                        elif self.get_parent_table_id(item_id_1) == item_id_2:
                            relation_type = RELATION_TYPE['ct_child']
                        else:
                            relation_type = RELATION_TYPE['ct_etc']
                    else:
                        # Column - Column
                        if item_id_1 == item_id_2:
                            relation_type = RELATION_TYPE['cc_identical']
                        elif [item_id_1, item_id_2] in self.get_foreign_primary_pairs():
                            relation_type = RELATION_TYPE['cc_foreign_primary']
                        elif [item_id_2, item_id_1] in self.get_foreign_primary_pairs():
                            relation_type = RELATION_TYPE['cc_primary_foreign']
                        elif self.get_parent_table_id(item_id_1) == self.get_parent_table_id(item_id_2):
                            relation_type = RELATION_TYPE['cc_sibling']
                        else:
                            relation_type = RELATION_TYPE['cc_etc']
                self._entity_relations[idx_1][idx_2] = relation_type

    def get_entity_relations(self):
        return self._entity_relations

    # To-Do: change name to get_entity_size
    def get_entity_num(self):
        return len(self._entity_id_original)

    def get_entity_id(self, name):
        return self._entity_id_original[name]

    def get_entity_original_name(self, id):
        entity = {id: key for key, id in self._entity_id_original.items()}
        return entity[id]

    def get_entity_name(self, id):
        entity = {id: key for key, id in self._entity_id.items()}
        return entity[id]

    def get_entity_type(self, id):
        return self._col_types[id] if id in self._col_types else None

    def get_all_entity_original_name(self):
        original_entity_names = [self.get_entity_original_name(id) for id in range(self.get_entity_num())]
        return original_entity_names

    def get_all_entity_name(self):
        entity_names = [self.get_entity_name(id) for id in range(self.get_entity_num())]
        return entity_names