def append_dup_cols(cols_list, sql):
    for from_conds in sql['from']['conds']:
        if from_conds == 'and':
            continue
        not_op, op_id, val_unit, val1, val2 = from_conds
        if op_id == 2:
            unit_op, col_unit1, col_unit2 = val_unit
            agg_id, col_id, isDistinct = col_unit1
            _, col_id2, isDistinct = val1
            if col_id in cols_list:
                cols_list.append(col_id2)
            elif col_id2 in cols_list:
                cols_list.append(col_id)
    return list(set(cols_list))


def check_dup_cols(cols_list, sql, col):
    for from_conds in sql['from']['conds']:
        not_op, op_id, val_unit, val1, val2 = from_conds
        if op_id == 2:
            unit_op, col_unit1, col_unit2 = val_unit
            agg_id, col_id, isDistinct = col_unit1
            _, col_id2, isDistinct = val1
            if col_id in cols_list and col == col_id2:
                return True
            if col_id2 in cols_list and col == col_id:
                return True
    return False


def get_select_cols_from_sql(sql):
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
    for val_unit in sql["select"][1]:
        agg_id, val_unit = val_unit
        yield from yield_for_val_unit(val_unit)

    # yield from get_select_cols_from_sql(sql["except"])
    # yield from get_select_cols_from_sql(sql["intersect"])
    # yield from get_select_cols_from_sql(sql["union"])

def get_select_tab_from_sql(sql, col_to_tab, question, query):
    if len(sql['from']['table_units']) == 1:
        return sql['from']['table_units'][0][1]
    else:
        table_list = []
        for tmp_t in sql['from']['table_units']:
            if type(tmp_t[1]) == int:
                table_list.append(tmp_t[1])
        table_set, other_set = set(table_list), set()
        for sel_p in sql['select'][1]:
            if sel_p[1][1][1] != 0:
                other_set.add(col_to_tab[sel_p[1][1][1]])

        if len(sql['where']) == 1:
            other_set.add(col_to_tab[sql['where'][0][2][1][1]])
        elif len(sql['where']) == 3:
            other_set.add(col_to_tab[sql['where'][0][2][1][1]])
            other_set.add(col_to_tab[sql['where'][2][2][1][1]])
        elif len(sql['where']) == 5:
            other_set.add(col_to_tab[sql['where'][0][2][1][1]])
            other_set.add(col_to_tab[sql['where'][2][2][1][1]])
            other_set.add(col_to_tab[sql['where'][4][2][1][1]])
        table_set = table_set - other_set
        if len(table_set) == 1:
            return list(table_set)[0]
        elif len(table_set) == 0 and sql['groupBy'] != []:
            return col_to_tab[sql['groupBy'][0][1]]
        else:
            print('column * table error: question: {}, query: {}'.format(question, query))
            return sql['from']['table_units'][0][1]


def get_group_cols_from_sql(sql):
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
    for col_unit in sql["groupBy"]:
        yield from yield_for_col_unit(col_unit)

    yield from get_group_cols_from_sql(sql["except"])
    yield from get_group_cols_from_sql(sql["intersect"])
    yield from get_group_cols_from_sql(sql["union"])
