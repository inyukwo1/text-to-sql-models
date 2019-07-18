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

    yield from get_select_cols_from_sql(sql["except"])
    yield from get_select_cols_from_sql(sql["intersect"])
    yield from get_select_cols_from_sql(sql["union"])
