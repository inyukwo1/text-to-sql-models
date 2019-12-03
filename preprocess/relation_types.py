# q - question, c - column, t - table
RELATION_LIST = [
    '[PAD]',
    'cc_identical', 'cc_sibling', 'cc_foreign_primary', 'cc_primary_foreign', 'cc_etc',
    'ct_primary_child', 'ct_child', 'ct_etc',
    'tc_primary_child', 'tc_child', 'tc_etc',
    'tt_identical', 'tt_foreign', 'tt_reversed', 'tt_both', 'tt_etc',
    'qt_exact', 'qt_partial', 'qt_no', 'qc_exact', 'qc_partial', 'qc_db', 'qc_no',
    'tq_exact', 'tq_partial', 'tq_no', 'cq_exact', 'cq_partial', 'cq_db', 'cq_no',
    'qq_-2', 'qq_-1', 'qq_0', 'qq_1', 'qq_2',
    'cls_q', 'q_cls', 'cls_c', 'c_cls', 'cls_t', 't_cls', 'cls_cls'
]

# Dictionary
RELATION_TYPE = {key: idx for idx, key in enumerate(RELATION_LIST)}
N_RELATIONS = len(RELATION_TYPE)
