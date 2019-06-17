import torch.nn as nn
from models.syntaxsql.modules.agg_predictor import AggPredictor
from models.syntaxsql.modules.col_predictor import ColPredictor
from models.syntaxsql.modules.desasc_limit_predictor import DesAscLimitPredictor
from models.syntaxsql.modules.having_predictor import HavingPredictor
from models.syntaxsql.modules.keyword_predictor import KeyWordPredictor
from models.syntaxsql.modules.multisql_predictor import MultiSqlPredictor
from models.syntaxsql.modules.root_teminal_predictor import RootTeminalPredictor
from models.syntaxsql.modules.andor_predictor import AndOrPredictor
from models.syntaxsql.modules.op_predictor import OpPredictor
from models.syntaxsql.modules.supermodel import SuperModel
from commons.embeddings.word_embedding import WordEmbedding

SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']


class SyntaxSQL(nn.Module):
    def __init__(self, H_PARAM):
        super(SyntaxSQL, self).__init__()
        self.model = None
        self.optimizer = None
        self._set_model(H_PARAM)
        self.optimizer = None

        self.acc = 0.0
        self.acc_num = 1
        self.save_dir = H_PARAM['save_dir']

    def _set_model(self, H_PARAM):
        model_name = H_PARAM['train_component']

        # word embedding layer
        embed_layer = WordEmbedding(H_PARAM['glove_path'].format(H_PARAM['B_WORD'], H_PARAM['N_WORD']), H_PARAM['N_WORD'],
                                         H_PARAM['gpu'], SQL_TOK, H_PARAM['use_bert'], trainable=H_PARAM['trainable_emb'])

        if model_name == "multi_sql":
            self.model = MultiSqlPredictor(H_PARAM, embed_layer)
        elif model_name == "keyword":
            self.model = KeyWordPredictor(H_PARAM, embed_layer)
        elif model_name == "col":
            self.model = ColPredictor(H_PARAM, embed_layer)
        elif model_name == "op":
            self.model = OpPredictor(H_PARAM, embed_layer)
        elif model_name == "agg":
            self.model = AggPredictor(H_PARAM, embed_layer)
        elif model_name == "root_tem":
            self.model = RootTeminalPredictor(H_PARAM, embed_layer)
        elif model_name == "des_asc":
            self.model = DesAscLimitPredictor(H_PARAM, embed_layer)
        elif model_name == "having":
            self.model = HavingPredictor(H_PARAM, embed_layer)
        elif model_name == "andor":
            self.model = AndOrPredictor(H_PARAM, embed_layer)
        else:
            self.model = SuperModel(H_PARAM, embed_layer)

    def forward(self, input_data):
        return self.model.forward(input_data)

    def loss(self, score, gt_data):
        return self.model.loss(score, gt_data)

    def step(self):
        self.optimizer.step()

    def evaluate(self, score, gt_data):
        return self.model.evaluate(score, gt_data)

    def gen_sql(self, sql, gt_data):
        return self.model.gen_sql(sql, gt_data)

    def preprocess(self, batch):
        return self.model.preprocess(batch)

    def save_model(self, acc):
        if acc > self.acc:
            self.acc = acc
            self.model.save_model(self.save_dir)

    def load_model(self):
        self.model.load_model()

    def set_test_table_dict(self, schemas, test_data):
        self.model.set_test_table_dict(schemas, test_data)
