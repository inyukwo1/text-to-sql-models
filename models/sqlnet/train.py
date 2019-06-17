import os, sys
import torch
import argparse
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from baselines.sqlnet.model.sqlnet import SQLNet
from commons.embeddings.bert_container import BertContainer
from commons.utils import *
from commons.logger import Logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--bert', action='store_true',
            help='If set, use bert to encode question.')
    parser.add_argument('--use_from', action='store_true',
            help='Apply from predictor')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--use_new_dataset', action='store_true',
            help='use new train dev test datasplit')
    parser.add_argument('--dataset_dir', type=str, default='',
            help='to dataset directory where includes train, test and table json file.')
    parser.add_argument('--save_dir', type=str, default='saved_models/sqlnet_train/',
            help='set model save directory.')
    parser.add_argument('--gpuid', type=int, default=0,
            help='Tell which gpu to use')
    parser.add_argument('--log_tag', type=str, default='',
            help='Any text to differentiate log dir')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=False
        BATCH_SIZE=4
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=4
    if GPU:
        torch.device('cuda')
        torch.cuda.set_device(args.gpuid)
    else:
        torch.device('cpu')

    learning_rate = 1e-3
    bert_learning_rate = 1e-5

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, schemas = load_dataset(args.dataset_dir, args.use_new_dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word),
            load_used=args.train_emb, use_small=USE_SMALL)

    if args.use_from and args.bert:
        print('Using bert...')
        bert_model = BertContainer()
    else:
        bert_model = None

    model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, use_from=args.use_from, bert=bert_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

    init_acc = [[0], [0, 0, 0, 0, 0, 0]]
    best_sel_acc = init_acc[1][0]
    best_cond_acc = init_acc[1][1]
    best_group_acc = init_acc[1][2]
    best_order_acc = init_acc[1][3]
    best_from_acc = init_acc[1][4]
    best_tot_acc = 0.0

    tf_logger = Logger('logdir/sqlnet', args.log_tag, over_write=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(300):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        bert_model.train()
        print(' Loss = %s' % epoch_train(model, optimizer, BATCH_SIZE, sql_data, table_data, schemas, bert_model, use_from=args.use_from))
        
        if i % 8 == 0:
            bert_model.eval()
            val_tot_acc, val_bkd_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, schemas, None, use_from=args.use_from, perfect_from=False, error_print=False) # for detailed error analysis, pass True to error_print
            print(' Dev acc_qm: %s' % val_tot_acc)
            print(' Breakdown results: sel: %s, from: %s, cond: %s, group: %s, order: %s'\
                % (val_bkd_acc[0], val_bkd_acc[4], val_bkd_acc[1], val_bkd_acc[2], val_bkd_acc[3]))

            # Save models
            if val_bkd_acc[0] > best_sel_acc:
                best_sel_acc = val_bkd_acc[0]
                print("Saving sel model...")
                torch.save(model.sel_pred.state_dict(), os.path.join(args.save_dir, 'sel_models.dump'))
            if val_bkd_acc[1] > best_cond_acc:
                best_cond_acc = val_bkd_acc[1]
                print("Saving cond model...")
                torch.save(model.cond_pred.state_dict(), os.path.join(args.save_dir, 'cond_models.dump'))
            if val_bkd_acc[2] > best_group_acc:
                best_group_acc = val_bkd_acc[2]
                print("Saving group model...")
                torch.save(model.group_pred.state_dict(), os.path.join(args.save_dir, 'group_models.dump'))
            if val_bkd_acc[3] > best_order_acc:
                best_order_acc = val_bkd_acc[3]
                print("Saving order model...")
                torch.save(model.order_pred.state_dict(), os.path.join(args.save_dir, 'order_models.dump'))
            if val_bkd_acc[4] > best_from_acc:
                best_from_acc = val_bkd_acc[4]
                print("Saving from model...")
                if args.use_from:
                    torch.save(model.from_pred.state_dict(), os.path.join(args.save_dir, 'from_models.dump'))
                if args.bert:
                    torch.save(bert_model.main_bert.state_dict(), os.path.join(args.save_dir + "bert_from_models.dump"))
                    torch.save(bert_model.bert_param.state_dict(), os.path.join(args.save_dir + "bert_from_params.dump"))
            if val_tot_acc > best_tot_acc:
                best_tot_acc = val_tot_acc

            tf_logger.scalar_summary('Sel', val_bkd_acc[0], i)
            tf_logger.scalar_summary('cond', val_bkd_acc[1], i)
            tf_logger.scalar_summary('group', val_bkd_acc[2], i)
            tf_logger.scalar_summary('order', val_bkd_acc[3], i)
            tf_logger.scalar_summary('from', val_bkd_acc[4], i)
            tf_logger.scalar_summary('tot', val_tot_acc, i)

            print(' Best val sel = %s, from = %s, cond = %s, group = %s, order = %s, tot = %s' % (best_sel_acc, best_from_acc, best_cond_acc, best_group_acc, best_order_acc, best_tot_acc))
