import os
import json
import torch
import torch.nn
import argparse
import datetime
from commons.utils import train, eval, parallel_train
import pyximport
import tensorflow as tf
import faulthandler; faulthandler.enable()
import torch.nn.parallel.data_parallel

if __name__ == '__main__':
    if torch.cuda.is_available():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            except RuntimeError as e:
                print(e)
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Model Name')
    parser.add_argument('--data_name', type=str, default='spider', help='Dataset Name')
    parser.add_argument('--param', type=str, default='parameters.json', help='json file containing parameters')
    parser.add_argument('--not_save', action='store_true')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    # Load Model
    if args.model_name == 'syntaxsql':
        from models.syntaxsql.syntaxsql import SyntaxSQL as Model
        H_PARAMS = json.loads(open('./models/syntaxsql/{}'.format(args.param)).read())

    elif args.model_name == 'typesql':
        from models.typesql.typesql import TypeSQL as Model
        H_PARAMS = json.loads(open('./models/typesql/{}'.format(args.param)).read())

    elif args.model_name == 'sqlnet':
        from models.sqlnet.sqlnet import SQLNet as Model
        H_PARAMS = json.loads(open('./models/sqlnet/{}'.format(args.param)).read())

    elif args.model_name == 'frompredictor':
        from models.frompredictor.from_predictor import FromPredictor as Model
        H_PARAMS = json.loads(open('./models/frompredictor/{}'.format(args.param)).read())
    elif args.model_name == 'generator':
        from models.frompredictor.generator import GeneratorWrapper as Model
        H_PARAMS = json.loads(open('./models/frompredictor/{}'.format(args.param)).read())
    elif args.model_name == 'groupgenerator':
        from models.frompredictor.groupgenerator import GroupGenerator as Model
        H_PARAMS = json.loads(open('./models/frompredictor/{}'.format(args.param)).read())
    else:
        print('Give correct model name!')
        exit(-1)

    if args.toy == True:
        H_PARAMS['toy'] = True
        H_PARAMS['gpu'] = False
    else:
        H_PARAMS['toy'] = False

    model = Model(H_PARAMS)
    # model.generator = torch.nn.DataParallel(model.generator)
    # model.load_model()

    # Load DataLoader
    if args.data_name == 'spider':
        from datasets.spider.data_loader import DataLoader
    else:
        print('Give correct dataset name!')
        exit(-1)

    load_option = H_PARAMS['load_option']if 'load_option' in H_PARAMS.keys() else None
    dataloader = DataLoader(H_PARAMS['batch_size'], args.toy)
    dataloader.load_data('train', load_option)

    # Prepare Optimizer
    model.optimizer = torch.optim.Adam(model.generator_params(), lr=H_PARAMS['lr'], weight_decay=H_PARAMS['weight_decay'])
    model.bert_optimizer = torch.optim.Adam(model.bert_params(), lr=H_PARAMS['bert_lr'], weight_decay=H_PARAMS['weight_decay'])

    # Epoch
    for epoch in range(H_PARAMS['epoch']):
        print('Epoch {} @ {} '.format(epoch + 1, datetime.datetime.now()), end='')
        # Training
        total_loss = train(model, dataloader)
        # total_loss = parallel_train(model, dataloader, 8)
        print('Loss: {}'.format(total_loss))

        # Evaluating
        if not epoch % H_PARAMS['eval_freq']:
            print('Evaluating...', end='')
            dataloader.batch_size = 2
            total_acc, total_loss = eval(model, dataloader)
            dataloader.batch_size = H_PARAMS['batch_size']
            if not args.not_save:
                # Save model if high acc
                model.save_model(total_loss)

