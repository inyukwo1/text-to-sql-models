import os
import json
import torch
import argparse
import datetime
from commons.utils import train, eval

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Model Name')
    parser.add_argument('--data_name', type=str, default='', help='Dataset Name')
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

    else:
        print('Give correct model name!')
        exit(-1)

    if args.toy == True:
        H_PARAMS['toy'] = True
        H_PARAMS['gpu'] = False
    model = Model(H_PARAMS)

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
    model.optimizer = torch.optim.Adam(model.parameters(), lr=H_PARAMS['lr'], weight_decay=H_PARAMS['weight_decay'])

    # Epoch
    for epoch in range(H_PARAMS['epoch']):
        print('Epoch {} @ {} '.format(epoch + 1, datetime.datetime.now()), end='')

        # Training
        total_loss = train(model, dataloader)
        print('Loss: {}'.format(total_loss))

        # Evaluating
        if not epoch % H_PARAMS['eval_freq']:
            print('Evaluating...', end='')
            total_acc = eval(model, dataloader)
            if not args.not_save:
                # Save model if high acc
                model.save_model(total_acc)

