from asc_dataset import get_asc_dataloader, get_esc_dataloader
import torch.optim as optim
from dNDF import NeuralDecisionForest
import torch
from pytorch.losses import get_loss_func
import os
import time
import logging
import numpy as np
import torch.nn as nn
from utils.utilities import (create_folder, get_filename, create_logging, Mixup,
                             StatisticsContainer)
from pytorch.pytorch_utils import (move_data_to_device, count_parameters, count_flops,
                                   do_mixup)
from pytorch.models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout,
                            Cnn6, Cnn4, Cnn3, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128,
                            Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19,
                            Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14,
                            Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128,
                            Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt, Cnn2, Cnn2_1024, Cnn3_bian, Cnn8)
from pytorch.evaluate import Evaluator
from early_stopping import EarlyStopping
from sklearn import metrics
from tqdm import tqdm
from dNDF import ASCFeatureLayer, Forest, NeuralDecisionForest, PANNs_FeatureLayer
#from linearSVM import multiClassHingeLoss
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch.AudioTransformer import AudioTransformer


def train(args):
    jointly_training = False
    epoch = args.epoch
    # Arugments & parameters
    workspace = args.workspace
    ####asc
    sample_rate = 48000
    #####esc-50
    #sample_rate = 41000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    model_type = args.model_type
    loss_type = 'clip_bce'
    # augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    # resume_iteration = args.resume_iteration
    # early_stop = args.early_stop
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ######asc
    classes_num = 10
    ######esc-50
    #classes_num = 50
    loss_func = get_loss_func(loss_type)
    exp_name = args.exp_name
    cls_type = args.cls_type
    n_tree = args.n_tree
    tree_depth = args.tree_depth
    tree_feature_rate = args.tree_feature_rate

    checkpoints_dir = os.path.join(workspace, exp_name, 'checkpoints')
    create_folder(checkpoints_dir)

    logs_dir = os.path.join(workspace, exp_name, 'logs')

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    ######loader asc dataset
    train_loader = get_asc_dataloader(split='train', batch_size=batch_size, shuffle=True)
    test_loader = get_asc_dataloader(split='test', batch_size=batch_size)
    ######loader esc dataset
    #train_loader = get_esc_dataloader(data_type='train', test_fold_num=4, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    #test_loader = get_esc_dataloader(data_type='test', test_fold_num=4, batch_size=batch_size)

    model_params = {'sample_rate': sample_rate,
                    'window_size': window_size,
                    'hop_size': hop_size,
                    'mel_bins': mel_bins,
                    'fmin': fmin,
                    'fmax': fmax,
                    'classes_num': classes_num}

    ######Multiple self-attention

    if cls_type == 'fc':
        Model = eval(model_type)
        model = Model(**model_params)

    elif cls_type == 'forest':
        feature_layer = ASCFeatureLayer(model_type=model_type, **model_params)
        #audioAttention = AudioTransformer(dim=batch_size, depth=1, heads=2, mlp_dim=1024, dim_head=batch_size, dropout=0.1)
        #audioAttention = AudioTransformer(dim=512, depth=1, heads=2, mlp_dim=1024, dim_head=512,dropout=0.1)
        forest_params = {'n_tree': n_tree,
                         'tree_depth': tree_depth,
                         'n_in_feature': feature_layer.get_out_feature_size(),
                         'tree_feature_rate': tree_feature_rate,
                         'n_class': 10,    #10
                         'jointly_training': jointly_training}
        forest = Forest(**forest_params)
        #model = NeuralDecisionForest(feature_layer, forest, audioAttention)
        model = NeuralDecisionForest(feature_layer, forest)
        print(model)
       #model = nn.DataParallel(model)


    if 'cuda' in str(device):
        model.to(device)

    # Evaluator
    evaluator = Evaluator(model=model)

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6,
                           amsgrad=True)

    earlystopping = EarlyStopping(patience=50, verbose=True)

    # Start training ...
    for epoch in range(epoch):
        epoch = epoch + 1
        if not jointly_training:
            print("Epoch %d : Two Stage Learing - Update PI" % (epoch))
            feat_batches = []
            target_batches = []
            train_pbar = tqdm(train_loader)
            train_pbar.set_description(f'Training at epoch: {epoch}')
            with torch.no_grad():
                for batch_data_dict in train_pbar:
                    batch_data_dict['waveform'] = batch_data_dict['waveform'].to(device)
                    batch_data_dict['target'] = batch_data_dict['target'].to(device)
                    #batch_data_dict['waveform'] = Variable(batch_data_dict['waveform'])
                    feats = feature_layer(batch_data_dict['waveform']).to(device)
                    #compute the attention weight for feats and then take K features with the max attention weight value
                    feats = torch.unsqueeze(feats, 0).to(device)       #tensor([6, 1, 512])  [1, 6, 512]
                    feats = feats.transpose(2, 1)                      #tensor[1, 512, 6]
                    audioAttention = AudioTransformer(dim=feats.shape[2], depth=1, heads=2, mlp_dim=1024,
                                                      dim_head=feats.shape[2], dropout=0.1)
                    feats2 = audioAttention(feats).to(device)           #tensor([6, 1, 512])
                                                     # dim = dim_head = 20 here, dim=noncomputeattention_matrix
                    feats2 = torch.squeeze(feats2, 0).to(device)         #tensor([512, 6])
                    feats = feats2.transpose(1, 0)       #tensor([6, 512])

                    #if feats2.shape[0] == feats.shape[2]:
                        #feats = feats2
                    #else:
                        #f = feats.shape[0]
                        #idx = f-1
                        #feats = feats2[:idx, :]
                    target = batch_data_dict['target']
                    feat_batches.append(feats)
                    target_batches.append(target)

        #Update \Pi for each tree
                for tree in model.forest.trees:
                    mu_batches = []
                    for feats in feat_batches:
                        mu = tree(feats)    # [batch_size,n_leaf]
                        mu_batches.append(mu)
                    for _ in range(20):                 #20
                       new_pi = torch.zeros((tree.n_leaf, tree.n_class))  # Tensor [n_leaf,n_class]
                       new_pi = new_pi.cuda()
                       for mu, target in zip(mu_batches, target_batches):
                           pi = tree.get_pi()  # [n_leaf,n_class]
                           prob = tree.cal_prob(mu, pi)  # [batch_size,n_class]
                    #Variable to Tensor
                           pi = pi.data        # [n_leaf,n_class]
                           prob = prob.data     # [batch_size,n_class]
                           mu = mu.data         #[batch_size,n_leaf]

                           #prob1 = prob.shape[0]
                           #mu1 = mu.shape[0]
                           #target1 = target.shape[0]

                           #if prob1 == batch_size:
                           #   prob = prob
                           #else:
                              #probxx = torch.zeros(1, tree.n_class).to(device)
                              #prob = torch.cat((prob, probxx), dim=0)

                           #if mu1 == batch_size:
                              #mu = mu
                           #else:
                              #muxx = torch.zeros(1, tree.n_leaf).to(device)
                              #mu = torch.cat((mu, muxx), dim=0)

                           #if target1 == batch_size:
                              #target = target
                           #else:
                              #targetxx = torch.zeros(1, tree.n_class).to(device)
                              #target = torch.cat((target, targetxx), dim=0)

                           _train_target = target.unsqueeze(1)  # [batch_size,1,n_class]
                           _pi = pi.unsqueeze(0)    # [1,n_leaf,n_class]
                           _mu = mu.unsqueeze(2)    # [batch_size,n_leaf,1]
                           _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

                           #_train_target1 = _train_target.shape[0]
                           #if _train_target1 == batch_size:
                               #_train_target = _train_target
                           #else:
                               #xx = torch.zeros(512, 1).to(device)
                               #x = torch.cat((x, xx), dim=1)

                           _new_pi = torch.mul(torch.mul(_train_target, _pi), _mu) / _prob  # [batch_size,n_leaf,n_class]
                           _new_pi1 = _new_pi.shape[0]

                           #if _new_pi1 == batch_size:
                               #_new_pi = _new_pi
                           #else:
                               #_new_pixx = torch.zeros(1, tree.n_leaf, tree.n_class).to(device)
                              # _new_pi = torch.cat((_new_pi, _new_pixx), dim=0)

                           new_pi += torch.sum(_new_pi, dim=0)

                       new_pi = F.softmax(Variable(new_pi), dim=1).data
                       tree.update_pi(new_pi)

            # Forward Update \Theta
        model.train()  # batch_feature = feature_layer(batch_data_dict['waveform']) [batch, 32000] -> [batch, feature_num (512)]
        for batch_data_dict in train_pbar:
            batch_data_dict['waveform'] = batch_data_dict['waveform'].to(device)
            batch_data_dict['target'] = batch_data_dict['target'].to(device)

            batch_output_dict = model(batch_data_dict['waveform'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""
            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

            # Loss
            if cls_type == 'fc':
                loss = loss_func(batch_output_dict, batch_target_dict)
            elif cls_type == 'forest':
                batch_target = torch.argmax(batch_target_dict['target'], axis=1)
                #batch_target1 = batch_target.shape[0]
                #if batch_target1 == batch_size:
                   #batch_target = torch.argmax(batch_target_dict['target'], axis=1)
               # else:
                    #batch_targetxx = torch.zeros(1, dtype=torch.int).to(device)
                    #batch_target = torch.cat((batch_target, batch_targetxx), dim=0)

                batch_output = batch_output_dict['clipwise_output']
                loss = F.nll_loss(torch.log(batch_output), batch_target)

            # Backward
            loss.backward()
            # print(loss)

            optimizer.step()
            optimizer.zero_grad()

        #model.eval()
        #test_loss = 0
        #correct = 0
        #average_precision = 0
        #test_pbar = tqdm(test_loader)
        #test_pbar.set_description(f'test')
        #with torch.no_grad():
            #for batch_data_dict in test_pbar:
                #batch_data_dict['waveform'] = batch_data_dict['waveform'].to(device)
                #data = Variable(batch_data_dict['waveform'])
                #batch_data_dict['target'] = batch_data_dict['target'].to(device)
                #batch_test_target = torch.argmax(batch_data_dict['target'], axis=1)
                #target = Variable(batch_test_target)
                #batch_output = model(batch_data_dict['waveform'])
                #output = batch_output['clipwise_output']
                #test_loss += F.nll_loss(torch.log(output), batch_test_target, size_average=False).item()
                #pred = output.max(1, keepdim=True)[1]
                #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                ####compute mAP
                #output2 = output.cpu()
                #pred2 = pred.cpu()
                #average_precision += metrics.average_precision_score(pred2, output2, average='macro')

            #test_loss /= len(test_loader)
            #logging.info('Accuracy: {}/{} ({:.6f})\n'.format(test_loss, correct, len(test_loader), correct / len(test_loader)))
            #logging.info('Validate test mAP: {:.3f}'.format(average_precision))

        # Evaluate
        test_statistics = evaluator.evaluate(test_loader)
        ave_precision = np.mean(test_statistics['average_precision'])
        acc = test_statistics['acc']
        each_acc = test_statistics['each_acc']

        logging.info('Validate test mAP: {:.3f}'.format(ave_precision))
        logging.info('Validate test acc: {:.3f}'.format(acc))
        logging.info('each_acc: {}'.format(each_acc))

        #val_loss = -ave_precision
        val_loss = -acc
        ckpt_path = os.path.join(checkpoints_dir, 'best.pth')
        earlystopping(val_loss, model, optimizer, epoch, ckpt_path)
        #earlystopping(test_loss, model, optimizer, epoch, ckpt_path)
        if earlystopping.early_stop:
            break

            # train_fin_time = time.time()
            # train_time = train_fin_time - train_bgn_time
            # validate_time = time.time() - train_fin_time
            # logging.info(
            #     'epoch: {}, train time: {:.3f} s, validate time: {:.3f} s'
            #     ''.format(epoch, train_time, validate_time))
            #
            # logging.info('------------------------------------')
            # train_bgn_time = time.time()
def save_model(model, optimizer, step, acc, each_acc, exp_name, ckpt_path):
    save_path = os.path.join(ckpt_path, exp_name + '.pt')
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': step,
    'acc': acc,
    'each_acc': each_acc
    }, save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--model_type', type=str, default='Cnn6')
    parser.add_argument('--cls_type', type=str, default='forest')  ### decide whether using fc or forest
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # config for dNDF
    parser.add_argument('--n_tree', type=int, default=100)  #100
    parser.add_argument('--tree_depth', type=int, default=5)
    parser.add_argument('--tree_feature_rate', type=float, default=1)
    parser.add_argument('--jointly_training', action='store_true', default=False)

    args = parser.parse_args()

    train(args)

  
