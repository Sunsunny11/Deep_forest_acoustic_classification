import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from pytorch.models import Cnn10, Cnn14, Cnn6, Cnn4, Cnn14_emb512, Cnn3, Cnn2, Cnn2_1024, Cnn3_bian, Cnn8


class PANNs_FeatureLayer(nn.Module):
    def __init__(self, model_type='Cnn10', **kwargs):
        super(PANNs_FeatureLayer, self).__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        kwargs['classes_num'] = 527
        self.model_type = model_type
        Model = eval(model_type)
        self.feature_layer = Model(**kwargs)
        checkpoint_path = 'panns_data/Cnn10_mAP=0.380.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.feature_layer.load_state_dict(checkpoint['model'])

    def forward(self, input):
        with torch.no_grad():
            self.eval()
            batch_output_dict = self.feature_layer(input)
        return batch_output_dict['embedding']

    @staticmethod
    def get_out_feature_size():
        return 512

class ASCFeatureLayer(nn.Module):
    def __init__(self, model_type, **kwargs):
        super(ASCFeatureLayer, self).__init__()
        self.model_type = model_type
        Model = eval(model_type)
        self.feature_layer = Model(**kwargs)

        # if self.model_type == 'Cnn10':
        #     self.feature_layer = Cnn10(**kwargs)
        # elif self.model_type == 'Cnn14':
        #     self.feature_layer == Cnn14(**kwargs)
        # elif self.model_type == 'Cnn6':
        #     self.feature_layer == Cnn6(**kwargs)
        #
        # self.fc = nn.Linear(1, 1)

    def get_out_feature_size(self):
        if self.model_type == 'Cnn10':
            return 2048
        elif self.model_type == 'Cnn14':
            return 2048
        elif self.model_type == 'Cnn6':
            return 512
        elif self.model_type == 'Cnn4':
            return 128
        elif self.model_type == 'Cnn14_emb512':
            return 1024
        elif self.model_type == 'Cnn3':
            return 256
        elif self.model_type == 'Cnn2':
            return 128
        elif self.model_type == 'Cnn2_1024':
            return 128
        elif self.model_type == 'Cnn3_bian':
            return 256
        elif self.model_type == 'Cnn8':
            return 1024

    def forward(self, input):
        return self.feature_layer(input)['embedding']


class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # leaf label distribution
        if jointly_training:
            self.pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=True)
        else:
            self.pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        # decision
        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask)  # ->[batch_size,n_used_feature]
        decision = self.decision(feats)  # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu * _decision  # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi

    def cal_prob(self, mu, pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi


class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree

        return prob


class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out = self.feature_layer(x)
        # out = out.view(x.size()[0], -1)
        clipwise_output = self.forest(out)

        output_dict = {'clipwise_output': clipwise_output}

        return output_dict
