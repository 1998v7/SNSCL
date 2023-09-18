import torch.nn as nn
from models.resnet import *

class Classifier(nn.Module):
    def __init__(self, inputs, class_num):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(inputs, class_num)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier_layer(x)


def get_cls_layer(args):
    projector_dim = args.projector_dim
    num_class = args.num_classes
    classifier = Classifier(projector_dim, num_class).cuda()
    return classifier


def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
    else:
        network = resnet50
        feature_dim = 2048

    return network, feature_dim

