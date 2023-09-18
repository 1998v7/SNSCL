from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from sklearn.mixture import GaussianMixture
import warnings
from helpers import *
from models.backbone import *
from dataloader.select_helper import my_dataloder_helper
from torch_ema import ExponentialMovingAverage

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--num_batches', default=1000, type=int)

parser.add_argument('--result_dir', type=str,  default='dividemix')
parser.add_argument('--gpu', default='3', type=str)
parser.add_argument('--noise', default=True, type=bool)
parser.add_argument('--noise_r', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--pretrain', default=True, type=bool)
parser.add_argument('--lr_ratio', default=10, type=float)
# contrastive learing
parser.add_argument('--projector_dim', default=1024, type=int)
parser.add_argument('--dataset', default='aircraft', type=str, choices=['aircraft', 'dog', 'cub', 'car'])
parser.add_argument('--lam_cls', default=0.1, type=float)
parser.add_argument('--loss', type=str, default='ce_loss', choices=['ce_loss', 'APL', 'Asym', 'GCE', 'Sym', 'label_smooth', 'confPenalty'])
parser.add_argument('--noise_type', default='sym', type=str, choices=['sym', 'asym'])
args = parser.parse_args()
set_env(args)


def warmup(my_model, classifier, optimizer, dataloader):
    my_model.train()
    classifier.train()
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        _, feature = my_model.encoder_q(inputs)
        outputs = classifier(feature)
        loss = CEloss(outputs, labels)

        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('| Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  ' % (batch_idx + 1, len(dataloader), loss.item()))
        sys.stdout.flush()
    sys.stdout.write('\n')


def test(epoch, my_model, classifier, test_loader):
    my_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, feature = my_model.encoder_q(inputs)
            outputs = classifier(feature)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("| Epoch %d  Test Acc: %.2f%%" % (epoch, acc))
    return acc


def eval_train(my_model, classifier, eval_loader):
    my_model.eval()
    classifier.eval()

    num_samples = len(eval_loader.dataset)
    losses = torch.zeros(num_samples)
    paths = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, feature = my_model.encoder_q(inputs)
            outputs = classifier(feature)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                paths.append(index[b])
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| ==== Val train losses on Net ====  Iter %3d\t' % (batch_idx))
            sys.stdout.flush()
    sys.stdout.write('\n')
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    return prob, paths


def train(epoch, my_model, classifier, optimizer, labeled_trainloader, unlabeled_trainloader):
    my_model.train()
    classifier.train()

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        # data
        img_labeled_q = inputs_x.cuda()
        img_labeled_k = inputs_x2.cuda()
        label = labels_x.cuda()

        try:
            inputs_u, inputs_u2, target_u, w_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, target_u, w_u = unlabeled_train_iter.next()

        img_unlabeled_q = inputs_u.cuda()
        img_unlabeled_k = inputs_u2.cuda()
        w_u = w_u.view(-1, 1).type(torch.FloatTensor).cuda()

        ## For Labeled Data
        SNSCL_logit_labeled, SNSCL_label_labeled, feat_labeled, sampler_loss1 = my_model(img_labeled_q, img_labeled_k, label)
        SNSCL_loss_labeled = criterions['KLDiv'](SNSCL_logit_labeled, SNSCL_label_labeled)
        out = classifier(feat_labeled)

        ## For Unlabeled Data
        q_c_unlabeled, q_f_unlabeled = my_model.encoder_q(img_unlabeled_q)
        logit_unlabeled = classifier(q_f_unlabeled)
        prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
        confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)

        SNSCL_logit_unlabeled, SNSCL_label_unlabeled, feat_unlabeled, sampler_loss2 = my_model(img_unlabeled_q, img_unlabeled_k, predict_unlabeled, w_u)
        SNSCL_loss_unlabeled = criterions['KLDiv'](SNSCL_logit_unlabeled, SNSCL_label_unlabeled)

        out2 = classifier(feat_unlabeled)
        prob_u = torch.softmax(out2.detach(), dim=-1)
        _, pred_u = torch.max(prob_u, dim=-1)

        target_x = torch.zeros(args.batch_size, args.num_classes).scatter_(1, labels_x.view(-1, 1), 1).cuda()
        target_u = torch.zeros(args.batch_size, args.num_classes).scatter_(1, target_u.view(-1, 1), 1).cuda()
        pred_u = torch.zeros(args.batch_size, args.num_classes).scatter_(1, pred_u.cpu().view(-1, 1), 1).cuda()
        target_u = w_u * target_u + (1 - w_u) * pred_u

        out = torch.cat([out, out2], dim=0)
        label = torch.cat([target_x, target_u], dim=0)
        classifier_loss = robust_loss(out, label)

        total_loss = args.lam_cls * classifier_loss + SNSCL_loss_labeled + SNSCL_loss_unlabeled + sampler_loss1 + sampler_loss2
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Train ours | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f\t  KL loss: %.4f ' %
                         (epoch, args.num_epochs, batch_idx + 1, num_iter, total_loss.item(), (sampler_loss1 + sampler_loss2).item()))
        sys.stdout.flush()
    sys.stdout.write('\n')


if __name__ == '__main__':
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    log_name = 'log/%s/%s/%s_%d_%s_%s_%s.txt' % ('ours', args.dataset, args.loss, args.batch_size, str(args.noise_r), args.noise_type, time)
    log = open(log_name, 'w')
    print(' == write logger to ', log_name)
    log.flush()

    print('| Building net')
    network, feature_dim = load_network('resnet18')

    if args.dataset == 'cub':
        warm_up_epoch = 13
        args.num_classes = 200
    elif args.dataset == 'dog':
        warm_up_epoch = 5
        args.num_classes = 120
    elif args.dataset == 'aircraft':
        warm_up_epoch = 13
        args.num_classes = 100
    elif args.dataset == 'car':
        warm_up_epoch = 13
        args.num_classes = 196

    from src.contrastive_l import SNSCL
    my_model = SNSCL(network=network, backbone='resnet18', queue_size=32, projector_dim=args.projector_dim, feature_dim=feature_dim,
                     class_num=args.num_classes, momentum=0.999, pretrained=True, pretrained_path='').cuda()
    classifier = Classifier(feature_dim, args.num_classes).cuda()
    print("backbone params: {:.2f}M".format(sum(p.numel() for p in my_model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    optimizer = torch.optim.SGD(
        [{'params': my_model.parameters()}, {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio}],
        lr=args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
    sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    robust_loss = get_lossf(args)

    best_acc = 0.0
    loader = my_dataloder_helper(args.dataset, args)

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch < warm_up_epoch:  # warm up
            train_loader = loader.run('warmup')
            warmup(my_model, classifier, optimizer, train_loader)
        else:
            pred1 = (prob1 > args.p_threshold)  # divide dataset
            reliable_trainloader, unreliable_trainloader = loader.run('train', choice=pred1, confident=prob1)
            train(epoch, my_model, classifier, optimizer, reliable_trainloader, unreliable_trainloader)

        test_loader = loader.run('test')  # validation
        acc = test(epoch, my_model, classifier, test_loader)

        log.write('Test Epoch:%d    Acc:%.2f\n' % (epoch, acc))
        log.flush()

        eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch
        prob1, paths1 = eval_train(my_model, classifier, eval_loader)
        print('\n')

        sch_lr.step()
        if acc > best_acc:
            best_acc = acc

    print('best acc:%.2f\n' % (best_acc))
    log.write('Test Accuracy:%.2f\n' % (best_acc))
    log.flush()

