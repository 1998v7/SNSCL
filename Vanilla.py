import sys
import argparse
from helpers import *
import warnings
from models.backbone import *
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--main_name', type=str, default='vanilla')
parser.add_argument('--result_dir', default='', type=str)
parser.add_argument('--dataset', default='aircraft', type=str, choices=['aircraft', 'dog', 'cub', 'car'])
parser.add_argument('--loss', type=str, default='ce_loss', choices=['ce_loss', 'APL', 'Asym', 'GCE', 'Sym', 'label_smooth', 'confPenalty'])

parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--noise', default=True, type=bool)
parser.add_argument('--noise_r', default=0.0, type=float)
parser.add_argument('--pretrain', default=True, type=bool)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--num_worker', default=16, type=int)
parser.add_argument('--gpu', default='3', type=str)

parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--bs', default=64, type=int)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--lr_ratio', default=10, type=float)

# contrastive learing
parser.add_argument('--projector_dim', default=1024, type=int)
args = parser.parse_args()

set_env(args)
logger, logger_name = get_logger(args)
train_dataset, test_dataset, train_loader, test_loader = get_dataloader(args)
backbone = get_model(args)
classifier = get_cls_layer(args)
loss_function = get_lossf(args)

optimizer = torch.optim.SGD([{'params': backbone.parameters()},
                             {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio}],
                            lr=args.lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40], gamma=0.1)

print(' == writer the logger to ', logger_name)
best_acc = 0.0

for epoch in range(args.num_epoch):
    backbone.train()
    classifier.train()

    train_loss = 0
    train_correct = 0

    for index, (inputs, labels, _) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        feature = backbone(inputs)
        output = classifier(feature)
        loss = loss_function(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch:[%2d/%2d]\t iter:[%4d/%4d]\t loss: %.3f\t lr: %.6f' % (epoch + 1, args.num_epoch, index + 1, len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))
        sys.stdout.flush()

        _, pred = output.max(1)
        train_correct += pred.eq(labels).sum().item()
        train_loss += loss.item()

    sys.stdout.write('\n')
    train_correct = 100. * train_correct / len(train_loader.dataset)
    train_loss = train_loss / len(train_loader)
    pprint('Epoch: [%2d/%2d]\t  |  train_acc:%.4f\t  train_loss:%.4f\t' % (epoch + 1, args.num_epoch, train_correct, train_loss), logger)

    backbone.eval()
    classifier.eval()
    correct = 0
    for batch_id, (inputs, labels, _) in enumerate(test_loader):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            feature = backbone(inputs)
            output = classifier(feature)
            _, pred = output.max(1)
            correct += pred.eq(labels).sum().item()
    val_acc = 100. * correct / len(test_loader.dataset)

    sch_lr.step()
    pprint('Epoch: [%2d/%2d]\t  |  val_acc:%.4f\t \n' % (epoch + 1, args.num_epoch, val_acc), logger)
    if best_acc < val_acc:
        best_acc = val_acc

logger.close()
print('best acc\t', best_acc)