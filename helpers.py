import torch
import os
import datetime
from torch.nn import CrossEntropyLoss
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch import nn


def set_env(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.dataset == 'aircraft':
        args.num_classes = 100
        args.num_class = 100
    elif args.dataset == 'dog':
        args.num_classes = 120
        args.num_class = 120
    elif args.dataset == 'cub':
        args.num_classes = 200
        args.num_class = 200
    elif args.dataset == 'car':
        args.num_class = 196
        args.num_classes = 196
    else:
        ValueError('dataset error in set_env! not in [aircraft, dog, car, cub]')


def pprint(context, log_name):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(time + ' | ' + context)
    log_name.write(time + ' | ' + context+'\n')
    log_name.flush()


def get_dataloader(args):
    transform_test = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    transform_train = T.Compose([
        T.Resize((int(args.img_size * 1.2), int(args.img_size * 1.2))),
        T.RandomHorizontalFlip(),
        T.RandomCrop((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    if args.dataset == 'aircraft':
        from dataloader.aircraft_loader import Aircraft
        data_path = 'dataset'
        train_dataset = Aircraft(data_path, transform=transform_train, train=True,  noise=args.noise, noise_r=args.noise_r, args=args)
        test_dataset = Aircraft(data_path, transform=transform_test, train=False,  args=args)
    elif args.dataset == 'dog':
        from dataloader.dogs_loader import Dogs
        data_path = '/home/weiqi/noise_FGVC/dataset/stf_dog'
        train_dataset = Dogs(data_path, transform=transform_train, train=True, noise=args.noise, noise_r=args.noise_r, args=args)
        test_dataset = Dogs(data_path, transform=transform_test, train=False, args=args)

    elif args.dataset == 'cub':
        from dataloader.cub_loader import Cub2011
        data_path = 'dataset/cub2011'
        train_dataset = Cub2011(data_path, transform=transform_train, train=True, noise=args.noise, noise_r=args.noise_r, args=args)
        test_dataset = Cub2011(data_path, transform=transform_test, train=False, args=args)
    else:
        ValueError('dataset error in get_dataloader! not in [aircraft, dog, cub]')

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader


def get_optimizer(args, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    return optimizer, sch_lr


def get_lossf(args):
    if args.loss == 'ce_loss':
        class CrossEntropy(torch.nn.Module):
            def __init__(self, reduction='mean'):
                super(CrossEntropy, self).__init__()
                self.reduction = reduction

            def forward(self, input, labels):
                logsoftmax = torch.nn.LogSoftmax(dim=1)
                res = - labels * logsoftmax(input)
                if self.reduction == 'mean':
                    return torch.mean(torch.sum(res, dim=1))
                elif self.reduction == 'sum':
                    return torch.sum(torch.sum(res, dim=1))
                else:
                    return res

        loss_function = CrossEntropy()
        print(' == use ce loss')

    elif args.loss == 'APL':
        from APL_loss import NFLandRCE
        loss_function = NFLandRCE(alpha=1, beta=1, num_classes=args.num_classes)
        print(' == use APL loss')

    elif args.loss == 'Asym':
        from Asym_loss import NCEandAGCE
        loss_function = NCEandAGCE()
        print(' == use Asym loss')

    elif args.loss == 'GCE':
        from APL_loss import GeneralizedCrossEntropy
        loss_function = GeneralizedCrossEntropy(num_classes=args.num_classes)
        print(' == use GCE loss')

    elif args.loss == 'Sym':
        from APL_loss import SCELoss
        loss_function = SCELoss(alpha=0.1, beta=1, num_classes=args.num_classes)
        print(' == use Sym loss')

    elif args.loss == 'label_smooth':
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, classes, smoothing=0.0, dim=-1):
                super(LabelSmoothingLoss, self).__init__()
                self.confidence = 1.0 - smoothing
                self.smoothing = smoothing
                self.cls = classes
                self.dim = dim

            def forward(self, pred, target):
                assert 0 <= self.smoothing < 1
                # compute hard label
                with torch.no_grad():
                    _, hard_target = torch.max(target, dim=-1)

                pred = pred.log_softmax(dim=self.dim)

                with torch.no_grad():
                    true_dist = torch.zeros_like(pred)
                    true_dist.fill_(self.smoothing / (self.cls - 1))
                    true_dist.scatter_(1, hard_target.data.unsqueeze(1), self.confidence)
                return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

        loss_function = LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1)
        print(' == use label smooth loss')


    elif args.loss == 'confPenalty':
        class NegEntropy(nn.Module):
            def __init__(self):
                super(NegEntropy, self).__init__()
                self.logsoftmax = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, labels):
                res = - labels * self.logsoftmax(input)
                ce_loss = torch.mean(torch.sum(res, dim=1))
                probs = torch.softmax(input, dim=1)
                return ce_loss + torch.mean(torch.sum(probs.log() * probs, dim=1))

        loss_function = NegEntropy()
        print(' == use NegEntropy loss')
    else:
        ValueError('Error args.loss name in get_lossf')
    return loss_function

        

def get_logger(args):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    fold = 'log/' + args.main_name + '/' + args.dataset + '/'
    if not os.path.exists(fold):
        os.makedirs(fold)

    name = fold + args.loss + str(args.bs) + '_' + time + '_' + str(args.noise) + '_' + str(args.noise_r) + '.txt'
    logger = open(name, 'w')
    pprint(str(args), logger)
    return logger, name
