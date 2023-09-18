from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from dataloader.dogs_loader import Dogs as Mydata
from torchvision import transforms as T
import torch.multiprocessing
from dataloader.randaugment import *
torch.multiprocessing.set_sharing_strategy('file_system')


class my_dataloder_helper():
    def __init__(self, dataset, args):
        self.noise_r = args.noise_r
        self.dataset = dataset
        self.args = args
        self.bs = args.batch_size

        if self.dataset == 'dog':
            self.data_path = 'dataset/stf_dog'
        elif self.dataset == 'aircraft':
            self.data_path = 'dataset/fgvc-aircraft-2013b'
        elif self.args.dataset == 'cub':
            self.data_path = 'dataset/cub2011'
        elif self.args.dataset == 'car':
            self.data_path = 'dataset/car'
        else:
            ValueError('error in Mydatalaoder helper ...')

        self.transform_train = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        #
        self.trainform_test = T.Compose([
            transforms.Resize((int(args.img_size * 1.2), int(args.img_size * 1.2))),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def run(self, mode, choice=None, confident=None):
        if self.args.dataset == 'dog':
            from dataloader.dogs_loader import Dogs as Mydata
        elif self.args.dataset == 'aircraft':
            from dataloader.aircraft_loader import Aircraft as Mydata
        elif self.args.dataset == 'cub':
            from dataloader.cub_loader import Cub2011 as Mydata
        elif self.args.dataset == 'car':
            from dataloader.car_loader import Cars as Mydata
        else:
            ValueError('error in select_help.py')

        if mode == 'warmup':
            all_dataset = Mydata(self.data_path, transform=self.transform_train, train=True, noise=self.args.noise, noise_r=self.args.noise_r, args=self.args)
            warmup_loader = DataLoader(dataset=all_dataset, batch_size=128, shuffle=True, num_workers=4)
            return warmup_loader

        elif mode == 'train':
            pred_idx = choice.nonzero()[0]
            labeled_dataset = Mydata(self.data_path, transform=self.transform_train, train=True, noise=self.args.noise, noise_r=self.args.noise_r, args=self.args, selected_index=pred_idx, confident=confident, mode='label')
            labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=self.bs, shuffle=True, num_workers=4, drop_last=True)

            pred_idx = (1 - choice).nonzero()[0]
            unlabeled_dataset = Mydata(self.data_path, transform=self.transform_train, train=True, noise=self.args.noise,  noise_r=self.args.noise_r, args=self.args, selected_index=pred_idx, confident=confident, mode='unlabel')
            unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=self.bs, shuffle=True, num_workers=4, drop_last=True)
            return labeled_loader, unlabeled_loader

        elif mode == 'test':
            test_dataset = Mydata(self.data_path, transform=self.trainform_test, train=False, noise=self.args.noise, noise_r=self.args.noise_r, args=self.args)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.bs, shuffle=False, num_workers=4)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = Mydata(self.data_path, transform=self.transform_train, train=True, noise=self.args.noise, noise_r=self.args.noise_r, args=self.args)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=self.bs, shuffle=False, num_workers=4)
            return eval_loader