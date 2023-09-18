import os
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import random
import json
import numpy as np


class Cars(VisionDataset):
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise=False, noise_r=0,
                 selected_index=None, confident=None, mode='test', args=None):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        self.noise = noise
        self.noise_r = noise_r
        self.args = args
        self.mode = mode
        self.confident = confident

        loaded_mat = sio.loadmat(os.path.join(self.root, 'cars_annos.mat'))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []

        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

        if self.train:
            self.samples = self.load_noise(self.samples)
        else:
            self.samples = self.samples

        if selected_index is not None:
            print(' == Selected partial data ...')
            self.samples = self.select_by_index(self.samples, selected_index)

    def select_by_index(self, samples, selected_index):
        selected = []
        for index in selected_index:
            img, target = samples[index]
            selected.append((img, target))
        return selected

    def load_noise(self, samples):
        images, targets = [], []
        for i in range(len(samples)):
            img, tar = samples[i]
            images.append(img)
            targets.append(tar)

        per_cls_num = {}
        for class_i in range(196):
            per_cls_num[class_i] = targets.count(class_i)

        noise_json = 'dataloader/noise_detail/' + self.args.dataset + '_' + self.args.noise_type + '_' + \
                     'noise_' + str(self.args.noise_r) + '.json'

        if os.path.exists(noise_json):
            if self.args.result_dir != 'dividemix':
                print(" == Load noise label file from %s ..." % noise_json)
            noise_label = json.load(open(noise_json, "r"))
        else:
            random.seed(42)
            noise_label = []
            if self.args.noise_type == 'sym':
                for label in range(196):
                    perClass_num = per_cls_num[label]
                    noise_num = int(perClass_num * self.noise_r)
                    for i in range(perClass_num):
                        if i < noise_num:
                            noise_label.append(random.randint(0, 195))
                        else:
                            noise_label.append(label)
            elif self.args.noise_type == 'asym':
                for label in range(196):
                    perClass_num = per_cls_num[label]
                    noise_num = int(perClass_num * self.noise_r)
                    for i in range(perClass_num):
                        if i < noise_num:
                            if label != 195:
                                noise_label.append(label + 1)
                            else:
                                noise_label.append(0)
                        else:
                            noise_label.append(label)
            else:
                print(' == Noise type Error! ')

            print(" == Save labels to %s ..." % noise_json)
            json.dump(noise_label, open(noise_json, "w"))
        if self.args.result_dir != 'dividemix':
            print('noise-ratio: ', 1 - sum(np.array(noise_label) == np.array(targets)) / len(targets))
        self.noise_or_not = np.transpose(targets) == np.transpose(noise_label)

        samples = []
        for i in range(len(images)):
            samples.append((images[i], noise_label[i]))
        return samples


    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)

        if self.args.result_dir == 'dividemix':
            if self.mode == 'label':
                image1 = self.transform(image)
                image2 = self.transform(image)
                conf = self.confident[index]
                return image1, image2, target, conf
            elif self.mode == 'unlabel':
                image1 = self.transform(image)
                image2 = self.transform(image)
                conf = self.confident[index]
                return image1, image2, target, conf
            else:
                image1 = self.transform(image)
                return image1, target, index
        else:
            image = self.transform(image)
            return image, target, index

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    train_dataset = Cars('../dataset/car', train=True, download=False)
    test_dataset = Cars('../dataset/car', train=False, download=False)