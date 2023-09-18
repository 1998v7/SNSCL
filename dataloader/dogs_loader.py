import json
import os

import numpy as np
import scipy.io
from os.path import join
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir
import random


class Dogs(VisionDataset):
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise=False, noise_r=0,
                 selected_index=None, confident=None, mode='test', args=None):
        super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.noise = noise
        self.noise_r = noise_r
        self.args = args
        self.mode = mode

        split, ann, real_label = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images
        if self.train and self.noise:
            noise_label = self.load_noise(real_label)
            self._flat_breed_images = [(ann[i] + '.jpg', noise_label[i]) for i in range(len(noise_label))]

        self.confident = confident
        if selected_index is not None:
            print(' == Selected partial data ...')
            self._flat_breed_images = self.select_by_index(self._flat_breed_images, selected_index)

    def select_by_index(self, samples, selected_index):
        selected = []
        for index in selected_index:
            img, target = samples[index]
            selected.append((img, target))
        return selected

    def load_noise(self, labels):
        noise_json = 'dataloader/noise_detail/' + self.args.dataset + '_' + self.args.noise_type + '_' + \
                     'noise_' + str(self.args.noise_r) + '.json'
        if os.path.exists(noise_json):
            if self.args.result_dir != 'dividemix':
                print(" == Load noise label file from %s ..." % noise_json)
            noise_label = json.load(open(noise_json, "r"))
        else:
            random.seed(42)
            dic = self.stats()
            noise_label = []
            if self.args.noise_type == 'sym':
                for label in range(120):
                    perClass_num = dic[label]
                    for i in range(perClass_num):
                        if i < int(perClass_num * self.noise_r):
                            noise_label.append(random.randint(0, 119))
                        else:
                            noise_label.append(label)
            elif self.args.noise_type == 'asym':
                for label in range(120):
                    perClass_num = dic[label]
                    for i in range(perClass_num):
                        if i < int(perClass_num * self.noise_r):
                            if label != 119:
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
            print(' == Real noise-ratio: ', 1 - sum((np.array(noise_label) == np.array(labels))) / len(labels))
        self.noise_or_not = np.transpose(labels) == np.transpose(noise_label)
        return noise_label

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = self.loader(image_path)

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

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels)), split, labels

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %d per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts


