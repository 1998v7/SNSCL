import os
import numpy as np
import pandas as pd
import random
import json
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive


class Cub2011(VisionDataset):
    base_folder = 'CUB_200_2011/images'
    filename = 'CUB_200_2011.tgz'
    def __init__(self, data_path, train=True, transform=None, target_transform=None, noise=True, noise_r=0.0, args=None,
                 selected_index=None, confident=None, mode='test'):
        super(Cub2011, self).__init__(root=data_path, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.noise = noise
        self.noise_r = noise_r
        self.args = args

        # special in DivideMix
        self.confident = confident
        self.mode = mode

        self._load_metadata()
        if selected_index is not None:
            print(' == Selected partial data ...')
            self.image_path, self.labels = self.select_by_index(self.image_path, self.labels, selected_index)

    def select_by_index(self, image_path, label, selected_index):
        sel_img, sel_lab = [], []
        for index in selected_index:
            img, target = image_path[index], label[index]
            sel_img.append(img)
            sel_lab.append(target)
        return sel_img, sel_lab

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'), sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            self.image_path, self.labels = self.load_data(self.data)
            if self.noise:
                self.image_path, self.labels = self.build_noise(self.image_path, self.labels)
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.image_path, self.labels = self.load_data(self.data)

    def load_data(self, data):
        image_path, targets = [], []
        for i in range(len(data)):
            sample = data.iloc[i]
            path = os.path.join(self.root, self.base_folder, sample.filepath)

            target = sample.target - 1
            image_path.append(path)
            targets.append(target)
        return image_path, targets
    
    def build_noise(self, image_path, targets):
        per_cls_num = {}
        for class_i in range(200):
            per_cls_num[class_i] = targets.count(class_i)

        noise_json = 'dataloader/noise_detail/' + self.args.dataset + '_' + self.args.noise_type + '_' + \
                     'noise_' + str(self.args.noise_r) + '.json'
        if os.path.exists(noise_json):
            if self.args.result_dir != 'dividemix':
                print(" == Load noise label file from %s ..." % noise_json)
            noise_label = json.load(open(noise_json, "r"))
        else:
            noise_label = []
            random.seed(42)
            if self.args.noise_type == 'sym':
                for label in range(200):
                    perClass_num = per_cls_num[label]
                    noise_num = int(perClass_num * self.noise_r)
                    for i in range(perClass_num):
                        if i < noise_num:
                            noise_label.append(random.randint(0, 199))
                        else:
                            noise_label.append(label)
            elif self.args.noise_type == 'asym':
                for label in range(200):
                    perClass_num = per_cls_num[label]
                    noise_num = int(perClass_num * self.noise_r)
                    for i in range(perClass_num):
                        if i < noise_num:
                            if label != 199:
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
            print(" == Load noise label file from %s ..." % noise_json)

        self.noise_or_not = np.transpose(targets) == np.transpose(noise_label)  # for co-teaching and JOCOR
        return image_path, noise_label


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        target = self.labels[idx]
        image = self.loader(img_path)

        if self.args.result_dir == 'dividemix':
            if self.mode == 'label':
                image1 = self.transform(image)
                image2 = self.transform(image)
                conf = self.confident[idx]
                return image1, image2, target, conf
            elif self.mode == 'unlabel':
                image1 = self.transform(image)
                image2 = self.transform(image)
                conf = self.confident[idx]
                return image1, image2, target, conf
            else:
                image1 = self.transform(image)
                return image1, target, idx
        else:
            image = self.transform(image)
            return image, target, idx


if __name__ == '__main__':
    train_dataset = Cub2011('../dataset/cub2011', train=True, download=False)
    test_dataset = Cub2011('../dataset/cub2011', train=False, download=False)
    # print(train_dataset.labels)