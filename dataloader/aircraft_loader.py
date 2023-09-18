import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import random
import json


class Aircraft(VisionDataset):
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join( 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None, target_transform=None, download=False, noise=False, noise_r=0.0, args=None,
                 selected_index=None, confident=None, mode='test'):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(split, ', '.join(self.splits)))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(class_type, ', '.join(self.class_types)))

        self.loader = default_loader
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data', 'images_%s_%s.txt' % (self.class_type, self.split))
        self.noise = noise
        self.noise_r = noise_r
        self.args = args

        # for divideMix
        self.mode = mode
        self.confident = confident

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)
        self.classes = classes
        self.class_to_idx = class_to_idx

        if train:
            self.samples = self.load_noise(samples)
        else:
            self.samples = samples

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
        for class_i in range(100):
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
                for label in range(100):
                    perClass_num = per_cls_num[label]
                    noise_num = int(perClass_num * self.noise_r)
                    for i in range(perClass_num):
                        if i < noise_num:
                            noise_label.append(random.randint(0, 99))
                        else:
                            noise_label.append(label)

            elif self.args.noise_type == 'asym':
                for label in range(100):
                    perClass_num = per_cls_num[label]
                    noise_num = int(perClass_num * self.noise_r)
                    for i in range(perClass_num):
                        if i < noise_num:
                            if label != 99:
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

    def make_dataset(self, image_ids, targets):
        random.seed(42)
        assert (len(image_ids) == len(targets))
        images = []
        per_cls_num = {}
        for i in range(100):
            per_cls_num[i] = targets.count(i)

        tup = []
        for i in range(len(image_ids)):
            tup.append((image_ids[i], targets[i]))
        tup.sort(key=lambda x: x[1])

        image_ids, targets = [], []
        for (iid, target) in tup:
            image_ids.append(iid)
            targets.append(target)

        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder, '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


    def __getitem__(self, index):
        path, target = self.samples[index]
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

    def find_classes(self):
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))
        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]
        return image_ids, targets, classes, class_to_idx


if __name__ == '__main__':
    train_dataset = Aircraft('', train=True, download=False, noise=True, noise_r=0.2)
    test_dataset = Aircraft('', train=False, download=False)