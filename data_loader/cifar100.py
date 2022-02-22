import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random 
import os
import json
from numpy.testing import assert_array_almost_equal



def get_cifar100(root, cfg_trainer, train=True,
                transform_train=None, transform_train_aug=None, transform_val=None,
                download=False, noise_file = ''):
    base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)
    if train:
        train_idxs, val_idxs = train_val_split(base_dataset.targets)
        train_dataset = CIFAR100_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, transform_aug=transform_train_aug)
        val_dataset = CIFAR100_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        elif cfg_trainer['instance']:
            train_dataset.instance_noise()
            val_dataset.instance_noise()
        elif cfg_trainer['real'] == 'noisy_label':
            new_targets = cifar100n(root, 'noisy_label', download)
            train_dataset.train_labels = np.array(new_targets)[train_dataset.indexs]
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()
        
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        train_dataset = []
        val_dataset = CIFAR100_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    
    
    
    return train_dataset, val_dataset

def download_cifarn(root):
        wget.download('http://128.114.59.66:5000/files/CIFAR-N.zip', out=root)
        with ZipFile(os.path.join(root, 'CIFAR-N.zip'), 'r') as f:
            f.extractall(root)

def cifar100n(root, key, download = True):
    label_path = os.path.join(root, 'CIFAR-N', 'CIFAR-100_human.pt')
    if not os.path.exists(label_path):
        if download:
            download_cifarn(root)
        else:
            raise FileNotFoundError(f'Labels not found. Please set download=True. Path: {data_path}')
    noise_label = torch.load(label_path)
    targets_new = torch.from_numpy(noise_label[key])
    return targets_new

def train_val_split(base_dataset: torchvision.datasets.CIFAR100):
    num_classes = 100
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, transform_aug = None, 
                 target_transform=None,
                 download=False):
        super(CIFAR100_train, self).__init__(root, train=train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]
        self.train_labels = np.array(self.targets)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.transform_aug = transform_aug
        #self.all_refs_encoded = torch.zeros(self.num_classes,self.num_ref,1024, dtype=np.float32)

        self.count = 0

        self.train_labels_gt = self.train_labels.copy()

    def symmetric_noise(self):
        
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y



    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert(noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy

    def instance_noise(self):

        noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C100.pt'.format(self.cfg_trainer['percent']))

        noisylabel = noise_label['noise_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)
            
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index],  self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform_aug is not None:
            img2 = self.transform_aug(img)


        if self.transform is not None:
            img = self.transform(img)

        

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img2, target, index, target_gt

    def __len__(self):
        return len(self.train_data)


class CIFAR100_val(torchvision.datasets.CIFAR100):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)
        self.train_labels_gt = self.train_labels.copy()
        self.indexs = indexs
    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert(noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy

    def instance_noise(self):

        noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C100.pt'.format(self.cfg_trainer['percent']))

        noisylabel = noise_label['noise_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)


    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt


    