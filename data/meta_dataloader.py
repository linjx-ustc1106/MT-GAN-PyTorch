import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
from glob import glob
def meta_preprocess(root):
    dictTrainAs = {}
    dictTrainBs = {}
    dataset_indx = 0
    datasets_dir = glob(os.path.join(root, '*'))
    
    print('All datasets are from: {}'.format(datasets_dir))
    for sub_dir in datasets_dir:
        if (os.path.isdir(sub_dir)):
            trainA_paths = glob(os.path.join(sub_dir, 'trainA/*'))
            np.random.shuffle(trainA_paths)
            for filename in trainA_paths:
                if dataset_indx in dictTrainAs.keys():
                    dictTrainAs[dataset_indx].append(filename)
                else:
                    dictTrainAs[dataset_indx] = [filename]
                    
            trainB_paths = glob(os.path.join(sub_dir, 'trainB/*'))
            np.random.shuffle(trainB_paths)
            for filename in trainB_paths:
                if dataset_indx in dictTrainBs.keys():
                    dictTrainBs[dataset_indx].append(filename)
                else:
                    dictTrainBs[dataset_indx] = [filename]
            dataset_indx = dataset_indx + 1
    dataset_num = dataset_indx
    print('Finished preprocessing the datasets...')
    print('Overall dataset number : {}'.format(dataset_num))
    return dictTrainAs, dictTrainBs, dataset_num
class MetaDataloader(Dataset):


    def __init__(self, root, batchsz, k_shot, k_query, resize, crop_size,  test_dataset_indx, dictTrainAs, dictTrainBs, dataset_num):

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz =  self.k_shot  # num of samples per set
        self.querysz = self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize 
        self.crop_size = crop_size  # crop size
        self.dataset_num = dataset_num
        print('shuffle b:%d, %d-shot, %d-query, resize:%d' % (
         batchsz, k_shot, k_query, resize))
        self.dictTrainAs, self.dictTrainBs = dictTrainAs, dictTrainBs

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize(self.resize),
                                             transforms.RandomCrop(self.crop_size),
                                             transforms.RandomHorizontalFlip(),
                                             # transforms.RandomRotation(5),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])

        
        dataset_num_list =list(range(self.dataset_num))
        dataset_num_list.pop(test_dataset_indx)
        print('Meta training dataset number : {}'.format(self.dataset_num))
        
        self.create_batch(self.batchsz, dataset_num_list, test_dataset_indx)

    
    
        
    def create_batch(self, batchsz, dataset_num_list, test_dataset_indx):
        """
        create batch for meta-learning.
        """
        self.support_A_batch = []  # support set batch
        self.support_B_batch = []  # support set batch
        self.query_A_batch = []  # query set batch
        self.query_B_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select one class randomly
            selected_cls = np.random.choice(dataset_num_list, 1, False)  
            support_A = []
            support_B = []
            query_A = []
            query_B = []
            selected_imgs_idxA = np.random.choice(len(self.dictTrainAs[selected_cls[0]]), self.k_shot + self.k_query, False)
            selected_imgs_idxB = np.random.choice(len(self.dictTrainBs[selected_cls[0]]), self.k_shot + self.k_query, False)
            np.random.shuffle(selected_imgs_idxA)
            np.random.shuffle(selected_imgs_idxB)
            
            
            indexDtrainA = np.array(selected_imgs_idxA[:self.k_shot])  # idx for Dtrain
            indexDtestA = np.array(selected_imgs_idxA[self.k_shot:])  # idx for Dtest
            indexDtrainB = np.array(selected_imgs_idxB[:self.k_shot])  # idx for Dtrain
            indexDtestB = np.array(selected_imgs_idxB[self.k_shot:])  # idx for Dtest
            
            support_A.append(
                np.array(self.dictTrainAs[selected_cls[0]])[indexDtrainA].tolist())  # get all images filename for current Dtrain
            query_A.append(np.array(self.dictTrainAs[selected_cls[0]])[indexDtestA].tolist())
            
            support_B.append(
                np.array(self.dictTrainBs[selected_cls[0]])[indexDtrainB].tolist())  # get all images filename for current Dtrain
            query_B.append(np.array(self.dictTrainBs[selected_cls[0]])[indexDtestB].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_A)
            random.shuffle(query_A)
            random.shuffle(support_B)
            random.shuffle(query_B)

            self.support_A_batch.append(support_A)  # append set to current sets
            self.query_A_batch.append(query_A)  # append sets to current sets\
            self.support_B_batch.append(support_B)  # append set to current sets
            self.query_B_batch.append(query_B)  # append sets to current sets
            
        print('Finished create batches of the datasets...')
        

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        """
        # [setsz, 3, resize, resize]
        support_A = torch.FloatTensor(self.setsz, 3, self.crop_size, self.crop_size)
        support_B = torch.FloatTensor(self.setsz, 3, self.crop_size, self.crop_size)
        # [setsz]
        # [querysz, 3, resize, resize]
        query_A = torch.FloatTensor(self.querysz, 3, self.crop_size, self.crop_size)
        query_B = torch.FloatTensor(self.querysz, 3, self.crop_size, self.crop_size)
        # [querysz]

        flatten_support_A = [item
                             for sublist in self.support_A_batch[index] for item in sublist]
        
        flatten_support_B = [item
                             for sublist in self.support_B_batch[index] for item in sublist]

        flatten_query_A = [item
                           for sublist in self.query_A_batch[index] for item in sublist]
        
        flatten_query_B = [item
                           for sublist in self.query_B_batch[index] for item in sublist]


        for i, path in enumerate(flatten_support_A):
            support_A[i] = self.transform(path)

        for i, path in enumerate(flatten_query_A):
            query_A[i] = self.transform(path)
            
        for i, path in enumerate(flatten_support_B):
            support_B[i] = self.transform(path)

        for i, path in enumerate(flatten_query_B):
            query_B[i] = self.transform(path)

        return support_A, support_B, query_A, query_B
    def __len__(self):
        
        return self.batchsz

class MetaTestDataloader(Dataset):

    def __init__(self, root, batchsz, k_shot, k_query, resize, crop_size,  test_dataset_indx, dictTrainAs, dictTrainBs, dataset_num):

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz =  self.k_shot  # num of samples per set
        self.querysz = self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.crop_size = crop_size  #
        self.dataset_num = dataset_num
        print('shuffle b:%d, %d-shot, %d-query, resize:%d' % (
         batchsz, k_shot, k_query, resize))
        self.dictTrainAs, self.dictTrainBs = dictTrainAs, dictTrainBs
        
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize((self.crop_size, self.crop_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])
        
        
        
        self.create_batch(self.batchsz, test_dataset_indx)

    
    
        
    def create_batch(self, batchsz, test_dataset_indx):
        """
        create batch for meta-learning.
        """
        self.support_A_batch = []  # support set batch
        self.support_B_batch = []  # support set batch
        self.query_A_batch = []  # query set batch
        self.query_B_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            selected_cls = test_dataset_indx
            support_A = []
            support_B = []
            query_A = []
            query_B = []
            selected_imgs_idxA = np.random.choice(len(self.dictTrainAs[selected_cls]), self.k_shot + self.k_query, False)
            selected_imgs_idxB = np.random.choice(len(self.dictTrainBs[selected_cls]), self.k_shot + self.k_query, False)
            np.random.shuffle(selected_imgs_idxA)
            np.random.shuffle(selected_imgs_idxB)
            
            
            indexDtrainA = np.array(selected_imgs_idxA[:self.k_shot])  # idx for Dtrain
            indexDtestA = np.array(selected_imgs_idxA[self.k_shot:])  # idx for Dtest
            indexDtrainB = np.array(selected_imgs_idxB[:self.k_shot])  # idx for Dtrain
            indexDtestB = np.array(selected_imgs_idxB[self.k_shot:])  # idx for Dtest
            
            support_A.append(
                np.array(self.dictTrainAs[selected_cls])[indexDtrainA].tolist())  # get all images filename for current Dtrain
            query_A.append(np.array(self.dictTrainAs[selected_cls])[indexDtestA].tolist())
            
            support_B.append(
                np.array(self.dictTrainBs[selected_cls])[indexDtrainB].tolist())  # get all images filename for current Dtrain
            query_B.append(np.array(self.dictTrainBs[selected_cls])[indexDtestB].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_A)
            random.shuffle(query_A)
            random.shuffle(support_B)
            random.shuffle(query_B)

            self.support_A_batch.append(support_A)  # append set to current sets
            self.query_A_batch.append(query_A)  # append sets to current sets\
            self.support_B_batch.append(support_B)  # append set to current sets
            self.query_B_batch.append(query_B)  # append sets to current sets
            
        print('Finished create batches of the datasets...')
        

    def __getitem__(self, index):

        # [setsz, 3, resize, resize]
        support_A = torch.FloatTensor(self.setsz, 3, self.crop_size, self.crop_size)
        support_B = torch.FloatTensor(self.setsz, 3, self.crop_size, self.crop_size)
        # [setsz]
        # [querysz, 3, resize, resize]
        query_A = torch.FloatTensor(self.querysz, 3, self.crop_size, self.crop_size)
        query_B = torch.FloatTensor(self.querysz, 3, self.crop_size, self.crop_size)
        # [querysz]

        flatten_support_A = [item
                             for sublist in self.support_A_batch[index] for item in sublist]
        
        flatten_support_B = [item
                             for sublist in self.support_B_batch[index] for item in sublist]

        flatten_query_A = [item
                           for sublist in self.query_A_batch[index] for item in sublist]
        
        flatten_query_B = [item
                           for sublist in self.query_B_batch[index] for item in sublist]

        for i, path in enumerate(flatten_support_A):
            support_A[i] = self.transform(path)

        for i, path in enumerate(flatten_query_A):
            query_A[i] = self.transform(path)
            
        for i, path in enumerate(flatten_support_B):
            support_B[i] = self.transform(path)

        for i, path in enumerate(flatten_query_B):
            query_B[i] = self.transform(path)

        return support_A, support_B, query_A, query_B
    def __len__(self):
        
        return self.batchsz


