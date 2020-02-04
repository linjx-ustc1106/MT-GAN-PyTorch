
import time
from options import Options
from data.meta_dataloader import *
from models import create_model
from torch.utils.data import  DataLoader
import os
from glob import glob

if __name__ == '__main__':
    opt = Options().parse()  
    opt.isTrain = False
    root = opt.meta_dataroot
    dictTrainAs, dictTrainBs, dataset_num = meta_preprocess(root)
    for test_dataset_indx in range(dataset_num):
        dataset = MetaDataloader(root, k_shot=opt.k_spt,
                            k_query=opt.k_qry,
                            batchsz=2000, resize=opt.load_size, crop_size = opt.crop_size, 
                            test_dataset_indx= test_dataset_indx, dictTrainAs=dictTrainAs, dictTrainBs=dictTrainBs, dataset_num=dataset_num)  
        dataset_test = MetaTestDataloader(root, k_shot=opt.k_spt,
                            k_query=opt.k_qry,
                            batchsz=2, resize=opt.load_size, crop_size = opt.crop_size, 
                            test_dataset_indx= test_dataset_indx, dictTrainAs=dictTrainAs, dictTrainBs=dictTrainBs, dataset_num=dataset_num)
        dataset_size = len(dataset)    
        print('The number of training images = %d' % dataset_size)

        dataset_loader = DataLoader(dataset, opt.task_num, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
        dataset_loader_test = DataLoader(dataset_test, opt.task_num_val, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
        
        opt.epoch = test_dataset_indx 
        model = create_model(opt)      
        model.setup(opt)               

       
        for j, (A_spt, B_spt, A_qry, B_qry) in enumerate(dataset_loader_test): 
            model.set_input(A_spt, B_spt, A_qry, B_qry)
            
            model.finetunning(test_dataset_indx,opt.load_iter, j) 
            model.finetunning_withoutmeta(test_dataset_indx,opt.load_iter,j) 
            model.plot_training_loss(test_dataset_indx)
            
            
    