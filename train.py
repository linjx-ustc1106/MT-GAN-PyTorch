
import time
from options import Options
from data.meta_dataloader import *
from models import create_model
from torch.utils.data import  DataLoader
import os
from glob import glob
if __name__ == '__main__':
    opt = Options().parse()   
    
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
        
        print('The number of training batches = %d' % dataset_size)
        dataset_loader = DataLoader(dataset, opt.task_num, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
        dataset_loader_test = DataLoader(dataset_test, opt.task_num_val, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
        
        model = create_model(opt)      
        model.setup(opt)               
        total_iters = 0                
        
        # CycleGAN training for comparison
        for j, (A_spt, B_spt, A_qry, B_qry) in enumerate(dataset_loader_test): 
            model.set_input(A_spt, B_spt, A_qry, B_qry) 
            model.finetunning_withoutmeta(test_dataset_indx,0,j) 
        # MT-GAN training 
        for i, (A_spt, B_spt, A_qry, B_qry) in enumerate(dataset_loader): 

            total_iters += opt.batch_size
            model.set_input(A_spt, B_spt, A_qry, B_qry)        
            model.meta_train(test_dataset_indx, total_iters)   
            print('========================================================')
            if total_iters % 100 == 0:   
                for j, (A_spt, B_spt, A_qry, B_qry) in enumerate(dataset_loader_test): 
                    model.set_input(A_spt, B_spt, A_qry, B_qry)         
                    model.finetunning(test_dataset_indx, total_iters, j) 
            if total_iters % opt.save_latest_freq == 0:   
                    print('saving the latest model (dataset %d, total_iters %d)' % (test_dataset_indx, total_iters))
                    save_suffix = 'dataset_%d_iter_%d' % (test_dataset_indx, total_iters)
                    model.save_networks(save_suffix)
           