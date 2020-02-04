import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
from .learner import generator, discriminator, generator_st, discriminator_st
from torchvision.utils import save_image
import os
import numpy as np
from  torch.nn import functional as F
from logger import Logger
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
class MtGANModel(BaseModel):
    '''
    Our model is based on CycleGAN's network architecture
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the MT-GAN class.

        """
        BaseModel.__init__(self, opt)
        
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        
        self.netG_A = generator().to(self.device)
        self.netG_B = generator().to(self.device)

        self.netD_A = discriminator().to(self.device)
        self.netD_B = discriminator().to(self.device)

        
        if opt.lambda_identity > 0.0:  
            assert(opt.input_nc == opt.output_nc)
        self.fake_A_pool = ImagePool(opt.pool_size)  
        self.fake_B_pool = ImagePool(opt.pool_size)  
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.meta_lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.meta_lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.lr = opt.lr
        self.beta1 = opt.beta1
        if self.isTrain:
            self.experiment_name = opt.name
            train_save_path = os.path.join('./checkpoints',self.experiment_name, 'images/')
            if not os.path.exists(train_save_path): 
                os.makedirs(train_save_path)
        else:
            self.experiment_name = os.path.join(opt.name,'test')
            
            test_save_path = os.path.join('./checkpoints',self.experiment_name, 'images/')
            if not os.path.exists(test_save_path): 
                os.makedirs(test_save_path)
        self.log_dir = os.path.join('./checkpoints',self.experiment_name, 'logs')
        if not os.path.exists(self.log_dir): 
            os.makedirs(self.log_dir)
        self.logger = Logger(self.log_dir)
    def set_input(self, real_A_support, real_B_support, real_A_query, real_B_query):
        if random.choice([True, False]):
            self.real_A_support = real_A_support.to(self.device)
            self.real_B_support = real_B_support.to(self.device)
            self.real_A_query = real_A_query.to(self.device)
            self.real_B_query = real_B_query.to(self.device)
        else:
            self.real_A_support = real_B_support.to(self.device)
            self.real_B_support = real_A_support.to(self.device)
            self.real_A_query = real_B_query.to(self.device)
            self.real_B_query = real_A_query.to(self.device)
    def D_losses(self, pred_real_B, pred_fake_B, pred_real_A, pred_fake_A):
        
        loss_D_A =  (self.criterionGAN(pred_fake_B, False) + self.criterionGAN(pred_real_B, True))*0.5
        loss_D_B =  (self.criterionGAN(pred_fake_A, False) + self.criterionGAN(pred_real_A, True))*0.5
        return loss_D_A, loss_D_B
        
        
    def G_losses(self, real_A, real_B, idt_A, idt_B, gan_A, gan_B, rec_A, rec_B):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if lambda_idt > 0:
            
            self.loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
            self.loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

      
        self.loss_G_A = self.criterionGAN(gan_A, True)
        
        self.loss_G_B = self.criterionGAN(gan_B, True)
        
        self.loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
        
        self.loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B
        
        loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        return loss_G
    def backward_D_basic(self, netD, real, fake):

     
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
       
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
      
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D    
    def forward_G(self, real_A, real_B, vars_GA, vars_GB, vars_DA, vars_DB):
        fake_B = self.netG_A(real_A, vars=vars_GA, bn_training=True)
        rec_A = self.netG_B(fake_B, vars=vars_GB, bn_training=True)  
        fake_A = self.netG_B(real_B, vars=vars_GB, bn_training=True)
        rec_B = self.netG_A(fake_A, vars=vars_GA, bn_training=True) 
        idt_A = self.netG_A(real_B, vars=vars_GA, bn_training=True)
        idt_B = self.netG_B(real_A, vars=vars_GB, bn_training=True)
        gan_A = self.netD_A(fake_B, vars=vars_DA, bn_training=True)
        gan_B = self.netD_B(fake_A, vars=vars_DB, bn_training=True)
        return fake_A, fake_B, idt_A, idt_B, gan_A, gan_B, rec_A, rec_B
        
    def forward_D(self, real_A, real_B, fake_A, fake_B, vars_DA, vars_DB):    
        fake_pool_B = self.fake_B_pool.query(fake_B)
        fake_pool_A = self.fake_A_pool.query(fake_A)
        pred_fake_B = self.netD_A(fake_pool_B.detach(), vars=vars_DA, bn_training=True)
        pred_fake_A = self.netD_B(fake_pool_A.detach(), vars=vars_DB, bn_training=True)
        pred_real_B = self.netD_A(real_B, vars=vars_DA, bn_training=True)
        pred_real_A = self.netD_B(real_A, vars=vars_DB, bn_training=True)
        return pred_real_B, pred_fake_B, pred_real_A, pred_fake_A
    def denorm(self, x):
        
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    def meta_train(self, test_dataset_indx, total_iters):
        """MT-GAN training process"""
        task_num, setsz, c_, h, w = self.real_A_support.size()
        querysz = self.real_A_query.size(1)

        lossD_A_q = 0
        lossD_B_q = 0
        lossG_q = 0
        self.netG_B.load_state_dict(self.netG_A.state_dict()) # network weights copy operation
            
        self.netD_B.load_state_dict(self.netD_A.state_dict()) # network weights copy operation
        for i in range(task_num):
            # Initialization
            self.real_A = self.real_A_support[i]
            self.real_B = self.real_B_support[i]
            self.real_A_q = self.real_A_query[i]
            self.real_B_q = self.real_B_query[i]
            self.fake_A, self.fake_B, self.idt_A, self.idt_B, self.gan_A, self.gan_B, self.rec_A, self.rec_B = self.forward_G(self.real_A, self.real_B, None, None, None, None)
            self.loss_G = self.G_losses(self.real_A, self.real_B, self.idt_A, self.idt_B, self.gan_A, self.gan_B, self.rec_A, self.rec_B)
            
            grad_A = torch.autograd.grad(self.loss_G, self.netG_A.parameters(),retain_graph=True)
            fast_weights_A = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_A, self.netG_A.parameters())))
            grad_B = torch.autograd.grad(self.loss_G, self.netG_B.parameters())
            fast_weights_B = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_B, self.netG_B.parameters())))

            self.pred_real_B, self.pred_fake_B, self.pred_real_A, self.pred_fake_A = self.forward_D(self.real_A, self.real_B, self.fake_A, self.fake_B, None, None)
            self.loss_D_A, self.loss_D_B = self.D_losses(self.pred_real_B, self.pred_fake_B, self.pred_real_A, self.pred_fake_A)
            grad_D_A = torch.autograd.grad(self.loss_D_A, self.netD_A.parameters())
            fast_weights_D_A = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_D_A, self.netD_A.parameters())))
            grad_D_B = torch.autograd.grad(self.loss_D_B, self.netD_B.parameters())
            fast_weights_D_B = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_D_B, self.netD_B.parameters())))
            # Meta-traning over update_steps
            for k in range(1, self.update_step):

                # 1. run the i-th task and compute loss for k=1~update_step-1

                self.fake_A, self.fake_B, self.idt_A, self.idt_B, self.gan_A, self.gan_B, self.rec_A, self.rec_B = self.forward_G(self.real_A, self.real_B, fast_weights_A, fast_weights_B, fast_weights_D_A, fast_weights_D_B)
                self.loss_G = self.G_losses(self.real_A, self.real_B, self.idt_A, self.idt_B, self.gan_A, self.gan_B, self.rec_A, self.rec_B)
                grad_A = torch.autograd.grad(self.loss_G, fast_weights_A,retain_graph=True)
                fast_weights_A = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_A, fast_weights_A)))
                grad_B = torch.autograd.grad(self.loss_G, fast_weights_B)
                fast_weights_B = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_B, fast_weights_B)))
                self.pred_real_B, self.pred_fake_B, self.pred_real_A, self.pred_fake_A = self.forward_D(self.real_A, self.real_B, self.fake_A, self.fake_B, fast_weights_D_A, fast_weights_D_B)
                self.loss_D_A, self.loss_D_B = self.D_losses(self.pred_real_B, self.pred_fake_B, self.pred_real_A, self.pred_fake_A)
                grad_D_A = torch.autograd.grad(self.loss_D_A, fast_weights_D_A)
                fast_weights_D_A = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_D_A, fast_weights_D_A)))
                grad_D_B = torch.autograd.grad(self.loss_D_B, fast_weights_D_B)
                fast_weights_D_B = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_D_B, fast_weights_D_B)))
                
                loss = {}
                loss['DA/loss'] = self.loss_D_A.item()
                loss['DB/loss'] = self.loss_D_B.item()
                loss['GA/loss_G_A'] = self.loss_G_A.item()
                loss['GA/loss_cycle_A'] = self.loss_cycle_A.item()
                loss['G/loss_idt_A'] = self.loss_idt_A.item()
                loss['GA/loss_G_B'] = self.loss_G_B.item()
                loss['GA/loss_cycle_B'] = self.loss_cycle_B.item()
                loss['G/loss_idt_B'] = self.loss_idt_B.item()
                
                if (k) % 10 == 0:
                    log = "Intra task training, test_dataset_indx [{}], Iteration [{}/{}/{}]".format(test_dataset_indx, total_iters, k, self.update_step)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                
            #Meta-testing on the query set    
            self.fake_A_q, self.fake_B_q, self.idt_A_q, self.idt_B_q, self.gan_A_q, self.gan_B_q, self.rec_A_q, self.rec_B_q = self.forward_G(self.real_A_q, self.real_B_q, fast_weights_A, fast_weights_B, fast_weights_D_A, fast_weights_D_B)
            self.loss_G_q = self.G_losses(self.real_A_q, self.real_B_q, self.idt_A_q, self.idt_B_q, self.gan_A_q, self.gan_B_q, self.rec_A_q, self.rec_B_q)
           
            lossG_q +=  self.loss_G_q
            
            self.pred_real_B_q, self.pred_fake_B_q, self.pred_real_A_q, self.pred_fake_A_q = self.forward_D(self.real_A_q, self.real_B_q, self.fake_A_q, self.fake_B_q, fast_weights_D_A, fast_weights_D_B)
            self.loss_D_A_q, self.loss_D_B_q = self.D_losses(self.pred_real_B_q, self.pred_fake_B_q, self.pred_real_A_q, self.pred_fake_A_q)
            lossD_A_q +=  self.loss_D_A_q
            lossD_B_q +=  self.loss_D_B_q
        if total_iters % 50 == 0:        
            x_A_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_real_A.jpg'.format(test_dataset_indx, total_iters)) 
            save_image(self.denorm(self.real_A_q.data.cpu()), x_A_path)
            print("[*] Samples saved: {}".format(x_A_path))
           
            
            x_AB_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_fake_B.jpg'.format(test_dataset_indx,total_iters)) 
            save_image(self.denorm(self.fake_B_q.data.cpu()), x_AB_path)
            
            x_ABA_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_rec_A.jpg'.format(test_dataset_indx,total_iters)) 
            save_image(self.denorm(self.rec_A_q.data.cpu()), x_ABA_path)
            
            x_B_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_real_B.jpg'.format(test_dataset_indx,total_iters)) 
            save_image(self.denorm(self.real_B_q.data.cpu()), x_B_path)
            print("[*] Samples saved: {}".format(x_B_path))
           
            
            x_BA_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_fake_A.jpg'.format(test_dataset_indx,total_iters)) 
            save_image(self.denorm(self.fake_A_q.data.cpu()), x_BA_path)
            
            x_BAB_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_rec_B.jpg'.format(test_dataset_indx,total_iters)) 
            save_image(self.denorm(self.rec_B_q.data.cpu()), x_BAB_path)    


        # optimize meta parameters
        self.optimizer_G.zero_grad()  
        lossG_q.backward()           
        self.optimizer_G.step()       
        
        self.optimizer_D.zero_grad()   
        lossD_A_q.backward()      
        lossD_B_q.backward()      
        self.optimizer_D.step()  
       
        
    def finetunning(self, test_dataset_indx, total_iters, total_iters2):
        """MT-GAN inference process (It is almost the same as CycleGAN training process)"""
        netG_A = generator().to(self.device)

        netG_B = generator().to(self.device)


        netD_A = discriminator().to(self.device)
                                        
        netD_B = discriminator().to(self.device)
        netG_A.load_state_dict(self.netG_A.state_dict()) 
        netG_B.load_state_dict(self.netG_A.state_dict())
        netD_A.load_state_dict(self.netD_A.state_dict()) 
        netD_B.load_state_dict(self.netD_A.state_dict()) 
        
        
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
        
        task_num, setsz, c_, h, w = self.real_A_support.size()
        querysz = self.real_A_query.size(1)

        lossD_A_q = 0
        lossD_B_q = 0
        lossG_q = 0
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        for i in range(task_num):
            self.real_A = self.real_A_support[i]
            self.real_B = self.real_B_support[i]
            self.real_A_q = self.real_A_query[i]
            self.real_B_q = self.real_B_query[i]
            x_A_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ft_real_A.jpg'.format(test_dataset_indx,self.finetune_step,total_iters2)) 
            save_image(self.denorm(self.real_A_q.data.cpu()), x_A_path)
            print("[*] Samples saved: {}".format(x_A_path))
            x_B_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ft_real_B.jpg'.format(test_dataset_indx,self.finetune_step,total_iters2)) 
            save_image(self.denorm(self.real_B_q.data.cpu()), x_B_path)
            print("[*] Samples saved: {}".format(x_B_path))
            
            self.mt_loss_cycle = []
            for k in range(1, self.finetune_step):

                self.fake_B = netG_A(self.real_A) 
                self.rec_A = netG_B(self.fake_B)   
                self.fake_A = netG_B(self.real_B)  
                self.rec_B = netG_A(self.fake_A)   
                
                self.set_requires_grad([netD_B, netD_A], False) 
                optimizer_G.zero_grad()
                
                if lambda_idt > 0:
                    
                    self.idt_A = netG_A(self.real_B)
                    self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                    
                    self.idt_B = netG_B(self.real_A)
                    self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
                else:
                    self.loss_idt_A = 0
                    self.loss_idt_B = 0

               
                self.loss_G_A = self.criterionGAN(netD_A(self.fake_B), True)
                
                self.loss_G_B = self.criterionGAN(netD_B(self.fake_A), True)
                
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
                
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
                
                self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
                self.loss_G.backward()
                optimizer_G.step() 
                
                self.set_requires_grad([netD_A, netD_B], True)
                optimizer_D.zero_grad() 
                
                fake_B = self.fake_B_pool.query(self.fake_B)
                self.loss_D_A = self.backward_D_basic(netD_A, self.real_B, fake_B)
                fake_A = self.fake_A_pool.query(self.fake_A)
                self.loss_D_B = self.backward_D_basic(netD_B, self.real_A, fake_A)
                optimizer_D.step()  

                
                self.mt_loss_cycle.append((self.loss_cycle_B+self.loss_cycle_A).data.cpu().detach().numpy())
                if (k) % 100 == 0:
                    loss = {}
                    loss['MTDA/loss'] = self.loss_D_A.item()
                    loss['MTDB/loss'] = self.loss_D_B.item()
                    loss['MTGA/loss_G_A'] = self.loss_G_A.item()
                    loss['MTGA/loss_cycle_A'] = self.loss_cycle_A.item()
                    loss['MTGA/loss_idt_A'] = self.loss_idt_A.item()
                    loss['MTGA/loss_G_B'] = self.loss_G_B.item()
                    loss['MTGA/loss_cycle_B'] = self.loss_cycle_B.item()
                    loss['MTGB/loss_idt_B'] = self.loss_idt_B.item()
                    
                    
                    log = "During fine tuning, test_dataset_indx [{}], Iteration [{}/{}/{}]".format(test_dataset_indx, total_iters, k, self.update_step)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                        self.logger.scalar_summary(tag, value, k)
                    print(log)
                if (k+1) % 500 == 0:
                    self.fake_B_q = netG_A(self.real_A_q) 
                    self.rec_A_q = netG_B(self.fake_B_q)  
                    self.fake_A_q = netG_B(self.real_B_q)  
                    self.rec_B_q = netG_A(self.fake_A_q)   
                    x_AB_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ft_fake_B.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.fake_B_q.data.cpu()), x_AB_path)
                    x_BA_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ft_fake_A.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.fake_A_q.data.cpu()), x_BA_path)
                    x_ABA_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ft_rec_A.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.rec_A_q.data.cpu()), x_ABA_path)
                    x_BAB_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ft_rec_B.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.rec_B_q.data.cpu()), x_BAB_path)
                
               
           
    def finetunning_withoutmeta(self, test_dataset_indx, total_iters, total_iters2):
        """CycleGAN training process with random initialization"""
        netG_A = generator().to(self.device)
        netG_B = generator().to(self.device)

        netD_A = discriminator().to(self.device)
        netD_B = discriminator().to(self.device)

        
        
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
        
        task_num, setsz, c_, h, w = self.real_A_support.size()
        querysz = self.real_A_query.size(1)

        lossD_A_q = 0
        lossD_B_q = 0
        lossG_q = 0
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        for i in range(task_num):
            self.real_A = self.real_A_support[i]
            self.real_B = self.real_B_support[i]
            self.real_A_q = self.real_A_query[i]
            self.real_B_q = self.real_B_query[i]
            
            x_A_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ftwt_real_A.jpg'.format(test_dataset_indx,self.finetune_step,total_iters2)) 
            save_image(self.denorm(self.real_A_q.data.cpu()), x_A_path)
            print("[*] Samples saved: {}".format(x_A_path))
            
            x_B_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ftwt_real_B.jpg'.format(test_dataset_indx,self.finetune_step,total_iters2)) 
            save_image(self.denorm(self.real_B_q.data.cpu()), x_B_path)
            print("[*] Samples saved: {}".format(x_B_path))
            self.cyc_loss_cycle = []
            for k in range(1, self.finetune_step):
                
                self.fake_B = netG_A(self.real_A)  
                self.rec_A = netG_B(self.fake_B)   
                self.fake_A = netG_B(self.real_B)  
                self.rec_B = netG_A(self.fake_A)  
                
                
                self.set_requires_grad([netD_B, netD_A], False) 
                optimizer_G.zero_grad()
                
                if lambda_idt > 0:
                    
                    self.idt_A = netG_A(self.real_B)
                    self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                    
                    self.idt_B = netG_B(self.real_A)
                    self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
                else:
                    self.loss_idt_A = 0
                    self.loss_idt_B = 0

                
                self.loss_G_A = self.criterionGAN(netD_A(self.fake_B), True)
                
                self.loss_G_B = self.criterionGAN(netD_B(self.fake_A), True)
                
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
                
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
                
                self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
                self.loss_G.backward()
                optimizer_G.step() 
               
                self.set_requires_grad([netD_A, netD_B], True)
                optimizer_D.zero_grad() 
                
                fake_B = self.fake_B_pool.query(self.fake_B)
                self.loss_D_A = self.backward_D_basic(netD_A, self.real_B, fake_B)
                fake_A = self.fake_A_pool.query(self.fake_A)
                self.loss_D_B = self.backward_D_basic(netD_B, self.real_A, fake_A)
                optimizer_D.step()  


                self.cyc_loss_cycle.append((self.loss_cycle_B+self.loss_cycle_A).data.cpu().detach().numpy())
                if (k) % 100 == 0:
                    loss = {}
                    loss['CycleDA/loss'] = self.loss_D_A.item()
                    loss['CycleDB/loss'] = self.loss_D_B.item()
                    loss['CycleGA/loss_G_A'] = self.loss_G_A.item()
                    loss['CycleGA/loss_cycle_A'] = self.loss_cycle_A.item()
                    loss['CycleGA/loss_idt_A'] = self.loss_idt_A.item()
                    loss['CycleGA/loss_G_B'] = self.loss_G_B.item()
                    loss['CycleGA/loss_cycle_B'] = self.loss_cycle_B.item()
                    loss['CycleGB/loss_idt_B'] = self.loss_idt_B.item()

                    log = "During fine tuning without meta, test_dataset_indx [{}], Iteration [{}/{}/{}]".format(test_dataset_indx, total_iters, k, self.update_step)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                        self.logger.scalar_summary(tag, value, k)
                    print(log)
                       
                if (k+1) % 500 == 0:
                    self.fake_B_q = netG_A(self.real_A_q)  
                    self.rec_A_q = netG_B(self.fake_B_q)   
                    self.fake_A_q = netG_B(self.real_B_q)  
                    self.rec_B_q = netG_A(self.fake_A_q)   
                    x_AB_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ftwt_fake_B.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.fake_B_q.data.cpu()), x_AB_path)
                    x_BA_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ftwt_fake_A.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.fake_A_q.data.cpu()), x_BA_path)    
                    x_ABA_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ftwt_rec_A.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.rec_A_q.data.cpu()), x_ABA_path)
                    x_BAB_path = os.path.join('./checkpoints',self.experiment_name, 'images/', '{}_{}_{}_ftwt_rec_B.jpg'.format(test_dataset_indx,k+1,total_iters2)) 
                    save_image(self.denorm(self.rec_B_q.data.cpu()), x_BAB_path)     
    def plot_training_loss(self,test_dataset_indx):
        plt.plot(range(self.finetune_step-1), self.cyc_loss_cycle, color='green', label='CycleGAN', linestyle='--')
        plt.plot(range(self.finetune_step-1), self.mt_loss_cycle, color='red', label='MT-GAN', linestyle='-')
        plt.xlabel("Step",fontsize=15)
        plt.ylabel("Cycle-Consistency Loss",fontsize=15)
        plt.grid()
        plt.legend(fontsize=20)
        plt.savefig('logging_task{}.png'.format(test_dataset_indx))   
        plt.close()   
        print('logging_task{} png has been saved'.format(test_dataset_indx))
    