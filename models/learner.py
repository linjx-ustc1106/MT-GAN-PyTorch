import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import torchvision.models as models
class resnet_feature(nn.Module):
    name = 'resnet_feature'
    def __init__(self):
        super(resnet_feature, self).__init__()
        res50_model = models.resnet50(pretrained=True)
        
        self.res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
        
        self.avp = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        output = self.avp(self.res50_conv(x))
        return output.view(output.size(0), output.size(1))

    
class generator(nn.Module):
    """
    generator of cyclegan
    """

    def __init__(self, config = None):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(generator, self).__init__()


        self.config = [
            ('conv2d', [64, 3, 7, 7, 1, 3]),
            ('bn', [64]),
            ('relu', [True]),
            ('conv2d', [128, 64, 3, 3, 2, 1]),
            ('bn', [128]),
            ('relu', [True]),
            ('conv2d', [256, 128, 3, 3, 2, 1]),
            ('bn', [256]),
            ('relu', [True]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('convt2d', [256, 128, 3, 3, 2, 1, 1]),
            ('bn', [128]),
            ('relu', [True]),
            ('convt2d', [128, 64, 3, 3, 2, 1, 1]),
            ('bn', [64]),
            ('relu', [True]),
            ('conv2d', [3, 64, 7, 7, 1, 3]),
            ('tanh', [True])
         
            ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.vars_bn_res = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'conv2d_res':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w0 = nn.Parameter(torch.ones(*param[0][:4]))
                torch.nn.init.kaiming_normal_(w0)
                self.vars.append(w0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                wb0 = nn.Parameter(torch.ones(param[0][0]))
                self.vars.append(wb0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                running_mean0 = nn.Parameter(torch.zeros(param[0][0]), requires_grad=False)
                running_var0 = nn.Parameter(torch.ones(param[0][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean0, running_var0])
                
                w1 = nn.Parameter(torch.ones(*param[1][:4]))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))
                
                
                wb1 = nn.Parameter(torch.ones(param[1][0]))
                self.vars.append(wb1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))

                # must set requires_grad=False
                running_mean1 = nn.Parameter(torch.zeros(param[1][0]), requires_grad=False)
                running_var1 = nn.Parameter(torch.ones(param[1][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean1, running_var1])

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'in']:
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
               

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def Resforward(self, x, idx, bn_res_idx, param, vars, bn_training=True):
        w0, b0 = vars[idx], vars[idx + 1]
        # remember to keep synchrozied of forward_encoder and forward_decoder!
        x = F.conv2d(x, w0, b0, stride=param[0][4], padding=param[0][5])
        
        wb0, bb0 = vars[idx+2], vars[idx + 3]
        running_mean0, running_var0 = self.vars_bn_res[bn_res_idx], self.vars_bn_res[bn_res_idx+1]
        x = F.batch_norm(x, running_mean0, running_var0, weight=wb0, bias=bb0, training=bn_training)
        x = F.relu(x, inplace=param[0][0])
        
        w1, b1 = vars[idx + 4], vars[idx + 5]
        x = F.conv2d(x, w1, b1, stride=param[1][4], padding=param[1][5])
        wb1, bb1 = vars[idx+6], vars[idx + 7]
        running_mean1, running_var1 = self.vars_bn_res[bn_res_idx+2], self.vars_bn_res[bn_res_idx+3]
        x = F.batch_norm(x, running_mean1, running_var1, weight=wb1, bias=bb1, training=bn_training)
        return x
    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        bn_res_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5], output_padding = param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'conv2d_res':
                x = x + self.Resforward( x, idx, bn_res_idx, param, vars, True) 

                idx += 8
                bn_res_idx += 4
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        assert bn_res_idx == len(self.vars_bn_res)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):

        return self.vars
class discriminator(nn.Module):
    """
    discriminator of cyclegan
    """

    def __init__(self, config = None):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(discriminator, self).__init__()


        self.config = [
            ('conv2d', [64, 3, 4, 4, 2, 1]),
            ('leakyrelu', [0.2, True]),
            ('conv2d', [128, 64, 4, 4, 2, 1]),
            ('bn', [128]),
            ('leakyrelu', [0.2, True]),
            ('conv2d', [256, 128, 4, 4, 2, 1]),
            ('bn', [256]),
            ('leakyrelu', [0.2, True]),
            ('conv2d', [512, 256, 4, 4, 1, 1]),
            ('bn', [512]),
            ('leakyrelu', [0.2, True]),
            ('conv2d', [1, 512, 4, 4, 1, 1])
            ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.vars_bn_res = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'conv2d_res':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w0 = nn.Parameter(torch.ones(*param[0][:4]))
                torch.nn.init.kaiming_normal_(w0)
                self.vars.append(w0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                wb0 = nn.Parameter(torch.ones(param[0][0]))
                self.vars.append(wb0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                running_mean0 = nn.Parameter(torch.zeros(param[0][0]), requires_grad=False)
                running_var0 = nn.Parameter(torch.ones(param[0][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean0, running_var0])
                
                w1 = nn.Parameter(torch.ones(*param[1][:4]))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))
                
                
                wb1 = nn.Parameter(torch.ones(param[1][0]))
                self.vars.append(wb1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))

                # must set requires_grad=False
                running_mean1 = nn.Parameter(torch.zeros(param[1][0]), requires_grad=False)
                running_var1 = nn.Parameter(torch.ones(param[1][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean1, running_var1])

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'in']:
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
               

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def Resforward(self, x, idx, bn_res_idx, param, vars, bn_training=True):
        w0, b0 = vars[idx], vars[idx + 1]
        # remember to keep synchrozied of forward_encoder and forward_decoder!
        x = F.conv2d(x, w0, b0, stride=param[0][4], padding=param[0][5])
        
        wb0, bb0 = vars[idx+2], vars[idx + 3]
        running_mean0, running_var0 = self.vars_bn_res[bn_res_idx], self.vars_bn_res[bn_res_idx+1]
        x = F.batch_norm(x, running_mean0, running_var0, weight=wb0, bias=bb0, training=bn_training)
        x = F.relu(x, inplace=param[0][0])
        
        w1, b1 = vars[idx + 4], vars[idx + 5]
        x = F.conv2d(x, w1, b1, stride=param[1][4], padding=param[1][5])
        wb1, bb1 = vars[idx+6], vars[idx + 7]
        running_mean1, running_var1 = self.vars_bn_res[bn_res_idx+2], self.vars_bn_res[bn_res_idx+3]
        x = F.batch_norm(x, running_mean1, running_var1, weight=wb1, bias=bb1, training=bn_training)
        return x
    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        bn_res_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'conv2d_res':
                x = x + self.Resforward( x, idx, bn_res_idx, param, vars, True) 

                idx += 8
                bn_res_idx += 4
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        assert bn_res_idx == len(self.vars_bn_res)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
class generator_st(nn.Module):
    """
    generator of stargan
    """

    def __init__(self, config = None, c_dim = 2):

        super(generator_st, self).__init__()

        self.config = [
            ('conv2d', [64, 3+c_dim, 7, 7, 1, 3]),
            ('bn', [64]),
            ('relu', [True]),
            ('conv2d', [128, 64, 3, 3, 2, 1]),
            ('bn', [128]),
            ('relu', [True]),
            ('conv2d', [256, 128, 3, 3, 2, 1]),
            ('bn', [256]),
            ('relu', [True]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('conv2d_res', [[256, 256, 3, 3, 1, 1],[256, 256, 3, 3, 1, 1]]),
            ('convt2d', [256, 128, 3, 3, 2, 1, 1]),
            ('bn', [128]),
            ('relu', [True]),
            ('convt2d', [128, 64, 3, 3, 2, 1, 1]),
            ('bn', [64]),
            ('relu', [True]),
            ('conv2d', [3, 64, 7, 7, 1, 3]),
            ('tanh', [True])
         
            ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.vars_bn_res = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'conv2d_res':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w0 = nn.Parameter(torch.ones(*param[0][:4]))
                torch.nn.init.kaiming_normal_(w0)
                self.vars.append(w0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                wb0 = nn.Parameter(torch.ones(param[0][0]))
                self.vars.append(wb0)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0][0])))
                
                running_mean0 = nn.Parameter(torch.zeros(param[0][0]), requires_grad=False)
                running_var0 = nn.Parameter(torch.ones(param[0][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean0, running_var0])
                
                w1 = nn.Parameter(torch.ones(*param[1][:4]))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))
                
                
                wb1 = nn.Parameter(torch.ones(param[1][0]))
                self.vars.append(wb1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1][0])))

                # must set requires_grad=False
                running_mean1 = nn.Parameter(torch.zeros(param[1][0]), requires_grad=False)
                running_var1 = nn.Parameter(torch.ones(param[1][0]), requires_grad=False)
                self.vars_bn_res.extend([running_mean1, running_var1])

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'in']:
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
               

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def Resforward(self, x, idx, bn_res_idx, param, vars, bn_training=True):
        w0, b0 = vars[idx], vars[idx + 1]
        # remember to keep synchrozied of forward_encoder and forward_decoder!
        x = F.conv2d(x, w0, b0, stride=param[0][4], padding=param[0][5])
        
        wb0, bb0 = vars[idx+2], vars[idx + 3]
        running_mean0, running_var0 = self.vars_bn_res[bn_res_idx], self.vars_bn_res[bn_res_idx+1]
        x = F.batch_norm(x, running_mean0, running_var0, weight=wb0, bias=bb0, training=bn_training)
        x = F.relu(x, inplace=param[0][0])
        
        w1, b1 = vars[idx + 4], vars[idx + 5]
        x = F.conv2d(x, w1, b1, stride=param[1][4], padding=param[1][5])
        wb1, bb1 = vars[idx+6], vars[idx + 7]
        running_mean1, running_var1 = self.vars_bn_res[bn_res_idx+2], self.vars_bn_res[bn_res_idx+3]
        x = F.batch_norm(x, running_mean1, running_var1, weight=wb1, bias=bb1, training=bn_training)
        return x
    def forward(self, x, c, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        bn_res_idx = 0
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5], output_padding = param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'conv2d_res':
                x = x + self.Resforward( x, idx, bn_res_idx, param, vars, True) 

                idx += 8
                bn_res_idx += 4
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        assert bn_res_idx == len(self.vars_bn_res)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
class discriminator_st(nn.Module):
    """
    discriminator of stargan
    """

    def __init__(self, config = None, imgc = 3, imgsz = 128, c_dim = 2):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(discriminator_st, self).__init__()
        kernel_size = int(imgsz / np.power(2, 4))

        self.config = [
            ('conv2d', [64, 3, 4, 4, 2, 1]),
            ('leakyrelu', [0.01, True]),
            ('conv2d', [128, 64, 4, 4, 2, 1]),
            ('bn', [128]),
            ('leakyrelu', [0.01, True]),
            ('conv2d', [256, 128, 4, 4, 2, 1]),
            ('bn', [256]),
            ('leakyrelu', [0.01, True]),
            ('conv2d', [512, 256, 4, 4, 2, 1]),
            ('bn', [512]),
            ('leakyrelu', [0.01, True]),
            ('conv2d_output', [[1, 512, 3, 3, 1, 1],[c_dim, 512, kernel_size, kernel_size, 1, 0]])
            ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.vars_bn_res = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'conv2d_output':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w0 = nn.Parameter(torch.ones(*param[0][:4]))
                torch.nn.init.kaiming_normal_(w0)
                self.vars.append(w0)
                # [ch_out]
                
                
                
                w1 = nn.Parameter(torch.ones(*param[1][:4]))
                torch.nn.init.kaiming_normal_(w1)
                self.vars.append(w1)
                # [ch_out]
               
            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'in']:
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
               

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        bn_res_idx = 0
        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'conv2d_output':
                w0 = vars[idx]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                out_src = F.conv2d(x, w0, None, stride=param[0][4], padding=param[0][5])
                
               
                w1 = vars[idx + 1]
                out_cls = F.conv2d(x, w1, None, stride=param[1][4], padding=param[1][5])

                idx += 2
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        #assert bn_res_idx == len(self.vars_bn_res)

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars