from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import torch.nn.init as init
from Lib.config import config


class elasnet_prox(nn.Module):
    r"""Applies the elastic net proximal operator,
    NOTS: it will degenerate to ell1_prox if mu=0.0

    The elastic net proximal operator function is given as the following function
    \argmin_{x} \lambda ||x||_1 + \mu /2 ||x||_2^2 + 0.5 ||x - input||_2^2

    Args:
      lambd: the :math:`\lambda` value on the ell_1 penalty term. Default: 0.5
      mu:    the :math:`\mu` value on the ell_2 penalty term. Default: 0.0

    Shape:
      - Input: :math:`(N, *)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(N, *)`, same shape as the input

    """

    def __init__(self, lambd=0.5, mu=0.0):
        super(elasnet_prox, self).__init__()
        self.lambd = lambd
        self.scaling_mu = 1.0 / (1.0 + mu)

    def forward(self, input):
        return F.softshrink(input * self.scaling_mu, self.lambd * self.scaling_mu)

    def extra_repr(self):
        return '{} {}'.format(self.lambd, self.scaling_mu)

class FWCC():
    # Frank-wolf
    def __init__(self, tau=10, l_search=1,) -> None:
        # tau is the limit of rank
        self.tau = tau
        self.l_search = l_search
        if torch.cuda.is_available():
    # 创建一个 CUDA 张量
            self.device = torch.device("cuda")
        else:
            # 如果没有可用的 CUDA 设备，则使用 CPU
            self.device = torch.device("cpu")

    
    # 张量展开为矩阵
    def ten2mat(self, ten:torch.Tensor, dim, k):
        # shape is [1, ndict, h, w]
        permuted_ten = ten.permute([k]+list(range(0,k))+list(range(k+1,len(dim))))
        col_dim = 1
        for i in range(len(dim)):
            if i != k:
                col_dim *= dim[i]
        mat = permuted_ten.reshape(dim[k], col_dim)

        return mat
    
    # 矩阵折叠为张量
    def mat2ten(self, mat:torch.Tensor,dim, k):
        dim0 = []
        dim0.append(dim[k])
        dim0 += dim[0:k]
        dim0 += dim[k+1:len(dim)]
        ten = mat.reshape(dim0).permute(list(range(1,k+1))+[0]+list(range(k+1,len(dim))))

        return ten
    
    # 无法批量操作
    def _powermethod(self, X:torch.Tensor, maxIter=None, u=None):
        if not maxIter:
            maxIter = 100
        
        n,m = X.size()
        
        # z = torch.rand(m,1).to(dtype=torch.float, device=self.device)
        z = 1/8 * torch.ones(m,1).to(dtype=torch.float, device=self.device)
        y = torch.mm(X, z)
        
        y = y/torch.norm(y,2)
        if u != None:
            y = u
        
        for t in range(maxIter):
            tmp = torch.mm(X.T, y)
            y = torch.mm(X,tmp)
            normy = torch.norm(y,2)
            if normy != 0:
                y = y/normy
        
        
        b = torch.mm(X.T, y)
        sigma = torch.norm(b,2)
        if sigma != 0:
            v = b/sigma
        else:
            v = b
        
        u = y
        
        return sigma, u, v
    
    def _subproblem(self, X:torch.Tensor):
        # X = F'(X)
        dims = X.size()
        fac_size = torch.sqrt(torch.tensor(dims).cuda())
        D = len(dims)
        
        ul = [None] * D
        vl = [None] * D
        sigmal = [None] * D
        
        X2 = [None] * D
        for d in list(range(D)):
            
            rdim = d
            cdims = list(range(len(X.size())))
            cdims = [dim for dim in cdims if dim != rdim]
            X2[d] = self.ten2mat(X, dims, d)
            sigmal[d], ul[d], vl[d] = self._powermethod(X2[d],maxIter=3)
            sigmal[d] = sigmal[d] * fac_size[d]
            
        sigmaT = torch.tensor(sigmal)
        max_d = torch.argmax(sigmaT)
        u = ul[max_d]
        sigma, u, v = self._powermethod(X2[max_d], maxIter=8, u=u)
        
        newcomp = self.tau*fac_size[max_d]* u@v.T
        newcomp = self.mat2ten(newcomp, dims, max_d)
        return newcomp
            
    
    def _linesearch(self, X, newcomp, X2):
        
        cpx = newcomp - X
        
        center = - torch.sum(-X2*cpx) /(torch.norm(-cpx)**2)

        if center<0:
            gamma = 0
            
        elif center > 1:
            gamma = 1
            
        else:
            gamma = center
        
        return gamma

    
    def fwforward(self,data_train, ):
        '''
        data_train: [m,h,w]
        '''
        dims = data_train.size()
        X = torch.zeros(dims)
        fac_size = torch.sqrt(torch.tensor(dims))
        for t in range(self.maxIter):
            gamma = 2/(t+2)
            X2 = data_train - X
            j, _, u, v = self._subproblem(X2, dims, fac_size)
            newcomp = self.tau*fac_size[j]* u@v.T
            newcomp = self.mat2ten(newcomp, dims, j)
            if self.l_search == 1:
                gamma = self._linesearch(X, newcomp, X2)
            
            if gamma == 0:
                gamma = 1e-6
            
            X = (1-gamma)*X + gamma * newcomp
        
        return X
    
class DictBlock(nn.Module):
    # c = argmin_c lmbd * ||c||_1  +  mu/2 * ||c||_2^2 + 1 / 2 * ||x - weight (@conv) c||_2^2
    def __init__(self, n_channel, dict_size, mu=0.0, lmbd=0.0, n_dict=1, non_negative=True,  # model parameters
                 stride=1, kernel_size=3, padding=1, share_weight=True, square_noise=True,  # optional model parameters
                 n_steps=10, step_size_fixed=True, step_size=0.1, w_norm=True, padding_mode="constant"):  # training parameters
        super(DictBlock, self).__init__()

        self.mu = mu
        self.lmbd = lmbd  # LAMBDA
        self.n_dict = n_dict
        self.stride = stride
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = padding
        self.padding_mode = padding_mode
        assert self.padding_mode in ['constant', 'reflect', 'replicate', 'circular']
        self.groups = 1
        self.n_steps = n_steps
        self.conv_transpose_output_padding = 0 if stride == 1 else 1
        self.w_norm = w_norm
        self.non_negative = non_negative
        self.v_max = None
        self.v_max_error = 0.
        self.xsize = None
        self.zsize = None
        self.lmbd_ = None
        self.square_noise = square_noise

        # n_variables = 1 if share_weight else self.n_steps
        self.weight = nn.Parameter(torch.Tensor(dict_size, self.n_dict * n_channel, kernel_size, kernel_size))

        with torch.no_grad():
            init.kaiming_uniform_(self.weight)

        # variables that are needed for ISTA/FISTA
        self.nonlinear = elasnet_prox(self.lmbd * step_size, self.mu * step_size)

        self.register_buffer('step_size', torch.tensor(step_size, dtype=torch.float))

        # if torch.cuda.device_count() == 1:
        #     self.register_buffer('step_size', torch.tensor(step_size, dtype=torch.float))
        # elif torch.cuda.device_count() > 1:
        #     # if # of gpus larger than 1,
        #     self.step_size = step_size
        # else:
        #     raise NotImplementedError

        # if step_size_fixed:
        #     self.step_size = [step_size for _ in range(n_variables)]
        # else:
        #     self.step_size = nn.ParameterList(
        #         [nn.Parameter(torch.Tensor([step_size]))  # [math.sqrt(dict_size / n_channel)]))
        #          for _ in range(n_variables)])

    def fista_forward(self, x):

        # self.c_error = []
        for i in range(self.n_steps):

            weight = self.weight
            step_size = self.step_size

            if i == 0:
                c_pre = 0.

                c = step_size * F.conv2d(x.repeat(1, self.n_dict, 1, 1), weight, bias=None, stride=self.stride,
                                         padding=self.padding)

                # print(f"c.device: {c.device}")
                # print(self.nonlinear)
                c = self.nonlinear(c)
            elif i == 1:
                c_pre = c
                # weight = self.normalize(weight)
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp

                if self.square_noise:
                    gra = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                else:

                    w = r.view(r.size(0), -1)
                    normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w).detach()
                    w = (w / normw).view(r.size())

                    gra = F.conv2d(w, weight, bias=None, stride=self.stride, padding=self.padding) * 0.5

                c = c + step_size * gra
                c = self.nonlinear(c)
                t = (math.sqrt(5.0) + 1.0) / 2.0
            else:
                t_pre = t
                t = (math.sqrt(1.0 + 4.0 * t_pre * t_pre) + 1) / 2.0
                a = (t_pre + t - 1.0) / t * c + (1.0 - t_pre) / t * c_pre
                c_pre = c
                # weight = self.normalize(weight)
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp

                if self.square_noise:
                    gra = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                else:

                    w = r.view(r.size(0), -1)
                    normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w).detach()
                    w = (w / normw).view(r.size())

                    gra = F.conv2d(w, weight, bias=None, stride=self.stride, padding=self.padding) * 0.5

                c = a + step_size * gra
                c = self.nonlinear(c)

            if self.non_negative:
                c = F.relu(c)

            # self.c_error.append(torch.sum((c) ** 2) / c.shape[0])
        return c, weight
    
    def frankwolf_forward(self, x, fwcc, svdFun):
        weight = self.weight
        
        if x.sum() == torch.zeros(1).cuda():
            return F.conv2d(x, weight, bias=None, stride=self.stride,
                                         padding=self.padding), weight
        
        weight = self.weight
        yitaA = 8*torch.norm(weight, 2)**2
        
        for t in range(self.n_steps):
            gamma = 2/(t+2)
            # Ax
            if t == 0:
                
                X = F.conv2d(x.repeat(1, self.n_dict, 1, 1), weight, bias=None, stride=self.stride,
                                         padding=self.padding)
                # 1.近似 
                X /=  yitaA
                ## A'T A 逆的做法
                # X = F.conv_transpose2d(X, weight, bias=None, stride=self.stride,
                #                          padding=self.padding)
                
                # X = 1/2 * F.conv2d(X, weight, bias=None, stride=self.stride,
                #                          padding=self.padding)
                
                beta = X
                
                X2 = beta
                dims = X2.size()
                fac_size = torch.sqrt(torch.tensor(dims))
                # X Az
                c = torch.zeros(dims).to(device=fwcc.device)
                
                
            else:
                c_pre = c
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride,
                                         padding=self.padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp
                A2 = F.conv2d(r, weight, bias=None, stride=self.stride,
                                         padding=self.padding)
                A1 = F.conv2d(xp, weight, bias=None, stride=self.stride,
                                         padding=self.padding)
                
                X = A1+A2
                
                X /= yitaA
                ## A'T A 逆的做法
                # X = F.conv_transpose2d(X, weight, bias=None, stride=self.stride,
                #                          padding=self.padding)
                
                # beta = 1/2 * F.conv2d(X, weight, bias=None, stride=self.stride,
                #                          padding=self.padding)
                
                
                beta = X
                X2 = beta - c_pre
                
            newcomp = fwcc._subproblem(X2)
            # fft-svd
            # newcomp = svdFun.forward(X2, 5.0, n=X2.shape[1])
            
            if fwcc.l_search == 1:
                
                gamma = fwcc._linesearch(c, newcomp, X2)
                # print("gamma",gamma)
            if gamma == 0:
                gamma = 1e-6
            
            c = (1-gamma)*c + gamma * newcomp
        
        # c = X
        # if self.non_negative:
        #         c = F.relu(c)
        return c, weight

    def forward(self, x):

        if self.xsize is None:
            self.xsize = (x.size(-3), x.size(-2), x.size(-1))
            print(self.xsize)
        else:
            assert self.xsize[-3] == x.size(-3) and self.xsize[-2] == x.size(-2) and self.xsize[-1] == x.size(-1)

        if self.w_norm:
            self.normalize_weight()

        # self.c_error = []
        # c, weight = self.fista_forward(x)
        fwcc = FWCC(tau=100, l_search=1)
        c, weight = self.frankwolf_forward(x, fwcc, None)

        # Compute loss
        xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                output_padding=self.conv_transpose_output_padding)
        r = x.repeat(1, self.n_dict, 1, 1) - xp
        r_loss = torch.sum(torch.pow(r, 2)) / self.n_dict
        c_loss = self.lmbd * torch.sum(torch.abs(c)) + self.mu / 2. * torch.sum(torch.pow(c, 2))

        if self.zsize is None:
            self.zsize = (c.size(-3), c.size(-2), c.size(-1))
            print(self.zsize)
        else:
            assert self.zsize[-3] == c.size(-3) and self.zsize[-2] == c.size(-2) and self.zsize[-1] == c.size(-1)

        if self.lmbd_ is None and config.MODEL.ADAPTIVELAMBDA:
            self.lmbd_ = self.lmbd * self.xsize[-3] * self.xsize[-2] * self.xsize[-1] / (self.zsize[-3] * self.zsize[-2] * self.zsize[-1])
            self.lmbd = self.lmbd_
            print("======")
            print("xsize", self.xsize)
            print("zsize", self.zsize)
            print("new lmbd: ", self.lmbd)

        return c, (r_loss, c_loss)

    def update_stepsize(self):
        step_size = 0.9 / self.power_iteration(self.weight)
        self.step_size = self.step_size * 0. + step_size
        # self.step_size.fill_(step_size)
        self.nonlinear.lambd = self.lmbd * step_size
        self.nonlinear.scaling_mu = 1.0 / (1.0 + self.mu * step_size)

    def normalize_weight(self):
        with torch.no_grad():
            w = self.weight.view(self.weight.size(0), -1)
            normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w)
            w = (w / normw).view(self.weight.size())
            self.weight.data = w.data

    def power_iteration(self, weight):

        max_iteration = 50
        v_max_error = 1.0e5
        tol = 1.0e-5
        k = 0

        with torch.no_grad():
            if self.v_max is None:
                c = weight.shape[0]
                v = torch.randn(size=(1, c, self.zsize[-2], self.zsize[-1])).to(weight.device)
            else:
                v = self.v_max.clone()

            while k < max_iteration and v_max_error > tol:

                tmp = F.conv_transpose2d(
                    v, weight, bias=None, stride=self.stride, padding=self.padding,
                    output_padding=self.conv_transpose_output_padding
                )
                v_ = F.conv2d(tmp, weight, bias=None, stride=self.stride, padding=self.padding)
                v_ = F.normalize(v_.view(-1), dim=0, p=2).view(v.size())
                v_max_error = torch.sum((v_ - v) ** 2)
                k += 1
                v = v_

            v_max = v.clone()
            Dv_max = F.conv_transpose2d(
                v_max, weight, bias=None, stride=self.stride, padding=self.padding,
                output_padding=self.conv_transpose_output_padding
            )  # Dv

            lambda_max = torch.sum(Dv_max ** 2).item()  # vTDTDv / vTv, ignore the vTv since vTv = 1

        self.v_max = v_max
        
        return lambda_max
