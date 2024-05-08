import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pywt
import cv2
import numpy as np
import torch

gau_k = 0.1
dwt_k = 20000

device = torch.device('cuda:1' if (torch.cuda.is_available()) else 'cpu')
from pytorch_wavelets import DWTForward, DWTInverse
dwt = DWTForward(J=5, wave='sym2', mode='periodization').to(device)

""" Performs a 2d DWT Forward decomposition of an image
Args:
    J (int): Number of levels of decomposition
    wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
        Can be:
        1) a string to pass to pywt.Wavelet constructor
        2) a pywt.Wavelet class
        3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
    mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
        padding scheme
    """

# from focal_frequency_loss import FocalFrequencyLoss as FFL
# ffl = FFL(loss_weight=0, alpha=1.0)

def get_gaussian_kernel(size=21):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def find_fake_freq(im, gauss_kernel, index=None):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    # 提取低频
    low_freq = gaussian_blur(im, gauss_kernel, padding=padding)
    # 转换为灰度
    im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    # 用灰度图片进行一次低频的过滤
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    # im_gray - low_gray 灰度减去低频 剩下高频
    return torch.cat((low_freq, im_gray - low_gray),1)

gauss_kernel = get_gaussian_kernel(21).to(device)  # 获取高斯核


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'Focal']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        """Calculate Focal Frequency loss"""
        # fake = self.fake_B.clone()
        # real = self.real_B.clone()
        # self.loss_Focal = ffl(fake, real)  # calculate focal frequency loss

        """Calculate hh_dwt loss"""
        imageA = self.real_B.clone()  #.detach().cpu().numpy()
        Al,Ah = dwt(imageA)   # 注意高通输出有一个额外的维度
        a0 = Al / torch.max(torch.abs(Al))
        a1 = Ah[0] / torch.max(torch.abs(Ah[0]))
        a2 = Ah[1] / torch.max(torch.abs(Ah[1]))
        a3 = Ah[2] / torch.max(torch.abs(Ah[2]))
        a4 = Ah[3] / torch.max(torch.abs(Ah[3]))
        a5 = Ah[4] / torch.max(torch.abs(Ah[4]))

        imageB = self.fake_B.clone()  # .detach().cpu().numpy()
        Bl, Bh = dwt(imageB)
        b0 = Bl / torch.max(torch.abs(Bl))
        b1 = Bh[0] / torch.max(torch.abs(Bh[0]))
        b2 = Bh[1] / torch.max(torch.abs(Bh[1]))
        b3 = Bh[2] / torch.max(torch.abs(Bh[2]))
        b4 = Bh[3] / torch.max(torch.abs(Bh[3]))
        b5 = Bh[4] / torch.max(torch.abs(Bh[4]))

        ll = torch.mean(torch.abs(a0 - b0))
        h1 = torch.mean(torch.abs(a1 - b1))
        h2 = torch.mean(torch.abs(a2 - b2))
        h3 = torch.mean(torch.abs(a3 - b3))
        h4 = torch.mean(torch.abs(a4 - b4))
        h5 = torch.mean(torch.abs(a5 - b5))
        dwt_loss = torch.mean(torch.stack([ll, h1, h2, h3, h4, h5], -1)) * dwt_k

        ##### 高斯核滤波
        trueB = find_fake_freq(self.real_B.clone(), gauss_kernel)
        predB = find_fake_freq(self.fake_B.clone(), gauss_kernel)
        gauss_l1 = self.criterionL1(trueB, predB)

        # combine loss and calculate gradients
        self.loss_Focal = gauss_l1 * gau_k + dwt_loss   # dwt_loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_Focal

        self.loss_G.backward()
        

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

