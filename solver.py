"""solver.py"""
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from datasets.datasets import return_data2
from datasets.datasets_ndarray import return_data # 读取用于存放gan生成的ndarray数组的数据集
from utils.utils import rm_dir, cuda, where
import pdb
import numpy as np
from bottleneck_transformer_pytorch.bottleneck_transformer_pytorch import BottleStack,MyGruNet
from bottleneck_transformer_pytorch.wgan import Generator,Discriminator,gradient_penalty
from torch import nn
from loguru import logger
import datetime
from torchvision import transforms


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Basic
        self.pixel_width = args.pixel_width
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eps = args.eps
        self.lr = args.lr
        self.y_dim = args.y_dim
        self.target = args.target
        self.dataset = args.dataset
        self.data_loader = return_data2(args)
        self.data_loader_gan = return_data(args)
        self.global_epoch = 0
        self.global_iter = 0
        self.print_ = not args.silent

        self.env_name = args.env_name
        self.tensorboard = args.tensorboard
        self.visdom = args.visdom

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)

        self.desc = args.desc
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization Tools
        self.visualization_init(args)

        # Histories
        self.history = dict()# self.history是一个字典
        self.history['acc'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0

        # Models & Optimizers
        self.model_init()
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '':
            self.load_checkpoint(self.load_ckpt) # 加载checkpoint，恢复信息

        # criterion = cuda(torch.nn.CrossEntropyLoss(), self.cuda)
        # criterion = F.cross_entropy

    def visualization_init(self, args):
        # Visdom
        if self.visdom:
            from utils.visdom_utils import VisFunc
            self.port = args.visdom_port
            self.vf = VisFunc(enval=self.env_name, port=self.port)

        # TensorboardX
        if self.tensorboard:
            from tensorboardX import SummaryWriter
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists():
                self.summary_dir.mkdir(parents=True, exist_ok=True)

            self.tf = SummaryWriter(log_dir=str(self.summary_dir))
            self.tf.add_text(tag='argument', text_string=str(args), global_step=self.global_epoch)

    def model_init(self):
        # 使用bottleneck
        self.net = cuda(MyGruNet(y_dim = self.y_dim,num_steps=self.pixel_width,input_size=self.pixel_width),self.cuda)
        # self.net = cuda(BottleStack(dim=1,fmap_size=self.pixel_width,dim_out=8,proj_factor=4,downsample=True,heads=4,dim_head=8,rel_pos_emb=False, y_dim=self.y_dim), self.cuda)
        self.G = cuda(Generator(self.pixel_width**2,400),self.cuda)
        self.D = cuda(Discriminator(self.pixel_width**2,400),self.cuda)

        # Optimizers
        # self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}],betas=(0.5, 0.999))
        self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}])
        self.optim_G = optim.Adam(self.G.parameters(), lr = self.lr, betas=(0.5, 0.9))
        self.optim_D = optim.Adam(self.D.parameters(), lr = self.lr, betas=(0.5, 0.9))

    def at_loss(self,x,y):
        x_adv = Variable(x.data,requires_grad=True)
        h_adv = self.net(x_adv)
        cost = F.cross_entropy(h_adv,y)
        self.optim.zero_grad()
        cost.backward()# 第一步反向传播求得梯度
        x_adv = x_adv - 0.03 * x_adv.grad#第二步根据梯度进行x值的更新
        x_adv2 = Variable(x_adv.data,requires_grad=True)
        h_adv = self.net(x_adv2)
        cost = F.cross_entropy(h_adv,y)
        return cost

    def interleave(self,xy, batch):
        def interleave_offsets(batch, nu):
            groups = [batch // (nu + 1)] * (nu + 1)
            for x in range(batch - sum(groups)):
                groups[-x - 1] += 1
            offsets = [0]
            for g in groups:
                offsets.append(offsets[-1] + g)
            assert offsets[-1] == batch
            return offsets
        nu = len(xy) - 1
        offsets = interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def linear_rampup(self,current):
        if self.epoch == 0:
            return 1.0
        else:
            current = np.clip(current / self.epoch, 0.0, 1.0)
            return float(current)

    def semi_loss(self, outputs_x, targets_x, outputs_u, targets_u, epoch):#[64,10],[64,10],[128,10],[128,10]
        probs_u = F.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, 75 * self.linear_rampup(epoch)

    def train(self,args):
        self.set_mode('train')
        lr = args.lr
        for e in range(1,self.epoch+1): # e从1开始算起
            self.global_epoch += 1
            # for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
            for batch_idx, (images, labels) in enumerate(self.data_loader_gan['train']): # 能不能换个数据集进行训练？
                self.global_iter += 1
                x = Variable(cuda(images, self.cuda)) # torch.Size([256, 1, 43, 43])
                y = Variable(cuda(labels, self.cuda)) # torch.Size([256])
                '''
                经过net处理之后得到一个10分类的输出,logit.shape:[100,10]，所以一个batch有100个样本
                logit[0] == [0.0545  0.1646  0.0683 -0.1407  0.0031  0.0560 -0.1895 -0.0183  0.0158  0.0183】
                '''
                logit = self.net(x)
                '''
                >>> import torch
                >>> a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
                >>> print(a)
                tensor([[ 1,  5, 62, 54],
                        [ 2,  6,  2,  6],
                        [ 2, 65,  2,  6]])
                >>> torch.max(a,1)
                torch.return_types.max(
                values=tensor([62,  6, 65]),
                indices=tensor([2, 1, 1]))
                >>> torch.max(a,1)[0]
                tensor([62,  6, 65])
                >>> torch.max(a,1)[1]
                tensor([2, 1, 1])
                >>>
                '''
                # logit.max(1)[1]其中(1)表示行的最大值，[0]表示最大的值本身,[1]表示最大的那个值在该行对应的index
                # soft_logit = F.softmax(logit)
                soft_logit = logit
                prediction = soft_logit.max(1)[1] # prediction.shape: torch.Size([100]),此时，y == [1,2,1,1,1,3,5...],prediction也是类似的形式
                correct = torch.eq(prediction, y).float().mean() # 先转换为flotaTensor，然后[0]取出floatTensor中的值：0.11999999731779099
                loss_ = F.cross_entropy(soft_logit, y) # cost也是一个Variable,计算出的cost是一个损失
                cost = loss_
                self.optim.zero_grad()
                cost.backward()  # 反向传播之后可以进行参数的更新,反向传播的是那个网络，更新的就是哪个网络的参数
                self.optim.step()
                if batch_idx % 500 == 0:
                    if self.print_:
                        logger.info('[{:03d}:{:05d}]'.format(self.global_epoch, batch_idx) + '  ' + 'acc:{:.3f} loss:{:.3f}'.format(correct, cost.data))
                    if self.tensorboard:
                        self.tf.add_scalars(main_tag='performance/acc',
                                            tag_scalar_dict={'train':correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={'train':1-correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={'train':cost.data[0]},
                                            global_step=self.global_iter)
            self.test()
            if e % 10 == 0:
                lr = lr * 0.8
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr

        if self.tensorboard:
            self.tf.add_scalars(main_tag='performance/best/acc',
                                tag_scalar_dict={'test':self.history['acc']},
                                global_step=self.history['iter'])
        print(" [*] Training Finished!")

    def test(self):
        self.set_mode('eval')

        correct = 0.
        cost = 0.
        total = 0.

        data_loader = self.data_loader['test'] # 可以直接使用mnist格式的数据集
        idx = 0
        for images, labels in data_loader:# 相当于测试的时候也是和训练的时候一样直接通过迭代器取出来的值
            idx += 1
            #pdb.set_trace()
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            # x = x.view(-1, 1, self.pixel_width,self.pixel_width)
            logit = self.net(x)
            prediction = logit.max(1)[1]
            #correct += torch.eq(prediction, y).float().sum() # 这里不是通过mean的方式，而是通过sum的方式加在一起（即：正确的样本的个数）
            cc = torch.eq(prediction, y).float().sum().item() # 这里不是通过mean的方式，而是通过sum的方式加在一起（即：正确的样本的个数）
            correct += cc
            ct = F.cross_entropy(logit, y).item()
            cost += ct
            if idx % 500 == 0:
                logger.info('【batch_id】' + '%.6s' % str(idx) + '【cur_correct】' + '%.6s' % str(cc) + '【cur_cost】' +'%.6s' % str(ct) + '【accumu_cost】' +'%.6s' % str(cost) + '【accumu_correct】' +'%.6s' % str(correct))
            total += x.size(0)

        accuracy = correct / total
        cost /= total


        if self.print_:
            logger.info('[{:03d}] TEST RESULT'.format(self.global_epoch) + ' ' + 'ACC:{:.4f}'.format(accuracy) + ' ' +'*TOP* ACC:{:.4f} at e:{:03d}, COST:{:.4f}'.format(accuracy, self.global_epoch,cost))

            if self.tensorboard:
                self.tf.add_scalars(main_tag='performance/acc',
                                    tag_scalar_dict={'test':accuracy},
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/error',
                                    tag_scalar_dict={'test':(1-accuracy)},
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/cost',
                                    tag_scalar_dict={'test':cost},
                                    global_step=self.global_iter)

        if self.history['acc'] < accuracy:
            self.history['acc'] = accuracy
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            self.save_checkpoint('best_acc.tar')

        self.set_mode('train')

    def gene_label(self): # 输出预测的标签和真实的标签
        self.set_mode('eval')
        data_loader = self.data_loader['test']
        idx = 0
        today = datetime.date.today()
        today = today.strftime('%Y-%m-%d')
        arrs_Ytrue = np.ndarray(0)
        arrs_Ypred = np.ndarray(0)
        for images, labels in data_loader:  # 相当于测试的时候也是和训练的时候一样直接通过迭代器取出来的值
            idx += 1
            # pdb.set_trace()
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            # x = x.view(-1, 1, self.pixel_width,self.pixel_width)
            logit = self.net(x)
            prediction = logit.max(1)[1]
            arrs_Ytrue = np.append(arrs_Ytrue,y.cpu().numpy())
            arrs_Ypred = np.append(arrs_Ypred,prediction.cpu().numpy())
        file_name1 = 'arrs_Ytrue{0}_{1}.npy'.format(today,self.desc)
        file_name2 = 'arrs_Ypred{0}_{1}.npy'.format(today,self.desc)
        np.save(file_name1, arrs_Ytrue)
        np.save(file_name2, arrs_Ypred)
        print(file_name1, '保存完成')
        print(file_name2, '保存完成')


    def gene_probability(self): # 以softmax的形式输出prediction，可以用于画出ROC曲线
        self.set_mode('eval')
        data_loader = self.data_loader['test']
        idx = 0
        today = datetime.date.today()
        today = today.strftime('%Y-%m-%d')
        arrs_Ytrue = np.ndarray(0)
        arrs_Ypred = np.empty(shape=[0,self.y_dim])
        for images, labels in data_loader:  # 相当于测试的时候也是和训练的时候一样直接通过迭代器取出来的值
            idx += 1
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            logit = self.net(x)
            # prediction = logit.max(1)[1]
            # prediction = F.softmax(logit)[:,1] # 属于攻击类别的概率
            prediction = F.softmax(logit) # 现在把所有类别的所有样本的概率都返回，后期处理的时候再进行筛选
            arrs_Ytrue = np.append(arrs_Ytrue, y.cpu().numpy())
            # 有梯度，则需要使用detach()来消除梯度,这里axis要表示为0，不然结果维度是1不是2
            arrs_Ypred = np.append(arrs_Ypred, prediction.detach().cpu().numpy(),axis=0)
        file_name1 = 'roc_arrs_Ytrue{0}_{1}.npy'.format(today, self.desc)
        file_name2 = 'roc_arrs_Ypred{0}_{1}.npy'.format(today, self.desc)
        np.save(file_name1, arrs_Ytrue)
        np.save(file_name2, arrs_Ypred)
        print(file_name1, '保存完成')
        print(file_name2, '保存完成')

    # 第1步，训练gan模型
    # python test.py --mode wgan --epoch 100000 --filename 'wgan.tar',训练gan模型,需要制定保存gan模型的文件名（对应类别4）
    # 对应的输入数据是指定类别的mnist格式数据集
    # python test.py - -mode wgan - -epoch 20000 - -filename 'wgan_3.tar,（对应类别3）
    # epoch指定训练的轮数
    def gan(self,filename):
        self.set_mode('train')
        print(len(self.data_loader['train']))
        for e in range(1, self.epoch + 1):  # e从1开始算起
            self.global_epoch += 1
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1
                x = Variable(cuda(images, self.cuda))  # torch.Size([256, 1, 43, 43])
                shape_0 = x.shape[0]
                x = x.view(shape_0,-1)
                predr = self.D(x)
                lossr = -predr.mean()

                # z = torch.randn(shape_0,1,self.pixel_width,self.pixel_width).cuda()
                # z = z.view(shape_0,self.pixel_width**2)
                z = torch.randn(shape_0, self.pixel_width**2).cuda()
                # z = x

                xf = self.G(z).detach() # 输入到神经网络中向量的类型要是float，不能是Byte或者Int
                predf = self.D(xf)
                lossf = predf.mean()
                # gp = gradient_penalty(self.D,xr=x,xf=xf.detach(),batch_size = shape_0)
                gp = gradient_penalty(self.D,xr=x,xf=xf.detach(),batch_size = shape_0)
                loss_D = lossr + lossf + 0.2 * gp
                self.optim_D.zero_grad()
                loss_D.backward()
                self.optim_D.step()
            if e % 5 == 0:
                lossD = loss_D
                lossG = loss_D
                for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                    shape_0 = images.shape[0]
                    z = torch.randn(shape_0, self.pixel_width**2).cuda()
                    xf = self.G(z)
                    predf = self.D(xf) # D的梯度是会去算的，也是有的，只是不对它进行更新就行
                    # max predf.mean()
                    loss_G = -predf.mean()
                    lossG = loss_G
                    # optimize
                    self.optim_G.zero_grad()
                    loss_G.backward()
                    self.optim_G.step() # 对G网络进行更新

                if e % 50 == 0:
                    logger.info('[{:03d}]'.format(e) + ' ' + 'loss D:{:.3f}, loss G:{:.3f}'.format(lossD.data,lossG.data))
            # if e >= 30:
        self.save_wgan_checkpoint(filename=filename)
            #     break
        self.set_mode('eval')

    def save_wgan_checkpoint(self, filename='wgan.tar'):  # 保存checkpoint
        states = {
            'G': self.G.state_dict(),
            'G_optim':self.optim_G.state_dict()
        }
        file_path = self.ckpt_dir / filename
        torch.save(states, file_path.open('wb+'))
        logger.info("saved checkpoint'{}'(iter{})".format(file_path, self.global_iter))

    def load_wgan_checkpoint(self, filename='wgan.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.G.load_state_dict(checkpoint['G'])
            self.optim_G.load_state_dict(checkpoint['G_optim'])
            print("=> loaded checkpoint '{}'".format(file_path))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    # 第2步，使用Generator生成样本
    # 使用训练好的gan模型生成样本，指定gan模型文件名
    # 对应的输入数据是指定类别的mnist格式数据集
    # python test.py --mode enhancement --epoch 100 --filename 'wgan.tar',epoch指定数据增强的轮数(对应类别为4)
    # python test.py - -mode enhancement - -epoch 10 - -filename 'wgan_3.tar',(对应类别为3)
    # 使用gan对指定类别的训练集进行增强
    def ganSamplerEnhancement(self,filename):
        self.load_wgan_checkpoint(filename=filename)
        arrs_data = np.empty(shape=[0,self.pixel_width**2])
        arrs_label = np.ndarray(0)
        for e in range(self.epoch):
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                shape_0 = images.shape[0]
                z = torch.randn(shape_0,self.pixel_width**2).cuda()
                output = self.G(z)
                arrs_label = np.append(arrs_label, labels.numpy(),axis=0)
                arrs_data = np.append(arrs_data, output.detach().cpu().numpy(), axis=0)
        np.save('./datasets/ARRAY/data_{0}.npy'.format(labels[0]),arrs_data)
        np.save('./datasets/ARRAY/label_{0}.npy'.format(labels[0]),arrs_label)
        print('train set saved to file')

    # 第3步，合并数据
    def merge_data(self):
        '''
        合并mnist数据集和增强之后的训练集
        :return:
        '''
        arrs_data = np.empty(shape=[0,self.pixel_width**2])
        arrs_label = np.ndarray(0)
        arrs_data_test = np.empty(shape = [0,self.pixel_width**2])
        arrs_label_test = np.ndarray(0)
        for batch_idx,(images, labels) in enumerate(self.data_loader['train']): # mnist训练集
            shape0 = images.shape[0]
            images = images.view(shape0,self.pixel_width**2)
            arrs_data = np.append(arrs_data, images.numpy(),axis=0)
            arrs_label = np.append(arrs_label, labels.numpy(),axis=0)
        for batch_idx,(images, labels) in enumerate(self.data_loader['test']): # mnist测试集
            shape0 = images.shape[0]
            images = images.view(shape0, self.pixel_width ** 2)
            arrs_data_test = np.append(arrs_data_test, images.numpy(), axis=0)
            arrs_label_test = np.append(arrs_label_test, labels.numpy(), axis=0)
        # 先保存测试集，这个不需要增添数据
        np.save('./datasets/ARRAY/data_test.npy',arrs_data_test)
        np.save('./datasets/ARRAY/label_test.npy',arrs_label_test)
        print('测试集保存成功')
        # 再保存训练集，这个需要增添数据
        train_data_4 = np.load('./datasets/ARRAY/data_4.npy')
        train_label_4 = np.load('./datasets/ARRAY/label_4.npy')
        train_data_3 = np.load('./datasets/ARRAY/data_3.npy')
        train_label_3 = np.load('./datasets/ARRAY/label_3.npy')
        arrs_data = np.append(arrs_data, train_data_4,axis=0)
        arrs_data = np.append(arrs_data, train_data_3,axis=0)
        arrs_label = np.append(arrs_label, train_label_4, axis=0)
        arrs_label = np.append(arrs_label, train_label_3, axis=0)
        print('训练集merge成功')
        np.save('./datasets/ARRAY/data.npy', arrs_data)
        np.save('./datasets/ARRAY/label.npy',arrs_label)
        print('训练集保存成功')

    def save_checkpoint(self, filename='ckpt.tar'):# 保存checkpoint
        model_states = {
            'net':self.net.state_dict(),# net和Adam都有state_dict()函数
            }
        optim_states = {
            'optim':self.optim.state_dict(),
            }
        states = {
            'iter':self.global_iter,
            'epoch':self.global_epoch,
            'history':self.history,
            'args':self.args,
            'model_states':model_states,#把字典也存储进去
            'optim_states':optim_states,
            }

        file_path = self.ckpt_dir / filename
        torch.save(states, file_path.open('wb+'))
        # print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))
        logger.info("saved checkpoint'{}'(iter{})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])

            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else: raise('mode error. It should be either train or eval')

    def scale(self, image):
        return image.mul(2).add(-1)

    def unscale(self, image):
        return image.add(1).mul(0.5)

    def summary_flush(self, silent=True):
        rm_dir(self.summary_dir, silent)

    def checkpoint_flush(self, silent=True):
        rm_dir(self.ckpt_dir, silent)
