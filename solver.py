"""solver.py"""
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from datasets.datasets import return_data2
from utils.utils import rm_dir, cuda, where
import pdb
import numpy as np
from bottleneck_transformer_pytorch.bottleneck_transformer_pytorch import BottleStack
from bottleneck_transformer_pytorch.bottleneck_transformer_pytorch import MyGruNet
from torch import nn
from loguru import logger
import datetime


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
        # self.data_loader = return_data(args)
        self.data_loader = return_data2(args)
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
        self.net = cuda(BottleStack(dim=1,fmap_size=self.pixel_width,dim_out=8,proj_factor=4,downsample=True,heads=4,dim_head=8,rel_pos_emb=False, y_dim=self.y_dim), self.cuda)
        # self.net = cuda(MyGruNet(y_dim = self.y_dim,pixel_width=self.pixel_width),self.cuda)

        # Optimizers
        # self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}],betas=(0.5, 0.999))
        self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}])

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
        print(len(self.data_loader['train']))
        for e in range(1,self.epoch+1): # e从1开始算起

            self.global_epoch += 1
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):

                self.global_iter += 1

                x = Variable(cuda(images, self.cuda)) # torch.Size([256, 1, 43, 43])
                y = Variable(cuda(labels, self.cuda)) # torch.Size([256])
                # x = torch.autograd.Variable(cuda(images, self.cuda))
                # y = torch.autograd.Variable(cuda(labels, self.cuda))
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
                # out = self._arcface(logit,y)
                # cost = F.cross_entropy(logit, y) # cost也是一个Variable,计算出的cost是一个损失

                loss_ = F.cross_entropy(soft_logit, y) # cost也是一个Variable,计算出的cost是一个损失
                # cost = F.cross_entropy(out, y) # cost也是一个Variable,计算出的cost是一个损失

                # lds = self.at_loss(x,y)
                # loss_lds = self.at_loss(x,y)
                # cost = loss_ + loss_lds + loss
                # cost = loss
                # cost = loss_ + loss_lds
                # cost = loss_ + loss
                # cost = loss_ + loss + loss_lds
                # cost = loss_lds
                cost = loss_
                self.optim.zero_grad()
                cost.backward()  # 反向传播之后可以进行参数的更新,反向传播的是那个网络，更新的就是哪个网络的参数
                self.optim.step()
                if batch_idx % 500 == 0:
                    if self.print_:
                        # print()
                        # print(self.env_name)
                        logger.info('[{:03d}:{:05d}]'.format(self.global_epoch, batch_idx) + '  ' + 'acc:{:.3f} loss:{:.3f}'.format(correct, cost.data))
                        # print('[{:03d}:{:03d}]'.format(self.global_epoch, batch_idx))
                        # print('acc:{:.3f} loss:{:.3f}'.format(correct, cost.data))

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

        data_loader = self.data_loader['test']
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
            # print()
            # print('[{:03d}]\nTEST RESULT'.format(self.global_epoch))
            # print('ACC:{:.4f}'.format(accuracy))
            # print('*TOP* ACC:{:.4f} at e:{:03d}, COST:{:.4f}'.format(accuracy, self.global_epoch,cost))
            # print()
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

    def gene_label(self):
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


    def gene_probability(self):
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


    def generate(self, num_sample=100, target=-1, epsilon=0.03, alpha=2/255, iteration=1):

        self.set_mode('eval')

        for e in range(1):#假设有5个epoch
            self.global_epoch += 1
            for batch_idx,(x_true,y_true) in enumerate(self.data_loader['train']):# x_true: [torch.FloatTensor of size 100x1x28x28]
                self.global_iter += 1
                y_target = None
        #x_true, y_true = self.sample_data(num_sample)
        #if isinstance(target, int) and (target in range(self.y_dim)): # range(10):[0,1,2,...,9]
        #    '''
        #    (Pdb) torch.LongTensor(3).fill_(10)
        #    10
        #    10
        #    10
        #    [torch.LongTensor of size 3]
        #    '''
        #    y_target = torch.LongTensor(y_true.size()).fill_(target)
        #else:
        #    y_target = None
        ## y_target可能为None，也可能不为None
                values = self.FGSM(x_true, y_true, y_target, epsilon, alpha, iteration)
                # accuracy, cost, accuracy_adv, cost_adv = values
                correct, cost = values
                if batch_idx % 100 == 0:
                    if self.print_:
                        print()
                        print(self.env_name)
                        print('[{:03d}:{:03d}]'.format(self.global_epoch,batch_idx))
                        print('acc:{:.3f} loss:{:.3f}'.format(correct,cost.data[0]))

                    if self.tensorboard:
                        self.tf.add_scalars(main_tag='performance/acc',
                                            tag_scalar_dict={'train': correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={'train': 1 - correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={'train': cost.data[0]},
                                            global_step=self.global_iter)
            self.test()
        if self.tensorboard:
            self.tf.add_scalars(main_tag='performance/best/acc',
                                tag_scalar_dict={'test': self.history['acc']},
                                global_step=self.history['iter'])
        print(" [*] Generating Finished!")

    def sample_data(self, num_sample=100):

        total = len(self.data_loader['test'].dataset)
        '''
        >>> torch.FloatTensor(10).uniform_(1,5).long()
        tensor([1, 1, 2, 1, 3, 4, 1, 1, 4, 1])
        '''
        seed = torch.FloatTensor(num_sample).uniform_(1, total).long() # seed.shape: torch.Size([100]),等价于产生100个1到total(10000)之间的数值，如产生100个数，每个数的取值范围都是[1,10000]
        x = torch.from_numpy(self.data_loader['test'].dataset.train_set[seed]) #  self.data_loader['test'].dataset.test_data[0:2], 2x28x28
        #pdb.set_trace()
        # [100x1x28x28],
        # x = self.scale(x.float().unsqueeze(1).div(255)) #  x.float().unsqueeze(1).div(255), x.float().unsqueeze(1).div(255).mul(2).add(-1)
        x = self.scale(x.float().div(255)) #  x.float().unsqueeze(1).div(255), x.float().unsqueeze(1).div(255).mul(2).add(-1)
        y = torch.from_numpy(self.data_loader['test'].dataset.train_labels[seed]).long()# Tensor的类型设置为long
        return x, y

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
