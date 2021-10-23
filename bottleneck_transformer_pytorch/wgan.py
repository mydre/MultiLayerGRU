import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import visdom
import random
import torch.autograd as autograd

hidden_dim = 400
batch_size = 512
# viz = visdom.Visdom()


class Generator(nn.Module):
    def __init__(self,data_dim,hidden_dim):
        super(Generator,self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            # z: [b,2(这个是z变量，隐藏变量，可以任意设定的)] => [b,2(为了画出真实的分布，这里指定为2)]
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim,self.data_dim),
        )

    def forward(self,z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(Discriminator,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(), # 判别器的输出是一个概率,当前x输入属于真实分布的概率,来自于生成器生成的
        )

    def forward(self, x): # 以原始的数据作为输入的材料
        output = self.net(x)
        return output.view(-1) #一维的向量


def data_generator():
    """
    8-gaussian mixture models
    :return:
    """
    scale = 2.
    centers = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x,y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02 # 随机生成两个数
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414 # 每次yield一个batch
        yield dataset # 使用yield实现一个无线的数据生成器

def gradient_penalty(D,xr,xf):
    """
    :param D:
    :param xr: [b,2]
    :param xf: [b,2]
    :return:
    """

    t = torch.rand(batch_size,1).cuda()
    t = t.expand_as(xr) # expand_as把一个向量扩展成另一个向量的维度
    mid = t * xr + (1-t) * xf
    # set it requires gradient
    mid.requires_grad()
    pred = D(mid)

    # 二次求导时用到create_graph,retain_graph
    grads = autograd.grad(outputs=pred,inputs=mid, grad_outputs=torch.ones_like(pred),
                          create_graph=True,retain_graph=True,only_inputs=True)[0]  # 对x冒求导
    # 先是求出二范数，然后减去1的平方，然后求均值
    gp = torch.pow((grads.norm(2, dim=1)-1),2).mean()
    return gp

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    data_iter = data_generator()

    G = Generator().cuda()
    D = Discriminator().cuda()
    print(G)
    print(D)
    optim_G = optim.Adam(G.parameters(),lr=5e-4,betas=(0.5,0.9))
    optim_D = optim.Adam(D.parameters(),lr=5e-4,betas=(0.5,0.9))

    for epoch in range(50000):
        # 1. train Discriminator firstly
        for _ in range(5):
            # 1.1 train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            predr = D(xr)
            # max predr, min lossr
            lossr = -predr.mean()
            # 1.2 train on fake data
            # [b,]
            z = torch.randn(batch_size,2).cuda()
            # 这里优化Descriminator，所有Genenator的梯度是不需要更新的
            xf = G(z).detach() # 类似于 tf.stop_gradient()
            predf = D(xf)
            lossf = predf.mean()

            # 1.3 gradient penalty
            gp = gradient_penalty(D,xr,xf.detach()) # 由于不需要对G求导，所以对xf进行detach
            # aggregate all
            loss_D = lossr + lossf + 0.2 * gp

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batch_size,2).cuda()
        xf = G(z)
        predf = D(xf)
        # max predf.mean()
        loss_G = -predf.mean()

        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            print(loss_D.item(),loss_G.item())


if __name__ == '__main__':
    main()