import math
import pdb

import torch
from torch import nn, einsum

from einops import rearrange

from .mish_activ import Mish
# translated from tensorflow code
# https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

# positional embedding helpers

# 如果是元组，则直接返回。如果不是元组，则返回元组，并且元组内部有两个元素
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


'''
b: 2 h: 256 l: 64 x.shape: torch.Size([2, 256, 64, 127])
flat_x_padded.shape: torch.Size([2, 256, 8255])
b: 2 h: 256 l: 64 x.shape: torch.Size([2, 256, 64, 127])
flat_x_padded.shape: torch.Size([2, 256, 8255])
b: 2 h: 128 l: 32 x.shape: torch.Size([2, 128, 32, 63])
flat_x_padded.shape: torch.Size([2, 128, 2079])
b: 2 h: 128 l: 32 x.shape: torch.Size([2, 128, 32, 63])
flat_x_padded.shape: torch.Size([2, 128, 2079])
b: 2 h: 128 l: 32 x.shape: torch.Size([2, 128, 32, 63])
flat_x_padded.shape: torch.Size([2, 128, 2079])
b: 2 h: 128 l: 32 x.shape: torch.Size([2, 128, 32, 63])
flat_x_padded.shape: torch.Size([2, 128, 2079])
'''
def rel_to_abs(x):  # 此时x的shape是：b,（h x） y r
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    print('b:',b,'h:',h,'l:',l,'x.shape:',x.shape)
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3) # 从dim=3开始拼接，说明dim=0,1,2的时候得到的都是相同的,前面的结构都是相同的，并且后面的结构也是相同的
    flat_x = rearrange(x, 'b h l c -> b h (l c)') # 最后两个维度拼接在一块
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)  # 再次进行拼接
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)  # 然后改变shape
    final_x = final_x[:, :, :l, (l-1):]  # 选取特定的而元素
    return final_x

def relative_logits_1d(q, rel_k):  # q(2,4,x,y,d)==(batch,head,height,width,d),   (2*width -1,128)
    b, heads, h, w, dim = q.shape  # 这个时候q.shape总共有5个维度
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')  # b (h x) y r，所以这个时候4个维度
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# positional embeddings

class AbsPosEmb(nn.Module):  # 绝对位置编码
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        # nn.Parameter的作用是将一个Tensor类型转换为一个可训练的类型parameter并把这个parameter绑定到module里面
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)
        '''
        print('AbsPosEmb--------------------')
        print('self.height.shape:',self.height.shape,'self.width.shape:',self.width.shape)
        总共调用了三次：
        AbsPosEmb--------------------
        self.height.shape: torch.Size([64, 128]) self.width.shape: torch.Size([64, 128])
        
        AbsPosEmb--------------------
        self.height.shape: torch.Size([32, 128]) self.width.shape: torch.Size([32, 128])
        
        AbsPosEmb--------------------
        self.height.shape: torch.Size([32, 128]) self.width.shape: torch.Size([32, 128])

        '''
    def forward(self, q):  # q是将要被编码的参数，含有batch，head, i(seq_len), d(嵌入的维度)
        #emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        one = rearrange(self.height,'h d -> h () d')  # 这里的()相当于是扩展了一个维度，为1
        two = rearrange(self.width, 'w d -> () w d')
        '''
        print('one.shape:', one.shape)
        print('two.shape:', two.shape)
        one.shape: torch.Size([64, 1, 128])
        two.shape: torch.Size([1, 64, 128])
        one.shape: torch.Size([32, 1, 128])
        two.shape: torch.Size([1, 32, 128])
        one.shape: torch.Size([32, 1, 128])
        two.shape: torch.Size([1, 32, 128])
        '''
        emb = one + two  # 这里在进行相加的时候直接把(1)这个维度给忽略了
        # print('__emb.shape:',emb.shape)
        emb = rearrange(emb, ' h w d -> (h w) d')
        # print('___emb.shape:',emb.shape)
        '''
        __emb.shape: torch.Size([64, 64, 128])
        ___emb.shape: torch.Size([4096, 128])
        __emb.shape: torch.Size([32, 32, 128])
        ___emb.shape: torch.Size([1024, 128])
        __emb.shape: torch.Size([32, 32, 128])
        ___emb.shape: torch.Size([1024, 128])
        '''
        # 这里的j d中的j表示seq_len,d表示Embedding维度
        '''
        print('q.shape:',q.shape)
        q.shape: torch.Size([2, 4, 4096, 128])
        q.shape: torch.Size([2, 4, 1024, 128])
        q.shape: torch.Size([2, 4, 1024, 128])
        '''
        logits = einsum('b h i d, j d -> b h i j', q, emb)  # 长度和宽度结合在一起，就变成了emb
        '''
        print('logits.shape:',logits.shape)
        logits.shape: torch.Size([2, 4, 4096, 4096])
        logits.shape: torch.Size([2, 4, 1024, 1024])
        logits.shape: torch.Size([2, 4, 1024, 1024])
        '''
        return logits

class RelPosEmb(nn.Module):  # 相对位置编码
    def __init__(
        self,
        fmap_size,
        dim_head  # dim_head是128？
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)  # 这里不是height，而是height * 2 - 1
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)  # 这里不是width，而是width * 2 - 1

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w) # 原先的是x*y,现在把它们拆开了
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        # print('-------------Attention-------------')
        print('self.heads:',self.heads,'inner_dim:',inner_dim,'self.scale:',self.scale,'dim:',dim,'fmap_size:',fmap_size)
        '''
        dim_head是128，dim是512
        self.heads: 4 inner_dim: 512 self.scale: 0.08838834764831845 dim: 512 fmap_size: (64, 64)
        self.heads: 4 inner_dim: 512 self.scale: 0.08838834764831845 dim: 512 fmap_size: (32, 32)
        self.heads: 4 inner_dim: 512 self.scale: 0.08838834764831845 dim: 512 fmap_size: (32, 32)
        '''
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False) # 512 * 3 = 1536

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb  # 使用绝对位置编码还是相对位置编码,positive,relative
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        #  fmap.shape:   torch.Size([2, 512, 32, 32])
        # print('fmap.shape:',fmap.shape)
        '''
        fmap.shape: torch.Size([2, 512, 64, 64])
        fmap.shape: torch.Size([2, 512, 32, 32])
        fmap.shape: torch.Size([2, 512, 32, 32])
        '''
        # print(self.to_qkv(fmap).shape)  torch.Size([2, 1536, 32, 32])
        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1) # chunk表示按照某个维度进行数据的平均拆分
        # print(q.shape,k.shape,v.shape)  #torch.Size([2, 512, 32, 32]) torch.Size([2, 512, 32, 32]) torch.Size([2, 512, 32, 32])
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))  # 这里可以把h理解为4，把d理解为128(嵌入的维度)
        # torch.Size([2, 4, 4096, 128]) torch.Size([2, 4, 4096, 128]) torch.Size([2, 4, 4096, 128])  # 其中的128相当于是dk，相当于是嵌入的维度，4096相当于是len_q,后者len_k,4096相当于是(1024*4)
        # print(q.shape,k.shape,v.shape)
        q = q * self.scale  # 相当于提前除以sqrt(balabala)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # q和k进行相乘，最后的结果是i,j。只是最后两个维度进行相乘
        sim = sim + self.pos_emb(q)  # 相当于attention再加上位置信息

        attn = sim.softmax(dim = -1)  # 进行softmax操作之后得到最终的注意力向量

        '''
        out.shape: torch.Size([2, 4, 4096, 128])
        out.shape: torch.Size([2, 512, 64, 64])
        out.shape: torch.Size([2, 4, 1024, 128])
        out.shape: torch.Size([2, 512, 32, 32])
        out.shape: torch.Size([2, 4, 1024, 128])
        out.shape: torch.Size([2, 512, 32, 32])
        '''
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 注意力向量和v向量进行相乘可以得到最终的context向量,可以得到最终的注意力向量
        #print('out.shape:',out.shape)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)  # 让head(4)和嵌入的维度进行相乘(128),最后得到(2,512,64,64)
        #print('out.shape:',out.shape)
        # 最后out的shape和输入的fmap的shape是一样的，这也体现了自注意力模型的处理过程
        return out

class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入通道的大小
        fmap_size,  # 特征的大小
        dim_out,  # 输出通道的大小
        proj_factor,  # 连接因子
        downsample,  # 下采样标志
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        '''
        例如总共有三层block
        第一个block的输入通道数为512，输出通道数为2048。下采样的时候，图片尺寸变小(总面积变为原来的1/4)，通道数增加到原来的8倍
        第二个block的时候，不仅图像的尺寸没有发生改变，而且通道数也没有发生改变
        第三个block和第二个block一样
        dim fmap_size dim_out proj_factor
        256 (64, 64) 2048 4
        --------------------------
        2048 (32, 32) 2048 4
        --------------------------
        2048 (32, 32) 2048 4
        -------------------BottleStack-------
        '''

        # shortcut

        if dim != dim_out or downsample:
            # kernel_size, stride, padding = (3, 2, 1)
            kernel_size, stride, padding = (3, 2, 0)

            self.shortcut = nn.Sequential(  # 卷积，BN，激活
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()# 相当于输入和输出是一样的，便于表示加深网络的深度

        # contraction and expansion

        attn_dim_in = dim_out // proj_factor  # 相当于2048/4 == 512
        attn_dim_out = heads * dim_head  # 4 * 128 = 512

        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias = False),  # 输入通道数目是256(这个是从输入数据中得来的)，输出512通道数目.(256, 512)
            nn.BatchNorm2d(attn_dim_in),  # 512
            activation,
            Attention(
                dim = attn_dim_in,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            # 这里经过一个平均池化，来缩小了图像的尺寸。没有这一行的话，结果是(64,64)，有这一行的话，结果是(32,32).在长度方向和宽度方向上进行了池化操作
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),  # attn_dim_out在这个地方用到
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )
        # 在init函数中构造出各个结构，如shotcut结构，和net结构

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    '''
    x.shape: torch.Size([2, 256, 64, 64])
    shortcut.shape: torch.Size([2, 2048, 32, 32])  # 经过shortcut网络之后，通道数增加了
    net之后：x.shape: torch.Size([2, 2048, 32, 32])  # 记过net网络之后，通道的数目变大了
    
    x.shape: torch.Size([2, 2048, 32, 32])
    shortcut.shape: torch.Size([2, 2048, 32, 32])
    net之后：x.shape: torch.Size([2, 2048, 32, 32])
    
    x.shape: torch.Size([2, 2048, 32, 32])
    shortcut.shape: torch.Size([2, 2048, 32, 32])
    net之后：x.shape: torch.Size([2, 2048, 32, 32])
    '''
    def forward(self, x):
        #>>>>>>>>>>>>>>>print('x.shape:',x.shape)
        shortcut = self.shortcut(x)
        #>>>>>>>>>>>>>>>>>>>print('shortcut.shape:',shortcut.shape)
        x = self.net(x)
        #>>>>>>>>>>>>>>>>>>print('net之后：x.shape:',x.shape)
        x = x + shortcut
        return self.activation(x)

# main bottle stack

class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        fmap_size = pair(fmap_size)
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []  # 通过for循环讲多个模型添加到这个list当中，因为添加的是三个相同的block

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)  # dim的取值有两种，一种是
            layer_downsample = is_first and downsample  # 下采样

            fmap_divisor = (2 if downsample and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))  # (64,64)->(64,64),(64,64)->(32,32)

            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,  # 在这里对fmap_size进行改变
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self._reshapeNet = ReshapeNet()
        self.linear1 = nn.Linear(8 * 9 * 9, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 5)
        self.net = nn.Sequential(*layers,self._reshapeNet,self.linear1,self.linear2,self.linear4,self.linear3)

    def forward(self, x):
        # print('x.shape:',x.shape)  # 2,256,64,64
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'  # 这个是assert的提示信息，如果报错了可以很方便查看提示信息
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        # return self.net(x)  # 这个net中有多个BottleBlock网络块
        return self.net(x)  # 这个net中有多个BottleBlock网络块
        # x = x.view(x.shape[0],-1)
        # x = self.net2(x)


class ReshapeNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.shape[0], -1)