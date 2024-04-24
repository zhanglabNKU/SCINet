import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

import  torch
import  torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import models
import torchvision.models.vgg
import  torch
import  torch.nn as nn
import torchvision.models as tv_models
from torch.nn import init
import torch.nn.functional as F
from functools import partial

from torchvision import models
import torchvision.models.vgg
##使用VGG16特征提取层+新的全连接层组成新的网络


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim,activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        # self.activation == nn.ReLU()
        # self.conv =BasicConv(4096, 2048, 1, stride=1, relu=False)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()


        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.sigmoid(energy)  # BX (N) X (N)
        # attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Self_Attn2(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn2, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        # self.activation == nn.ReLU()
        # self.conv =BasicConv(4096, 2048, 1, stride=1, relu=False)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        # self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        # self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        #
        # self.query_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        # self.key_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        # self.value_conv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(in_channels=2*in_dim, out_channels=in_dim, kernel_size=1)

        self.sigmoid = nn.Sigmoid()  ###可能更好
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1,x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize1, C1, width1, height1 = x1.size()



        proj_query1 = self.query_conv2(x2).view(m_batchsize1, -1, width1 * height1).permute(0, 2, 1)  # B X CX(N)

        proj_key = self.key_conv(x1).view(m_batchsize1, -1, width1 * height1)  # B X C x (*W*H)
        energy1 = torch.bmm(proj_query1, proj_key)  # transpose check
        # attention1 = self.sigmoid(energy1)  # BX (N) X (N)
        attention1 = self.softmax(energy1)  # BX (N) X (N)

        proj_value = self.value_conv(x1).view(m_batchsize1, -1, width1 * height1)  # B X C X N
        out1 = torch.bmm(proj_value, attention1.permute(0, 2, 1))


        out1 = out1.view(m_batchsize1, C1, width1, height1)

        out1 = self.gamma * out1 + x1
        return out1


class MultiAtrous(nn.Module):
    def __init__(self, in_channel, out_channel, size, dilation_rates=[3, 6, 9]):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel/4),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(out_channel/4), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode='bilinear')
        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat

class MultiAtrous2(nn.Module):
    def __init__(self, in_channel, out_channel, dilation_rates=[3, 6, 9]):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.Conv2d(in_channel, int(out_channel), kernel_size=1),

        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_planes)   #eps=1e-5, momentum=0.01
        # self.relu = nn.ReLU() if relu else None

        init.kaiming_normal_(self.conv.weight)
        init.constant_(self.bn.weight,1)
        init.constant_(self.bn.bias,0)
        # init.kaiming_normal_(self.classifer.weight)  #最好用这个吧
        # init.constant_(self.classifer.bias,0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x






def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class TSNet(nn.Module):
    def __init__(self , model0,model1):
        super(TSNet, self).__init__()
        self.resnet_layer_0 = nn.Sequential(*list(model0.children())[:-2])
        self.resnet_layer_1 = nn.Sequential(*list(model1.children())[:-2])
        self.Linear_layer= nn.Linear(1024, 4)
    def forward_once_0(self,x):
        x = self.resnet_layer_0(x)
        x = x.view(x.size(0), -1)
        return x
    def forward_once_1(self,x):
        x = self.resnet_layer_1(x)
        x = x.view(x.size(0), -1)
        return x
    def forward(self, input0,input1):
        output0 = self.forward_once_0(input0)
        output1 = self.forward_once_1(input1)
        output=torch.cat((output0,output1),1)
        output_C=self.Linear_layer(output)
        output_C = F.dropout(output_C, p=0.5, training=self.training)
        return output_C



class DENetattention3(nn.Module):
    def __init__(self,model0,model1):
        super(DENetattention3, self).__init__()
        #self.adavgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.admaxpool = nn.MaxPool2d((7, 7)) #224
        #self.admaxpool = nn.MaxPool2d((14, 14)) #448
        # self.maxpool = nn.MaxPool2d((7, 7))
        # self.avgpool = nn.AvgPool2d((7, 7))
        #self.maxpool = nn.MaxPool2d((1, 1))
        #self.avgpool = nn.AvgPool2d((1, 1))
        #self.conve1=nn.Conv2d(4096, 2048, kernel_size=1, stride=1)  # 可能还得改 7 3 前面还是后面
        #self.conve2 = nn.Conv2d(4096, 1, kernel_size=3, stride=1)  # cq 1 7

       # self.spatial =BasicConv(4096, 2048, 1, stride=1)  #等于7很大
        # self.spatial2 =BasicConv(4096, 1, 1, stride=1,  relu=False)
        # self.spatial =BasicConv(4096, 2048, 1, stride=1,  relu=False)  #等于7很大
        # self.spatial2 =BasicConv(4096, 2048, 1, stride=1,  relu=False)

        # self.attention2 = Self_Attn2(2048, nn.ReLU())
        self.attention2 = Self_Attn2(1024, nn.ReLU())
        # self.MultiAtrous=MultiAtrous(2048,2048,1)
        # self.MultiAtrous2 = MultiAtrous2(2048, 2048)
        self.v0=model0
        self.v1=model1
        #self.res_layer_0 = nn.Sequential(*list(model0.children())[:-1])
        #self.res_layer_1 = nn.Sequential(*list(model1.children())[:-1])
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(p=0.5),  # 按照一定的比例将网络中的神经元丢弃，可以防止模型训练过度
        #     # nn.Linear(8192, 512),   #concat
        #     nn.Linear(4096, 512),  #sum
        #     nn.ReLU(),
        #     # nn.LeakyReLU(),
        #     # nn.Dropout(p=0.5),  # 按照一定的比例将网络中的神经元丢弃，可以防止模型训练过度
        #     nn.Linear(512, 4),  # 10为输出的类别
        # )#34 18

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),  # 按照一定的比例将网络中的神经元丢弃，可以防止模型训练过度
            # nn.Linear(8192, 512),   #concat
            nn.Linear(2048, 512),  # sum
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.5),  # 按照一定的比例将网络中的神经元丢弃，可以防止模型训练过度
            nn.Linear(512, 4),  # 10为输出的类别
        )

        # init.kaiming_normal_(self.classifer.weight)  #最好用这个吧
        # init.constant_(self.classifer.bias,0)
    def forward_once_0(self,x):

        #x = self.res_layer_0(x)
        x= self.v0(x)

        return x
    def forward_once_1(self,x):
        #x = self.res_layer_1(x)
        x = self.v1(x)

        return x
    def forward(self, input0,input1):
        #output0 = self.vit1(input0)   #2048
        #output1 = self.vit2(input1)
        #output0 =output0.unsqueeze(-1).unsqueeze(-1)
        #output1 = output1.unsqueeze(-1).unsqueeze(-1)
        output0 = self.forward_once_0(input0)  # 2048

        output1 = self.forward_once_1(input1)
        output0 = output0.unsqueeze(-1).unsqueeze(-1)
        output1 = output1.unsqueeze(-1).unsqueeze(-1)
#########################################################  金字塔
        # output0=self.MultiAtrous(output0)
        # output1=self.MultiAtrous(output1)
        # output0 = self.MultiAtrous2(output0)
        # output1 = self.MultiAtrous2(output1)
        ################################################


        # outputattention,attention= self.attention(outputcat)  #单独注意力 linear 4096
        outputattention0 = self.attention2(output0,output1)#双注意力 #2048
        ####################第一种直接加就完事了
        # # outputattention=outputattention0+outputattention1 #
        # # output = output0+output1  # 加起来
        # # output = torch.cat((output, outputattention), 1)  #4096
        # ##############concat路子  第一种降维
        # output= torch.cat((output0, output1), 1)  #拼起来linear 8192      可能加起来也更好  但还是算了吧
        # output = self.conve1(output)  # 2048
        # outputattention = torch.cat((outputattention0, outputattention1), 1)  # 拼起来linear 8192      可能加起来也更好  但还是算了吧
        # outputattention = self.conve1(outputattention)  # 2048
        output =torch.cat((output0, outputattention0), 1)  #8192
        #output=output1+output0

        #output = torch.cat((output0, outputattention0), 1)  # 拼起来linear 8192      可能加起来也更好  但还是算了吧
        # output = self.conve1(output)  # 2048
        # outputattention = torch.cat((output1, outputattention1), 1)  # 拼起来linear 8192      可能加起来也更好  但还是算了吧
        # outputattention = self.conve1(outputattention)  # 2048
        # output = torch.cat((output, outputattention), 1)  # 8192


        #############
        output = output.view(output.size(0), -1)
        # output = output0+output1   sum
        output_C=self.classifier(output)
        # nn.Dropout(p=0.5),
        # output_C = nn.dropout(output_C, p=0.5, training=self.training)

        # output = torch.flatten(output, start_dim=1)
        # output_C = self.classifier(output)


        return output_C

    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             # nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.constant_(m.bias, 0)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=512, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # (224,224)
        img_size = (img_size, img_size)
        # (16,16)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # (14,14)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 196
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        # print(x.shape)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # ??token?dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # print((attn @ v).shape)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            #x = self.head(x)
            x=x
            #print(x.shape)

        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


