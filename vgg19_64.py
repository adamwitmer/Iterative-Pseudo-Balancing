import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import pdb
from torchvision.transforms.functional import center_crop
import math


class Vgg19(nn.Module):

    def __init__(self, num_classes=4, projection_dim=128, init_weights=True):
        super(Vgg19, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            # 8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            # 9
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # 15
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # 16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
        )

        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, projection_dim)
        )

        self.classifier = nn.Sequential(
            # 17
            nn.Linear(projection_dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            # 18
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            # 19
            nn.Linear(128, num_classes)
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(num_classes, 2),
        #     )
        if init_weights:
            self._initialize_weights()

    def forward(self, x): # , x_i, x_j):  # x_i, x_j, x):
        x = self.features(x)
        # h_i = self.features(x_i)
        # h_j = self.features(x_j)
        #~ print(x.size())
        h_x = x.view(x.size()[0], -1)
        # h_i = h_i.view(h_i.size()[0], -1)
        # h_j = h_j.view(h_j.size()[0], -1)

        z_x = self.projection(h_x)
        # z_i = self.projection(h_i)
        # z_j = self.projection(h_j)

        class_out = self.classifier(z_x)
        # out_x_i = self.classifier(z_i)
        # out_x_j = self.classifier(z_j)
        # y = self.linear(x)
        return class_out, z_x # h_i, h_j, z_i, z_j, z_x, out_x_i, out_x_j, class_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg19(**kwargs):
    model = Vgg19(**kwargs)
    return model

"""previous version of vgg 6.5MM params"""
# class Vgg19_multiscale(nn.Module):

#     def __init__(self, num_classes, projection_dim, init_weights=True):
#         super(Vgg19_multiscale, self).__init__()
#         self.features_256 = nn.Sequential(
#             # 1
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=False),
#             # 2
#             # nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
#             # 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction
#             # 4
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
#             # 5
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction

#             # 6
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=False),
#             # 7
#             # nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=False),
#             # # 8
#             # nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # moved to here from vvv

#             # these layers converted from 512 to 256
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # moved from here to ^^^
#             # # 10
#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),
#             # # 11
#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),
#             # # 
#             # this layer converted from 512 to 256
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8
#             # # 13
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # # 14
#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),
#             # # # 15
#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),
#             # # 16
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2, 8 --> 4
            
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),

#             # # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # # nn.BatchNorm2d(512),
#             # # # nn.ReLU(inplace=False),

#             # nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=False),
            
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2

#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),

#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),

#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),

#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),

#             # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2, 8 --> 4
#             )
#         self.features_128 = nn.Sequential(
#             # 1
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=False),
#             # 2
#             # nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
#             # 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction

#             # 4
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
#             # 5
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction

#             # 6
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 

#             # 7
#             # nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=False),
#             # # 8
#             # nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=False),
#             # 9
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 
#             # # 10
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # # 11
#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),
#             # # # 12
#             # nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=False),
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8
#             # 13
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # # # 14
#             # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # # nn.BatchNorm2d(512),
#             # # nn.ReLU(inplace=False),
#             # # # 15
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # # 16
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2, 8 --> 4
            
#             # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=False),
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
#         )

#         # self.features_64 = nn.Sequential(
#         #     # 1
#         #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(inplace=False),
#         #     # 2
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
#         #     # 3
#         #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(inplace=False),
#         #     # 4
#         #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
#         #     # 5
#         #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 6
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 7
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 8
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 9
#         #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 
#         #     # 10
#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     # 11
#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     # 12
#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8

#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
#         # )

#         # self.features_32 = nn.Sequential(
#         #     # 1
#         #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(inplace=False),
#         #     # 2
#         #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
#         #     # 3
#         #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(inplace=False),
#         #     # 4
#         #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
#         #     # 5
#         #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 6
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 7
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 8
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=False),
#         #     # 9
#         #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 
#         #     # 10
#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     # 11
#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     # 12
#         #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=False),
#         #     nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8

#         #     # nn.Conv2d(512, 512, kernel_size=3, padding=1),
#         #     # nn.BatchNorm2d(512),
#         #     # nn.ReLU(inplace=False),
#         #     # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
#         # )

#         self.projection = nn.Sequential(
#             nn.BatchNorm1d(512*2),
#             nn.Linear(512*2, 256),
#             nn.ReLU(inplace=False),
#             # nn.Linear(4096, 2048),
#             # nn.ReLU(inplace=False),
#             # nn.Linear(2048, 512),
#             # nn.ReLU(inplace=True),
#             nn.Linear(256, projection_dim)
#         )

#         self.classifier = nn.Sequential(
#             # 17
#             nn.Linear(projection_dim, 128),
#             nn.ReLU(inplace=False),
#             nn.Dropout(),
#             # 18
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=False),
#             nn.Dropout(),
#             # 19
#             nn.Linear(128, num_classes)
#         )

#         self.scale = nn.Linear(2048, 512)

#         # self.linear = nn.Sequential(
#         #     nn.Linear(num_classes, 2),
#         #     )
#         if init_weights:
#             self._initialize_weights()

#     def forward(self,input_large, input_small):  # x_i, x_j, x): x
#         x = input_large
#         y = input_small
#         # x, y, z = input.chunk(3, 1)
#         # x, y = input.chunk(2, 1)
#         # y = center_crop(y, input.shape[-1]//2)
#         # z = center_crop(y, 64)
#         # y = center_crop(y, 64)
#         # z = center_crop(y, 32)
#         # x, y = z
#         x = self.features_256(x)
#         y = self.features_128(y)
#         # z = self.features_64(z)
#         # z = self.features_32(z)
#         # h_i = self.features(x_i)
#         # h_j = self.features(x_j)
#         #~ print(x.size())
#         h_x = x.view(x.size()[0], -1)
#         h_y = y.view(y.size()[0], -1)
#         # h_z = z.view(z.size()[0], -1)
#         # h_x = self.scale(h_x)
#         # concatenate feature vectors before projection head 
#         h_z = torch.cat((h_x, h_y), 1) # h_z --> 6144
#         # h_i = h_i.view(h_i.size()[0], -1)
#         # h_j = h_j.view(h_j.size()[0], -1)

#         z_x = self.projection(h_z)
#         # z_i = self.projection(h_i)
#         # z_j = self.projection(h_j)

#         class_out = self.classifier(z_x)
#         # out_x_i = self.classifier(z_i)
#         # out_x_j = self.classifier(z_j)
#         # y = self.linear(x)
#         return class_out, z_x  # h_i, h_j, z_i, z_j, z_x, out_x_i, out_x_j, class_out

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()

"""multiscale network with 23M params"""
class Vgg19_multiscale(nn.Module):

    def __init__(self, num_classes, projection_dim, init_weights=True):
        super(Vgg19_multiscale, self).__init__()
        self.features_256 = nn.Sequential(
            # 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            # 2
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction

            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            # 7
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=False),
            # # 8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # moved to here from vvv

            # these layers converted from 512 to 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # moved from here to ^^^
            # # 10
            # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(512),
            # # nn.ReLU(inplace=False),
            # # 11
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # # 
            # this layer converted from 512 to 256
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8
            # # 13
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),
            # # 14 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # # # 15
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # # 16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2, 8 --> 4
            
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),

            # # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # # # nn.BatchNorm2d(512),
            # # # nn.ReLU(inplace=False),

            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=False),
            
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2

            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),

            # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(512),
            # # nn.ReLU(inplace=False),

            # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(512),
            # # nn.ReLU(inplace=False),

            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),

            # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2, 8 --> 4
            )
        self.features_128 = nn.Sequential(
            # 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            # 2
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction

            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # inserted for dimensionality reduction

            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 

            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            # # 8
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=False),
            # 9
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 
            # # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # # 11
            # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(512),
            # # nn.ReLU(inplace=False),
            # # # 12
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8
            # 13
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),
            # # # 14
            # # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(512),
            # # nn.ReLU(inplace=False),
            # # # 15
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            # # 16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2, 8 --> 4
            
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
        )

        # self.features_64 = nn.Sequential(
        #     # 1
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False),
        #     # 2
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
        #     # 3
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=False),
        #     # 4
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
        #     # 5
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 6
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 7
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 8
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 9
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 
        #     # 10
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     # 11
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     # 12
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8

        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
        # )

        # self.features_32 = nn.Sequential(
        #     # 1
        #     nn.Conv2d(1, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False),
        #     # 2
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 64 --> 32, 128 --> 64
        #     # 3
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=False),
        #     # 4
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32 --> 16, 64 --> 32
        #     # 5
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 6
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 7
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 8
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=False),
        #     # 9
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16 --> 8, 32 --> 16 
        #     # 10
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     # 11
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     # 12
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=False),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 8 --> 4, 16 --> 8

        #     # nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512),
        #     # nn.ReLU(inplace=False),
        #     # nn.MaxPool2d(kernel_size=2, stride=2),  # 4 --> 2
        # )

        self.projection = nn.Sequential(
            nn.BatchNorm1d(512*2),
            nn.Linear(512*2, 256),
            nn.ReLU(inplace=False),
            # nn.Linear(4096, 2048),
            # nn.ReLU(inplace=False),
            # nn.Linear(2048, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim)
        )

        self.classifier = nn.Sequential(
            # 17
            nn.Linear(projection_dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            # 18
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            # 19
            nn.Linear(128, num_classes)
        )

        self.scale = nn.Linear(2048, 512)

        # self.linear = nn.Sequential(
        #     nn.Linear(num_classes, 2),
        #     )
        if init_weights:
            self._initialize_weights()

    def forward(self, input_large, input_small):  # x_i, x_j, x): x
        x = input_large
        y = input_small
        # x, y, z = input.chunk(3, 1)
        # x, y = input.chunk(2, 1)
        # y = center_crop(y, input.shape[-1]//2)
        # z = center_crop(y, 64)
        # y = center_crop(y, 64)
        # z = center_crop(y, 32)
        # x, y = z
        x = self.features_256(x)
        y = self.features_128(y)
        # z = self.features_64(z)
        # z = self.features_32(z)
        # h_i = self.features(x_i)
        # h_j = self.features(x_j)
        #~ print(x.size())
        h_x = x.view(x.size()[0], -1)
        h_y = y.view(y.size()[0], -1)
        # h_z = z.view(z.size()[0], -1)
        # h_x = self.scale(h_x)
        # concatenate feature vectors before projection head 
        h_z = torch.cat((h_x, h_y), 1) # h_z --> 6144
        # h_i = h_i.view(h_i.size()[0], -1)
        # h_j = h_j.view(h_j.size()[0], -1)

        z_x = self.projection(h_z)
        # z_i = self.projection(h_i)
        # z_j = self.projection(h_j)

        class_out = self.classifier(z_x)
        # out_x_i = self.classifier(z_i)
        # out_x_j = self.classifier(z_j)
        # y = self.linear(x)
        return class_out, z_x  # h_i, h_j, z_i, z_j, z_x, out_x_i, out_x_j, class_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg19(**kwargs):
    model = Vgg19_multiscale(**kwargs)
    return model
#~ a = torch.FloatTensor(1,1,128,128)
#~ net  = Vgg19()
#~ out = net(a)
