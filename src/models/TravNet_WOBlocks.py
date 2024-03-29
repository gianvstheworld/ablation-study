# Descrição: Modelo TravNet (ResNet18 Unet)
# Importação de modelos tradicionais de redes neurais
from torchvision import models

# Importação de bibliotecas
import torch
import torch.nn as nn
import torch.nn.functional as F

class TravNet(nn.Module):
    def __init__(self, params) -> None:
        '''
        Construtor da classe
        :param params: Parâmetros de configuração do modelo
        '''
        super().__init__()
        model = models.resnet18(pretrained=params.pretrained)

        self.out_dim = (params.output_size[1], params.output_size[0])

        self.block1 = nn.Sequential(*(list(model.children())[:3]))
        # ORIGINAL
        self.block2 = nn.Sequential(model.maxpool, model.layer1)
        # MODIFICADO
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
        #               padding=3, stride=2, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     model.maxpool,
        #     model.layer1)
        self.block3 = model.layer2
        self.block4 = model.layer3
        # ORIGINAL
        self.block5 = model.layer4
        # MODIFICADO
        # self.block5 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=512,
        #               kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=512,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )

        self.block1_depth = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                      padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # ORIGINAL
        self.block2_depth = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # MODIFICADO
        # self.block2_depth = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
        #               padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))

        self.block3_depth = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        # ORIGINAL
        self.block4_depth = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        # MODIFICADO
        # self.block4_depth = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128,
        #               kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
        #               padding=1, bias=False),  # Adicionada camada de convolução
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True))

        # ORIGINAL
        self.block5_depth = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        # MODIFICADO
        # self.block5_depth = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256,
        #               kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=256, out_channels=512,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )

        # ORIGINAL
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=params.bottleneck_dim,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=params.bottleneck_dim,
        #               out_channels=256, kernel_size=3, padding=1, bias=False),
        #     nn.ReLU(inplace=True))
        # MODIFICADO
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=params.bottleneck_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=params.bottleneck_dim,
                      out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+256,
                               out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        # ORIGINAL
        # self.convTrans2 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=256+128,
        #                        out_channels=128, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=64,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        # MODIFICADO
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ORIGINAL
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128+64,
                               out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        # MODIFICADO
        # self.convTrans3 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=256, out_channels=64, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=32,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True)
        # )

        # ORIGINAL
        # self.convTrans4 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=192,  # Ajuste o número de canais de entrada
        #                        out_channels=128, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=64,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        # MODIFICADO
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=128,
                            kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64,
                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=params.output_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(params.output_channels),
            nn.Sigmoid())

        self.conv_adjust = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=1, stride=1, bias=False)

    def adjust_channel(self, x):
        return self.conv_adjust(x)

    def forward(self, rgb_img, depth_img) -> torch.Tensor:
        out1 = self.block1(rgb_img)
        out1_depth = self.block1_depth(depth_img)
        out1 = out1 + out1_depth

        # ORIGINAL
        out2 = self.block2(out1)
        out2_depth = self.block2_depth(out1_depth)
        # MODIFICADO
        # out2_depth = F.interpolate(
        #     out2_depth, size=out2.shape[2:], mode='bilinear', align_corners=False)
        # out2 = self.adjust_channel(out2)
        out2 = out2 + out2_depth

        out3 = self.block3(out2)
        out3_depth = self.block3_depth(out2_depth)
        out3 = out3 + out3_depth

        out4 = self.block4(out3)
        out4_depth = self.block4_depth(out3_depth)
        out4 = out4 + out4_depth

        # out5 = self.block5(out3)
        # out5_depth = self.block5_depth(out3_depth)
        # out5 = out5 + out5_depth

        x = self.bottleneck(out4)

        # x = torch.cat((x, out5), dim=1)
        # x = self.convTrans1(x)

        diffY = out4.size()[2] - x.size()[2]
        diffX = out4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX //
                  2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, out4), dim=1)
        x = self.convTrans2(x)

        diffY = out3.size()[2] - x.size()[2]
        diffX = out3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX //
                  2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, out3), dim=1)
        x = self.convTrans3(x)

        x = torch.cat((x, out2), dim=1)
        x = self.convTrans4(x)

        x = torch.cat((x, out1), dim=1)
        x = self.convTrans5(x)

        x = F.interpolate(x, size=self.out_dim)
        return x
