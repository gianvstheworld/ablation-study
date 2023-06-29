# Descrição: Modelo TravNet (ResNet18 Unet) 
# Importação de modelos tradicionais de redes neurais
from torchvision import models

# Importação de bibliotecas
import torch
import torch.nn as nn
import torch.nn.functional as F

class TravNet_Filters(nn.Module):
    '''
    Classe para o modelo da rede neural TravNet (Modelo de rede neural usando a
    arquitetura ResNet18 como base e adaptando-o para o formato Unet)
    '''
    def __init__(self, params) -> None:
        '''
        Construtor da classe
        :param params: Parâmetros de configuração do modelo
        '''
        super().__init__() # Construtor da classe pai
        model = models.resnet18(pretrained=params.pretrained) # Carrega o modelo ResNet18 pré-treinado

        self.out_dim = (params.output_size[1], params.output_size[0]) # Dimensão da saída

        # RGB encoder - carrega as camadas do modelo ResNet18
        self.block1 = nn.Sequential(*(list(model.children())[:3])) # Cria um novo objeto a partir das três primeiras camadas da rede neural (Conv - BatchNorm - ReLu) 
        self.block2 = nn.Sequential(model.maxpool, model.layer2) # MaxPool e três camadas de Conv - BatchNorm - ReLu
        self.block3 = model.layer3 # Quatro camadas (Conv - BatchNorm - ReLu)
        self.block4 = model.layer4 # Quatro camadas (Conv - BatchNorm - ReLu)
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Esses blocos serão usados para codificar a imagem de entrada em um vetor de características

        # Depth Encoder - Alteração no número de canais de entrada em comparação ao código original
        self.block1_depth = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.block2_depth = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.block3_depth = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.block4_depth = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.block5_depth = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        # Bottleneck - reduz o número de canais de saída do encoder antes da etapa de codidicação (evita overfitting)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=params.bottleneck_dim, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(params.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=params.bottleneck_dim, out_channels=256, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # Decoder - usado para decodificar o vetor de características gerado pelo encoder
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+256, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256+64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128+32, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64+32, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=params.output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(params.output_channels),
            nn.Sigmoid())

    def forward(self, rgb_img, depth_img) -> torch.Tensor:
        '''
        Método forward da rede neural, processa as imagens de entrada e retorna a imagem de saída
        :param rgb_img: Imagem RGB de entrada
        :param depth_img: Imagem de profundidade de entrada
        '''

        # Encoder - capturando contexto da imagem
        out1 = self.block1(rgb_img)
        out1_depth = self.block1_depth(depth_img)
        out1 = out1 + out1_depth
        out2 = self.block2(out1)
        out2_depth = self.block2_depth(out1_depth)
        desired_out2 = (2, 128, 60, 106)
        out2 = F.pad(out2, (0, desired_out2[3] - out2.shape[3], 0, desired_out2[2] - out2.shape[2]))
        out2 = out2 + out2_depth
        out3 = self.block3(out2)
        out3_depth = self.block3_depth(out2_depth)
        out3 = out3 + out3_depth
        out4 = self.block4(out3)
        out4_depth = self.block4_depth(out3_depth)
        out4 = out4 + out4_depth
        print(out4.shape)
        out5 = self.block5(out4)
        out5_depth = self.block5_depth(out4_depth)
        out5 = out5 + out5_depth

        # Bottleneck - reduz o número de canais de saída do encoder antes da etapa de codidicação 
        x = self.bottleneck(out5)

        # Decoder
        x = torch.cat((x, out5), dim=1)
        x = self.convTrans1(x)
        diffY = out4.size()[2] - x.size()[2]
        diffX = out4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, out4), dim=1)
        x = self.convTrans2(x)
        diffY = out3.size()[2] - x.size()[2]
        diffX = out3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, out3), dim=1)
        x = self.convTrans3(x)
        x = torch.cat((x, out2), dim=1)
        x = self.convTrans4(x)
        x = torch.cat((x, out1), dim=1)
        x = self.convTrans5(x)
        
        x = F.interpolate(x, size=self.out_dim)
        return x
