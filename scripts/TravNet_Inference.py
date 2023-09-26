import torch
import time

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import sys

sys.path.append('../src')

from models.TravNet import TravNet
from utils.Dataloader_Inference import TravNetDataset

class Object(object):
    pass

params = Object() # Cria um objeto para armazenar os parâmetros
# Parametros do modelo 
params.pretrained = True
params.load_network_path = None 
params.input_size       = (424, 240)
params.output_size      = (424, 240)
params.output_channels  = 1
params.bottleneck_dim   = 256

# Carregando o modelo e passando os parâmetros 
model = TravNet(params=params)
state_dict = torch.load('/home/gian/Documentos/Códigos/IC/ablation-study/scripts/checkpoints/prechanges_25/best_predictor_depth.pth')
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "")  # Remove o prefixo "module."
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# Carregando a imagem
rgb = Image.open('../../data/rgb/rgb_ts_2021_11_09_16h15m31s_000000.tif')
depth = Image.open('../../data/depth/depth_ts_2021_11_09_16h15m31s_000000.tif')

# Transformações
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

dataset = TravNetDataset(rgb, depth, transform=transform)

# Carregando o dataset
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Obtendo os dados
rgb, depth = next(iter(train_loader))

# Envie a imagem para a GPU, se disponível
if torch.cuda.is_available():
    model = model.cuda()

# Calculando o tempo de inferência
start = time.time()
with torch.no_grad():
    output = model(rgb.cuda().float(), depth.cuda().float())

    pred = output.squeeze().cpu().numpy()
end = time.time()

# Exibindo a imagem
plt.imshow(pred, cmap='gray')
plt.colorbar() 
plt.show()

print('Tempo de inferência: {:.4f} segundos'.format(end - start))
