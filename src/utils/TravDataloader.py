#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importando bibliotecas
import os
import random

import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class TravNetDataset(Dataset):
    '''
    Classe para carregar os dados do dataset
    '''

    def __init__(self, params, transform) -> None:
        '''
        Construtor da classe
        :param params: parâmetros do dataset
        :param transform: transformação a ser aplicada nas imagens
        '''
        print("Initializing dataset")
        self.root = params.data_path  # Caminho para os dados
        self.transform = transform  # Transformação dos dados
        self.image_size = params.input_size  # Tamanho da entrada
        self.output_size = params.output_size  # Tamanho da saída
        self.depth_mean = params.depth_mean  # Média da profundidade
        self.depth_std = params.depth_std  # Desvio padrão da profundidade
        self.bin_width = 0.2  # Largura da máscara binária

        # Lê as linhas do arquivo csv
        self.data = pd.read_csv(params.csv_path)

        # Prepara os dados e obtém os valores máximos e mínimos
        self.color_fname, self.depth_fname, self.path_fname, self.mu_fname, self.nu_fname = self.prepare(self.data)

        # Prepara os pesos e o número de intervalos definidos 
        self.weights, self.bins = self.prepare_weights()

        # Obtém as estatísticas de profundidade
        self.get_depth_stats()

        # Define se o dataset será pré-processado
        self.preproc = params.preproc

    def __len__(self) -> int:
        '''
        Retorna o tamanho do dataset
        '''
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        '''
        Retorna uma amostra do dataset
        :param idx: índice da amostra
        '''
        # Obtém os índices dos arquivos
        color_fname = self.color_fname[idx]
        depth_fname = self.depth_fname[idx]
        path_fname = self.path_fname[idx]
        mu_fname = self.mu_fname[idx]
        nu_fname = self.nu_fname[idx]

        # Lê as imagens
        color_img = cv2.imread(os.path.join(self.root, color_fname), -1)
        depth_img = cv2.imread(os.path.join(self.root, depth_fname), -1)
        path_img = cv2.imread(os.path.join(self.root, path_fname), -1)
        mu_img = cv2.imread(os.path.join(self.root, mu_fname), -1)
        nu_img = cv2.imread(os.path.join(self.root, nu_fname), -1)

        # Aplica o pré-processamento
        if self.preproc:
            color_img, depth_img, path_img, mu_img, nu_img = self.random_flip(color_img, depth_img, path_img, mu_img, nu_img)

        # Converte a imagem para RGB e redimensiona (PyTorch usa RGB)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, self.image_size, interpolation=cv2.INTER_AREA)
        color_img = self.transform(color_img)

        # Converte a imagem para profundidade e redimensiona
        depth_img = cv2.resize(depth_img, self.image_size, interpolation=cv2.INTER_AREA)
        depth_img = np.uint16(depth_img)
        depth_img = depth_img*10**-3

        # Normaliza a imagem de profundidade
        depth_img = (depth_img-self.depth_mean)/self.depth_std
        depth_img = np.expand_dims(depth_img, axis=2)
        depth_img = np.transpose(depth_img, (2, 0, 1))

        # Redimensiona as imagens de mu e nu
        mu_img = cv2.resize(mu_img, self.output_size, interpolation=cv2.INTER_AREA)
        nu_img = cv2.resize(nu_img, self.output_size, interpolation=cv2.INTER_AREA)
        path_img = cv2.resize(path_img, self.output_size, interpolation=cv2.INTER_AREA)

        # Expande as dimensões das imagens
        mu_img = np.expand_dims(mu_img, 0)
        nu_img = np.expand_dims(nu_img, 0)
        path_img = np.expand_dims(path_img, 0)

        # Normaliza as imagens
        mu_img = mu_img/255.0
        nu_img = nu_img/255.0
        path_img = (path_img/255.0).astype(bool)

        # Obtém os pesos
        weight_idxs = np.digitize(mu_img, self.bins[:-1]) - 1
        weight = self.weights[weight_idxs] * path_img

        return color_img, depth_img, path_img, mu_img, nu_img, weight

    def prepare(self, data: tuple) -> tuple:
        '''
        Prepara os dados
        :param data: dados
        '''
        # Listas para armazenar os nomes dos arquivos
        color_fname_list = []
        depth_fname_list = []
        path_fname_list = []
        mu_fname_list = []
        nu_fname_list = []

        # Itera sobre os dados
        for color_fname, depth_fname, path_fname, mu_fname, nu_fname, _, _, _, _, _, _ in data.iloc:
            # Adiciona os nomes dos arquivos às listas
            color_fname_list.append(color_fname)
            depth_fname_list.append(depth_fname)
            path_fname_list.append(path_fname)
            mu_fname_list.append(mu_fname)
            nu_fname_list.append(nu_fname)

        return color_fname_list, depth_fname_list, path_fname_list, mu_fname_list, nu_fname_list

    def random_flip(self, color_img, depth_img, path_img, mu_img, nu_img) -> tuple:
        '''
        Aplica o flip aleatório
        :param color_img: imagem colorida
        :param depth_img: imagem de profundidade
        :param path_img: imagem de caminho
        :param mu_img: imagem de mu
        :param nu_img: imagem de nu
        '''

        # Aplica o flip aleatório
        if random.random() < 0.5:
            color_img_lr = np.fliplr(color_img).copy()
            depth_img_lr = np.fliplr(depth_img).copy()
            path_img_lr = np.fliplr(path_img).copy()
            mu_img_lr = np.fliplr(mu_img).copy()
            nu_img_lr = np.fliplr(nu_img).copy()
            return color_img_lr, depth_img_lr, path_img_lr, mu_img_lr, nu_img_lr

        return color_img, depth_img, path_img, mu_img, nu_img

    def prepare_weights(self) -> tuple:
        '''
        Prepara os pesos
        '''

        # Lista para armazenar os pesos
        labels_data = []

        # Itera sobre os nomes dos arquivos
        for idx in range(len(self.mu_fname)):
            # Obtém os nomes dos arquivos
            mu_fname = self.mu_fname[idx]
            path_fname = self.path_fname[idx]

            # Carrega as imagens
            mu_img = cv2.imread(os.path.join(self.root, mu_fname), -1)
            mu_img = cv2.resize(mu_img, self.output_size, interpolation=cv2.INTER_AREA)
            mu_img = mu_img/255.0

            path_img = cv2.imread(os.path.join(self.root, path_fname), -1)
            path_img = cv2.resize(path_img, self.output_size, interpolation=cv2.INTER_AREA)
            path_img = (path_img/255.0).astype(bool)

            data_image = mu_img[path_img]
            labels_data.extend(data_image.flatten().tolist())

        # Faz o histograma de pesos
        values, bins = np.histogram(labels_data, bins=int(1/self.bin_width), range=(0, 1), density=True)

        return (1-values*self.bin_width), bins

    def get_depth_stats(self) -> None:
        '''
        Obtém as estatísticas da imagem de profundidade
        '''
        
        psum = 0.0 
        psum_sq = 0.0

        for idx in range(len(self.depth_fname)):
            depth_fname = self.depth_fname[idx]

            depth_img = cv2.imread(os.path.join(self.root, depth_fname), -1)
            depth_img = depth_img*1e-3
            psum += np.sum(depth_img)
            psum_sq += np.sum(depth_img**2)

        count = len(self.depth_fname)*depth_img.shape[0]*depth_img.shape[1]
        total_mean = psum/count
        total_std = np.sqrt(psum_sq / count - (total_mean ** 2))

        print('Max score:', np.max(depth_img))
        print('Min score:', np.min(depth_img))
        print('Depth mean:', total_mean)
        print('Depth std:', total_std) 
