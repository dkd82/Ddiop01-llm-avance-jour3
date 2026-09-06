import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer
import logging


class ResidualBlock(nn.Module):
    # Bloc résiduel : deux couches linéaires + normalisation, avec une "connexion saute" (skip connection)
    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Connexion résiduelle (skip connection) : on ajoute l'entrée à la sortie
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    # Réseau de neurones profond : une couche d'entrée, plusieurs blocs résiduels, une couche de sortie
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # Première couche (couche d'entrée)
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Blocs résiduels
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Couche de sortie (renvoie une seule valeur : le prix transformé)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


# Moyenne et écart-type de la cible (utilisés pour dénormaliser la prédiction)
Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    # Classe d'inférence : configure le vectoriseur + le modèle, charge les poids, puis prédit un prix
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.device = None

        # Graines fixes pour la reproductibilité
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        # Vectoriseur de texte (HashingVectorizer -> 5000 caractéristiques binaires) + modèle + choix du device
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(5000)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")          # GPU NVIDIA
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")           # GPU Apple Silicon
        else:
            self.device = torch.device("cpu")           # Processeur

        logging.info(f"Le réseau de neurones utilise {self.device}")

        self.model.to(self.device)

    def load(self, path):
        # Charge les poids entraînés depuis le fichier .pth
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def inference(self, text):
        # Estime le prix d'un texte : vectorisation -> passage dans le modèle -> dénormalisation
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            # Le modèle prédit log(prix+1) normalisé : on inverse la transformation
            result = torch.exp(pred * Y_STD + Y_MEAN) - 1
            result = result.item()
        return max(0, result)   # On ne renvoie jamais un prix négatif
