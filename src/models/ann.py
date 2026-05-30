"""Redes Neurais Artificiais (MLP) - Ponte PyTorch nativa para GPU."""

import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.dlpack import from_dlpack, to_dlpack
from config import RANDOM_SEED, HYPERPARAMS, DEVICE

class PyTorchMLP(nn.Module):
    """Arquitetura base da Rede Neural em PyTorch."""
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SklearnPyTorchMLP:
    """
    Wrapper que faz o PyTorch se comportar como um modelo do scikit-learn/cuML.
    Utiliza DLPack para transferir dados da matriz CuPy para tensores Torch na VRAM 
    sem custo computacional (Zero-Copy).
    """
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=300, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = DEVICE
        self.model = None

    def _cupy_to_torch(self, cp_array):
        # Transfere ponteiros de memória da VRAM diretamente para o PyTorch
        return from_dlpack(cp_array.toDlpack())

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        
        # Garante o formato CuPy e converte para Tensor Torch
        X_t = self._cupy_to_torch(cp.asarray(X, dtype=cp.float32))
        y_t = self._cupy_to_torch(cp.asarray(y, dtype=cp.float32)).unsqueeze(1)
        
        self.model = PyTorchMLP(X_t.shape[1], self.hidden_layer_sizes).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()
            
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = self._cupy_to_torch(cp.asarray(X, dtype=cp.float32))
        
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).squeeze(1)
            
        # Converter o Tensor de volta para CuPy via DLPack
        probs_cp = cp.from_dlpack(to_dlpack(probs))
        
        # O orquestrador espera duas colunas: probabilidade da classe 0 e da classe 1
        return cp.column_stack((1 - probs_cp, probs_cp))

def build_model():
    params = HYPERPARAMS.get("ANN (MLP)", {}).copy()
    
    # Removemos hiperparâmetros exclusivos do sklearn que não se aplicam ao nosso wrapper base
    params.pop("early_stopping", None)
    params.pop("validation_fraction", None)
        
    return SklearnPyTorchMLP(random_state=RANDOM_SEED, **params)