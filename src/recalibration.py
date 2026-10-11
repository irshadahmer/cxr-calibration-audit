import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def fit_temperature(logits, labels, max_iter=50, lr=0.01):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t).squeeze(), labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    T = scaler.temperature.item()
    calibrated_probs = 1.0 / (1.0 + np.exp(-(logits / T)))
    return T, calibrated_probs

def fit_platt(logits, labels):
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(logits.reshape(-1, 1), labels)
    calibrated_probs = clf.predict_proba(logits.reshape(-1, 1))[:, 1]
    return clf, calibrated_probs
