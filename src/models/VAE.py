import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, columns, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(len(columns), 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.columns = columns

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output):
        super(Decoder, self).__init__()
        self.columns = output
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, len(output))

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class LogisticDecoder(nn.Module):
    def __init__(self, latent_dim, output):
        super(LogisticDecoder, self).__init__()
        self.columns = output
        # Single linear layer
        self.linear = nn.Linear(latent_dim, len(output))

    def forward(self, z):
        # Linear combination followed by sigmoid activation
        return torch.sigmoid(self.linear(z))
    
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.columns = decoder.columns

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def embed(self, reactions):
        with torch.no_grad():
            mu, _ = self.encoder(torch.tensor(reactions).float())
        return mu.numpy()

    def predict(self, params, queries=None):
        if queries is None:
            queries = self.columns
        if isinstance(queries, int):
            queries = str(queries)
        if isinstance(queries, str):
            queries = [queries]
        with torch.no_grad():
            Z = self.decoder(torch.tensor(params).float())
        return Z[:,self.columns.isin(queries)].numpy()

    # take transformed coordinates directly (that's why for border we divide by weights)
    def objective(self, params, answers):
        probs = self.predict(params.reshape(-1, 2), answers.index)
        return np.nanmean(np.square(probs - answers.values),axis=1)

def MaskedLoss(recon_x, x, mu, logvar, mask, beta=1):
    MSE = F.mse_loss(recon_x*mask, x*mask, reduction='sum')/mask.mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

def FullLoss(recon_x, x, mu, logvar, **kwargs):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
