import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encode_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        
        # Encoder: input -> encode_dim -> encode_dim*2 -> encode_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, encode_dim * 2),
            nn.ReLU(),
            nn.Linear(encode_dim * 2, encode_dim),
            nn.ReLU()
        )
        # Decoder: encoded -> encode_dim -> encode_dim*2 -> output (with sigmoid)
        self.decoder = nn.Sequential(
            nn.Linear(encode_dim, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, encode_dim * 2),
            nn.ReLU(),
            nn.Linear(encode_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def bottleneck(self, x):
        # Return the encoded representation (bottleneck)
        return self.encoder(x)
    
    def fit(self, x_train, epochs=100, batch_size=8, lr=1e-3):
        # Ensure training mode is set
        self.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Prepare dataset and dataloader
        x_train = torch.tensor(x_train, dtype=torch.float32)
        dataset = data.TensorDataset(x_train)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            # Uncomment the next line to print loss per epoch
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.6f}")

def compute_representation(
    train,
    test,
    copula_samples,
    n_components=2,
    rep_type="pca",
    seed=42,
):
    """
    Compute a low-dimensional representation of the data using either PCA or an autoencoder.

    Args:
      train: training data (numpy array)
      test: test data (numpy array)
      copula_samples: copula samples (numpy array)
      n_components: target number of dimensions (default=2)
      rep_type: representation type; either "pca" or "ae"
      seed: random seed (default=42)

    Returns:
      Transformed representations for train, test, and copula_samples.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(train)
    combined_X_train_sc = scaler.transform(train)
    combined_X_test_sc = scaler.transform(test)
    copula_sc = scaler.transform(copula_samples)

    if rep_type == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        pcs_train = pca.fit_transform(combined_X_train_sc)
        pcs_test = pca.transform(combined_X_test_sc)
        pcs_copula = pca.transform(copula_sc)
    elif rep_type == "ae":
        # Train PyTorch autoencoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae = AutoEncoder(input_dim=combined_X_train_sc.shape[1], encode_dim=n_components)
        ae.fit(combined_X_train_sc, epochs=100, batch_size=8)
        ae.eval()
        with torch.no_grad():
            x_train_tensor = torch.tensor(combined_X_train_sc, dtype=torch.float32).to(device)
            x_test_tensor = torch.tensor(combined_X_test_sc, dtype=torch.float32).to(device)
            copula_tensor = torch.tensor(copula_sc, dtype=torch.float32).to(device)
            pcs_train = ae.bottleneck(x_train_tensor).cpu().numpy()
            pcs_test = ae.bottleneck(x_test_tensor).cpu().numpy()
            pcs_copula = ae.bottleneck(copula_tensor).cpu().numpy()
    else:
        raise ValueError("Unknown rep_type. Please choose either 'pca' or 'ae'.")

    return pcs_train, pcs_test, pcs_copula

def representation_class_based(
    train,
    copula_samples,
    n_components=2,
    rep_type="pca",
    seed=42,
):
    """
    Computes a representation of the training and copula samples.
    Currently supports PCA only.

    Args:
      train: training data (numpy array)
      copula_samples: copula samples (numpy array)
      n_components: number of PCA components (default=2)
      rep_type: type of representation; only "pca" is supported (default="pca")
      seed: random seed (default=42)

    Returns:
      Transformed training data, transformed copula samples, the PCA object, and the scaler.
    """
    scaler = StandardScaler()
    scaler.fit(train)
    combined_X_train_sc = scaler.transform(train)
    copula_sc = scaler.transform(copula_samples)

    if rep_type == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        pcs_train = pca.fit_transform(combined_X_train_sc)
        pcs_copula = pca.transform(copula_sc)
    else:
        raise ValueError("Unknown rep_type. Only 'pca' is supported in this function.")

    return pcs_train, pcs_copula, pca, scaler
