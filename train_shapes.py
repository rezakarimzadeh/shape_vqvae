import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pathlib
import nibabel as nib

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 1
        self.num_embeddings = 4
        self.embedding_dim = 32
        self.commitment_cost = 0.25
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.num_epochs = 30
        self.print_every = 10
        self.save_every = 1000
        self.val_split = 0.1
        # self.data_path = '../com_regressor/dataset/labelsTr'
        self.data_path = '../panorama_dataset/cropped_based_on_pancreas/pancreas_labels/'


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self._in_channels = in_channels
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._embedding_dim = embedding_dim

        self._conv_1 = nn.Conv3d(in_channels=self._in_channels,
                                 out_channels=self._num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 = nn.Conv3d(in_channels=self._num_hiddens // 2,
                                 out_channels=self._num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_3 = nn.Conv3d(in_channels=self._num_hiddens,
                                 out_channels=self._num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = nn.ModuleList([
            ResidualBlock(self._num_hiddens, self._num_residual_hiddens, self._num_hiddens)
            for _ in range(self._num_residual_layers)
        ])

        self._conv_to_embedding = nn.Conv3d(in_channels=self._num_hiddens,
                                            out_channels=self._embedding_dim,
                                            kernel_size=1,
                                            stride=1)


    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))

        for layer in self._residual_stack:
            x = layer(x)

        x = F.relu(x)
        return self._conv_to_embedding(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, out_channels):
        super(ResidualBlock, self).__init__()
        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1, padding=1)
        self._conv_2 = nn.Conv3d(in_channels=num_hiddens,
                                 out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = F.relu(self._conv_1(x))
        h = self._conv_2(h)
        return x + h

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._out_channels = out_channels

        self._conv_from_embedding = nn.Conv3d(in_channels=self._embedding_dim,
                                              out_channels=self._num_hiddens,
                                              kernel_size=3,
                                              stride=1, padding=1)

        self._residual_stack = nn.ModuleList([
            ResidualBlock(self._num_hiddens, self._num_residual_hiddens, self._num_hiddens)
            for _ in range(self._num_residual_layers)
        ])

        self._conv_trans_1 = nn.ConvTranspose3d(in_channels=self._num_hiddens,
                                                out_channels=self._num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose3d(in_channels=self._num_hiddens // 2,
                                                out_channels=self._out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self._conv_from_embedding(x))

        for layer in self._residual_stack:
            x = layer(x)

        x = F.relu(x)
        x = F.relu(self._conv_trans_1(x))
        x_reconstructed = self._conv_trans_2(x)
        return x_reconstructed


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        # Initialize the codebook (embeddings)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        # Initialize weights with a uniform distribution (common practice)
        self._embedding.weight.data.uniform_(-1./self._num_embeddings, 1./self._num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, C_emb, D, H, W) where C_emb is embedding_dim
        # Permute to (B, D, H, W, C_emb) for easier distance calculation
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input to (B*D*H*W, C_emb)
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances between flattened input and embedding vectors
        # distances = sum((x - y)^2) = sum(x^2) + sum(y^2) - 2*sum(x*y)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Quantize the input by replacing with the closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self._embedding(encoding_indices).view(input_shape)

        vq_loss = F.mse_loss(quantized.detach(), inputs) + self._commitment_cost * F.mse_loss(quantized, inputs.detach())

        # Straight-Through Estimator (STE)
        quantized_for_decoder = inputs + (quantized - inputs).detach()

        # Permute back to (B, C_emb, D, H, W) for the decoder
        quantized_for_decoder = quantized_for_decoder.permute(0, 4, 1, 2, 3).contiguous()

        return quantized_for_decoder, vq_loss, encoding_indices.squeeze()


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.config = config
        self._encoder = Encoder(config.input_channels,
                                num_hiddens=128,
                                num_residual_layers=2,
                                num_residual_hiddens=32,
                                embedding_dim=config.embedding_dim)

        self._vq_layer = VectorQuantizer(config.num_embeddings,
                                         config.embedding_dim,
                                         config.commitment_cost)

        self._decoder = Decoder(config.embedding_dim,
                                num_hiddens=128,
                                num_residual_layers=2,
                                num_residual_hiddens=32,
                                out_channels=config.input_channels)

    def forward(self, x):
        z_e = self._encoder(x)
        quantized_z, vq_loss, perplexity_indices = self._vq_layer(z_e)
        x_reconstructed = self._decoder(quantized_z)
        return x_reconstructed, vq_loss, perplexity_indices

    def get_quantized_indices(self, x):
        with torch.no_grad():
            z_e = self._encoder(x)
            _, _, indices = self._vq_layer(z_e)
        return indices


class PancreasDataset(Dataset):
    def __init__(self, data_path, config=None):
        self.data_path = pathlib.Path(data_path)
        self.files = [f for f in self.data_path.iterdir()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = nib.load(self.files[idx]).get_fdata()
        # x = np.load(self.files[idx])
        return torch.from_numpy(x).unsqueeze(0).float()


def train(config):
    print(f"Using device: {config.device}")

    dataset = PancreasDataset(data_path=config.data_path)
    train_len = int(len(dataset) * (1 - config.val_split))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VQVAE(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    reconstruction_loss_fn = nn.MSELoss(reduction='mean')


    print("Model Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")


    model.train()
    for epoch in range(config.num_epochs):
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        num_batches = 0

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = data_batch.to(config.device)

            optimizer.zero_grad()

            x_reconstructed, vq_loss, _ = model(data_batch)
            recon_loss = reconstruction_loss_fn(x_reconstructed, data_batch)
            total_loss = recon_loss + vq_loss

            total_loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            num_batches += 1

            if (batch_idx + 1) % config.print_every == 0:
                avg_recon_loss = total_recon_loss / num_batches
                avg_vq_loss = total_vq_loss / num_batches
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Avg Recon Loss: {avg_recon_loss:.6f}, Avg VQ Loss: {avg_vq_loss:.6f}, "
                      f"Total Loss: {(avg_recon_loss + avg_vq_loss):.6f}")

                # Reset accumulators
                total_recon_loss = 0.0
                total_vq_loss = 0.0
                num_batches = 0


            if (batch_idx + 1) % config.save_every == 0:
                 torch.save(model.state_dict(), f"vqvae_3d_checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth")
                 print(f"Saved model checkpoint at epoch {epoch+1}, batch {batch_idx+1}")

        with torch.no_grad():
            total_val_loss = 0.0
            for val_batch in val_loader:
                val_batch = val_batch.to(config.device)
                x_reconstructed, vq_loss, _ = model(val_batch)
                recon_loss = reconstruction_loss_fn(x_reconstructed, val_batch)
                total_val_loss += (recon_loss + vq_loss)

            total_val_loss /= len(val_loader)
            print(f'Validation Loss: {total_val_loss.item():.6f}')
        print(f"--- End of Epoch {epoch+1} ---")

    print("Training complete.")
    torch.save(model.state_dict(), "vqvae_3d_final.pth")
    print("Final model saved as vqvae_3d_final.pth")

if __name__ == '__main__':
    config = Config()
    train(config)