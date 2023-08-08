import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# Set device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Hyperparameters
latent_size = 100
hidden_size = 256
image_size = 3 * 64 * 64 # assuming images are RGB and of size 64x64
batch_size = 64
num_epochs = 100
learning_rate = 0.0002

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

# Load custom dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
custom_data = CustomDataset(root_dir='./dataset/covid19/train/Covid', transform=transform)
data_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)

# Discriminator network
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
).to(device)

# Generator network
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Tensorboard writer
writer = SummaryWriter()

# Train GAN
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Adversarial ground truth
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Training discriminator
        images = images.reshape(batch_size, -1).to(device)
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Training generator
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print losses
        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

        # Log losses to Tensorboard
        writer.add_scalar('Loss/discriminator', d_loss.item(), epoch * total_step + i)
        writer.add_scalar('Loss/generator', g_loss.item(), epoch * total_step + i)
        writer.add_scalar('Score/real', real_score.mean().item(), epoch * total_step + i)
        writer.add_scalar('Score/fake', fake_score.mean().item(), epoch * total_step + i)

        # Save generated images to Tensorboard
        if (i+1) % 1000 == 0:
            with torch.no_grad():
                z = torch.randn(batch_size, latent_size).to(device)
                fake_images = G(z)
                fake_images = fake_images.reshape(-1, 1, 28, 28)
                writer.add_images('Generated Images', fake_images, epoch * total_step + i)

# Save models
torch.save(D.state_dict(), 'discriminator.pth')
torch.save(G.state_dict(), 'generator.pth')
