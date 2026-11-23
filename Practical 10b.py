import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__, "| device:", device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
print("Training images:", len(mnist), "| batches per epoch:", len(loader))

imgs, labels = next(iter(loader))
grid = torchvision.utils.make_grid(imgs[:32], nrow=16, normalize=True)
plt.figure(figsize=(12, 2)); plt.imshow(grid.permute(1, 2, 0)); plt.axis("off")
plt.title("Real MNIST digits"); plt.show()

NZ, NCLASS = 64, 10

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NCLASS, 16)
        self.fc = nn.Sequential(nn.Linear(NZ + 16, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.label_emb(y)], 1)).view(-1, 128, 7, 7)
        return self.net(h)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NCLASS, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, self.label_emb(y).view(-1, 1, 28, 28)], 1)).squeeze(1)

G, D = Generator().to(device), Discriminator().to(device)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)

G.apply(init_weights); D.apply(init_weights)
print("Generator parameters:", sum(p.numel() for p in G.parameters()))
print("Discriminator parameters:", sum(p.numel() for p in D.parameters()))

criterion = nn.BCEWithLogitsLoss()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
fixed_z = torch.randn(100, NZ, device=device)
fixed_y = torch.arange(10, device=device).repeat_interleave(10)

def show_grid(title):
    G.eval()
    with torch.no_grad():
        g = torchvision.utils.make_grid(G(fixed_z, fixed_y).cpu(), nrow=10, normalize=True)
    G.train()
    plt.figure(figsize=(6, 6)); plt.imshow(g.permute(1, 2, 0)); plt.axis("off"); plt.title(title); plt.show()

EPOCHS = 6
lossG_hist, lossD_hist = [], []
for epoch in range(1, EPOCHS + 1):
    t = time.time(); lg, ld = [], []
    for real, y in loader:
        real, y = real.to(device), y.to(device); b = real.size(0)
        z = torch.randn(b, NZ, device=device)
        fake = G(z, y)
        lossD = (criterion(D(real, y), torch.full((b,), 0.9, device=device))
                 + criterion(D(fake.detach(), y), torch.zeros(b, device=device)))
        optD.zero_grad(); lossD.backward(); optD.step()
        lossG = criterion(D(fake, y), torch.ones(b, device=device))
        optG.zero_grad(); lossG.backward(); optG.step()
        lg.append(lossG.item()); ld.append(lossD.item())
    lossG_hist.append(np.mean(lg)); lossD_hist.append(np.mean(ld))
    print(f"Epoch {epoch}/{EPOCHS} loss_D {lossD_hist[-1]:.4f} loss_G {lossG_hist[-1]:.4f} ({time.time() - t:.0f}s)")
    if epoch in (1, EPOCHS):
        show_grid(f"Generated digits after epoch {epoch} (row = requested digit)")

plt.figure(figsize=(7, 3.5))
plt.plot(range(1, EPOCHS + 1), lossD_hist, "o-", label="Discriminator")
plt.plot(range(1, EPOCHS + 1), lossG_hist, "o-", label="Generator")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("cGAN training losses"); plt.legend(); plt.show()

G.eval()
number = "9876543210"
with torch.no_grad():
    digits = G(torch.randn(len(number), NZ, device=device),
               torch.tensor([int(c) for c in number], device=device)).cpu()
plt.figure(figsize=(10, 1.5))
plt.imshow(torchvision.utils.make_grid(digits, nrow=len(number), normalize=True).permute(1, 2, 0))
plt.axis("off"); plt.title(f"Generated on request: {number}"); plt.show()

with torch.no_grad():
    z0, z1 = torch.randn(1, NZ, device=device), torch.randn(1, NZ, device=device)
    alphas = torch.linspace(0, 1, 10, device=device).view(-1, 1)
    z = (1 - alphas) * z0 + alphas * z1
    rows = [G(z, torch.full((10,), d, device=device)).cpu() for d in (3, 7)]
plt.figure(figsize=(10, 2.2))
plt.imshow(torchvision.utils.make_grid(torch.cat(rows), nrow=10, normalize=True).permute(1, 2, 0))
plt.axis("off")
plt.title("Latent interpolation z0 -> z1 for digits 3 and 7 (style changes, identity kept)"); plt.show()

torch.save(G.state_dict(), "cgan_mnist_generator.pt")
print("Generator weights saved to cgan_mnist_generator.pt")
