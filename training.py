import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, generator, discriminator, loader, noise_dim=100, lr=0.001):
        self.loader = loader
        self.batch_size = loader.batch_size
        self.noise_dim = noise_dim
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.g = generator.to(self.device)
        self.d = discriminator.to(self.device)
        self.g_opt = optim.Adam(self.g.parameters(), lr=lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=lr)
        self.sample_noise = self.get_noise(16).to(self.device)

    def get_noise(self, count):
        return torch.randn((count, self.noise_dim)).to(self.device)

    def train_d(self, real_images):
        self.d.zero_grad()

        noise = self.get_noise(self.batch_size)
        with torch.no_grad():
            fake_images = self.g(noise)

        fake_labels = self.d(fake_images)
        real_labels = self.d(real_images)

        d_loss = self.criterion(fake_labels, torch.ones_like(fake_labels)) * 0.5
        d_loss += self.criterion(real_labels, torch.zeros_like(real_labels)) * 0.5
        d_loss.backward()
        self.d_opt.step()
        return d_loss.item()

    def train_g(self):
        self.g.zero_grad()

        noise = self.get_noise(self.batch_size)
        fake_images = self.g(noise)
        fake_labels = self.d(fake_images)

        g_loss = self.criterion(fake_labels, torch.zeros_like(fake_labels))
        g_loss.backward()
        self.g_opt.step()
        return g_loss.item()

    def generate_images(self, epoch):
        with torch.no_grad():
            fake_images = (self.g(self.sample_noise).cpu().numpy() + 1) / 2
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(fake_images[i, 0], cmap="gray")
        plt.savefig("samples/epoch_{}".format(epoch + 1))

    def train(self, epochs):
        print("Starting training on {}.".format(self.device))
        num_batches = len(self.loader)
        for epoch in range(epochs):
            for i, batch in enumerate(self.loader):
                images = batch[0].to(self.device)
                g_loss = self.train_g()
                d_loss = self.train_d(images)
                print("Epoch: {}/{} Batch: {}/{} g_loss: {:.4f} d_loss: {:.4f}".format(
                    epoch + 1, epochs, i + 1, num_batches, g_loss, d_loss), end="\r")
            self.generate_images(epoch)
