import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

class GanTrainer:
    def __init__(self, generator, discriminator, loader, device, noise_dim=100, lr=0.0001, loss="mse"):
        self.loader = loader
        self.batch_size = loader.batch_size
        self.noise_dim = noise_dim
        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "bce":
            self.criterion = nn.BCELoss()
        else:
            raise Exception("Invalid loss function!")
        self.device = device
        self.g = generator.to(self.device)
        self.d = discriminator.to(self.device)
        self.g_opt = optim.Adam(self.g.parameters(), lr=lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=lr)
        self.sample_noise = self.get_noise(16).to(self.device)

    def get_noise(self, count):
        return torch.randn((count, self.noise_dim)).to(self.device)

    def clean_output_folder(self):
        for f in os.listdir("output"):
            os.remove("output/" + f)

    def train_d(self, real_images):
        self.d.zero_grad()

        noise = self.get_noise(self.batch_size)
        with torch.no_grad():
            fake_images = self.g(noise)

        fake_labels = self.d(fake_images)
        real_labels = self.d(real_images)

        d_loss = self.criterion(fake_labels, torch.zeros_like(fake_labels)) * 0.5
        d_loss += self.criterion(real_labels, torch.ones_like(real_labels)) * 0.5
        d_loss.backward()
        self.d_opt.step()
        return d_loss.item()

    def train_g(self):
        self.g.zero_grad()

        noise = self.get_noise(self.batch_size)
        fake_images = self.g(noise)
        fake_labels = self.d(fake_images)

        g_loss = self.criterion(fake_labels, torch.ones_like(fake_labels))
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
        plt.savefig("output/epoch_{}".format(epoch + 1))
        plt.close()

    def generate_charts(self, g_losses, d_losses):
        plt.figure(figsize=(20,10))
        plt.plot(g_losses)
        plt.plot(d_losses)
        plt.savefig("output/0-charts")
        plt.close()

    def calculate_fid(self):
        a = self.get_noise(2048).to(self.device)
        with torch.no_grad():
            fake_images = (self.g(self.sample_noise).cpu().numpy() + 1) / 2

    def train(self, epochs):
        self.clean_output_folder()
        print(self.g)
        print(self.d)
        print("Starting training on {}.".format(self.device))
        num_batches = len(self.loader)
        avg_g_loss = 0
        avg_d_loss = 0
        g_losses, d_losses = [], []
        for epoch in range(epochs):
            pbar = tqdm(self.loader, ascii=True)
            for i, batch in enumerate(pbar):
                images = batch[0].to(self.device)
                g_loss, d_loss = self.train_g(), self.train_d(images)
                avg_g_loss = (g_loss + i * avg_g_loss) / (i + 1)
                avg_d_loss = (d_loss + i * avg_d_loss) / (i + 1)
                pbar.set_description("Epoch: {}/{}".format(
                    epoch + 1, epochs, i + 1, num_batches))
                pbar.set_postfix({
                    "g_loss": "{:.2f}".format(avg_g_loss),
                    "d_loss": "{:.2f}".format(avg_d_loss),
                })
                # print("Epoch: {}/{} Batch: {}/{} g_loss: {:.4f} d_loss: {:.4f}        ".format(
                #     epoch + 1, epochs, i + 1, num_batches, avg_g_loss, avg_d_loss), end="\r")
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            self.generate_charts(g_losses, d_losses)
            self.generate_images(epoch)
            pbar.close()
