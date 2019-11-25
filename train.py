from datasets import get_mnist_loader
from models import Generator, Discriminator
from training import ModelTrainer

def main():
    NOISE_DIM = 100
    BATCH_SIZE = 32

    loader = get_mnist_loader(batch_size=BATCH_SIZE)
    gen = Generator(noise_dim=NOISE_DIM)
    disc = Discriminator()
    trainer = ModelTrainer(gen, disc, loader)
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()
