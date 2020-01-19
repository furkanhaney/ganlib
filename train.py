import torch
import argparse
from datasets import get_mnist_loader
from gan import Generator, Discriminator
from training import GanTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="Learning rate for the adam optimizer.", type=float, default=0.0002)
    parser.add_argument("--gen_batchnorm", help="Enables batch_norm for generator layers.", action='store_true')
    parser.add_argument("--gen_dropout", help="Enables dropout for generator layers.", action='store_true')
    parser.add_argument("--disc_batchnorm", help="Enables batch_norm for discriminator layers.", action='store_true')
    parser.add_argument("--disc_dropout", help="Enables dropout for discriminator layers.", action='store_true')
    parser.add_argument("--device", help="Device to train the model on: cpu or gpu", choices=["cpu", "gpu"], default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", help="Number of epochs to train.", type=int, default=250)
    parser.add_argument("--batch_size", help="Size of minibatches.", type=int, default=128)
    parser.add_argument("--num_workers", help="Number of workers for the data loaders.", type=int, default=2)
    #parser.add_argument("--num_layers", help="Number of layers in the model", type=int, default=4)
    parser.add_argument("--network_width", help="Scale up the number of units in each layer", type=float, default=1.0)
    parser.add_argument("--loss", help="Loss function to train.", choices=["bce", "mse"], default="bce")
    parser.add_argument("--noise_dim", help="Number of dimensions for the size vector", type=int, default=100)

    args = parser.parse_args()
    device = torch.device("cuda:0") if args.device == "gpu" else torch.device("cpu")
    loader = get_mnist_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    gen = Generator(noise_dim=args.noise_dim, bn=args.gen_batchnorm, dropout=args.gen_dropout, width=args.network_width)
    disc = Discriminator(bn=args.disc_batchnorm, dropout=args.disc_dropout, width=args.network_width)
    trainer = GanTrainer(gen, disc, loader, device, noise_dim=args.noise_dim, lr=args.lr, loss=args.loss)
    trainer.train(epochs=args.epochs)

if __name__ == "__main__":
    main()
