import shutil
import os
from model import Convolution_net
from training import train_model
import argparse


def main(model, lr, epochs, batch_size, name=False):

    # set up checkpoint and logging paths
    if name:
        path = f"./checkpoints/{name}/"
        log_path = f"./logs/{name}_lr={lr}_batch_size={batch_size}/"
    else:
        path = f"./checkpoints/{model}/"
        log_path = f"./logs/{model}_lr={lr}_batch_size={batch_size}/"
    os.makedirs(path, exist_ok=True)

    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)

    # select model
    models = {"conv_net": Convolution_net()}
    model = models[model]

    # run training
    train_model(model=model, epochs=epochs, lr = lr, batch_size=batch_size, path=path, log_path=log_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='conv_net')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--name', type=str, help='set name for run')
    args = parser.parse_args()

    main(
        model=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        name=args.name
    )