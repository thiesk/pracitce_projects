from dataset import fashionMnist
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_model(model, epochs, lr, batch_size, path, log_path):
    '''
    Train the model
    :return:
    '''

    # logging in tensorboard
    train_writer = SummaryWriter(f"{log_path}/train_{lr}")
    val_writer = SummaryWriter(f"{log_path}/conv_net/val_{lr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    device = torch.device(device)

    model.to(device)
    train_loader = DataLoader(fashionMnist(train=True), batch_size=batch_size)
    val_loader = DataLoader(fashionMnist(train=False), batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        epoch = e + 1
        # training
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (img, label) in enumerate(tqdm(train_loader, f"epoch {epoch}")):

            img = img.to(device)
            label = label.to(device)
            # forward pass
            out = model(img)
            loss = loss_fn(out, label)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss and acc
            train_loss += loss.item()
            train_acc += torch.sum((label == out.argmax(dim=1))) / batch_size


        # validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for i, (img, label) in enumerate(tqdm(val_loader, "validation")):
            img = img.to(device)
            label = label.to(device)

            # forward pass
            out = model(img)
            loss = loss_fn(out, label)

            # log loss and acc
            val_loss += loss.item()
            val_acc += torch.sum((label == out.argmax(dim=1))) / batch_size

        # save checkpoints and log stats
        torch.save(model.state_dict(), f"{path}/conv_net_lr{lr}_bs{batch_size}_e{epoch}.pth")
        train_writer.add_scalar("loss", train_loss / len(train_loader), epoch)
        train_writer.add_scalar("accuracy", train_acc / len(train_loader), epoch)
        val_writer.add_scalar("loss", val_loss / len(val_loader), epoch)
        val_writer.add_scalar("accuracy", val_acc / len(val_loader), epoch)

