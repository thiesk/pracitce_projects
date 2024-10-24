import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import lightning as L
from model import Convolution_net
from dataset import fashionMnist

class PL_convnet(L.LightningModule):
    '''
    Pytorch lightning module.
    Apparently this works.
    '''
    def __init__(self):
        super().__init__()
        self.conv_net = Convolution_net()

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self.conv_net(img)
        loss = F.cross_entropy(pred, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.conv_net(img)
        loss = F.cross_entropy(pred, label)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
         img, label = batch
         pred = self.conv_net(img)
         loss = F.cross_entropy(pred, label)
         self.log("test_loss", loss)
         return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

if __name__ == '__main__':
    # initialize and split data
    fashion_mnist = fashionMnist()
    seed = torch.Generator().manual_seed(42)
    train_size = int(len(fashion_mnist)*.8)

    train_set, val_set = data.random_split(fashion_mnist, [train_size, len(fashion_mnist)- train_size], generator=seed)
    test_set = fashionMnist(train=False)

    train_loader, val_loader, test_loader = DataLoader(train_set), DataLoader(val_set), DataLoader(test_set)

    # make model
    conv_net = PL_convnet()

    # training
    trainer = L.Trainer()
    trainer.fit(conv_net, train_loader, val_loader)
    trainer.test(conv_net, test_set)


