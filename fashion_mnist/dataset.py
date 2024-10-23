import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

def get_transform(split='train'):
    '''
    Make data transforms.
    :param split: split of what split the transforms are for
    :return:
    '''
    aug = transforms.Compose([
        transforms.ToTensor(),
    ])
    return aug


class fashionMnist(Dataset):
    def __init__(self, train=True):
        self.transform = get_transform()
        self.data = torchvision.datasets.FashionMNIST(root="/data", train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]