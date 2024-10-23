import torchvision
import seaborn as isns
from torch.utils.data import Dataset

def get_transform(split='train'):
    '''
    Make data transforms.
    :param split: split of what split the transforms are for
    :return:
    '''
    return None


class fashionMnist(Dataset):
    def __init__(self, ):
        self.transform = get_transform()
        self.data = torchvision.datasets.FashionMNIST(root="/data", download=True, transform=get_transform())

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]